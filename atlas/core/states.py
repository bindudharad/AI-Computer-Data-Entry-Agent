"""Agent state machine.

The floating assistant and the workflow loop both render this state. States map
to the full lifecycle: idle -> waiting for attach -> watching -> analyzing ->
planning -> acting (typing/clicking/scrolling) -> verifying -> done, with
pause/stop/error/recovery as cross-cutting states.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum


class AgentState(str, Enum):
    """All observable states of the agent."""

    IDLE = "idle"
    WAITING_ATTACH = "waiting_attach"  # waiting for the user to click a field
    ATTACHING = "attaching"
    INSPECTING_UI = "inspecting_ui"  # inspecting the real window's UIA tree (Step 2)
    WAITING_FOR_START_FIELD = "waiting_for_start_field"  # waiting for the user to click the first form field
    BUILD_UI_TREE = "build_ui_tree"  # enumerating the window's UIA tree + panels
    BUILDING_TREE = "building_tree"  # alias for BUILD_UI_TREE (Step 8 dashboard state)
    SCREEN_MODEL = "screen_model"  # turning UI tree + panel crops into a screen model
    RECORD_EXTRACTION = "record_extraction"  # building a Record from the left panel
    READING_RECORD = "reading_record"  # alias for RECORD_EXTRACTION (Step 8 dashboard state)
    FIELD_MAPPING = "field_mapping"  # building the UIA field map from the start control
    MAPPING_FIELDS = "mapping_fields"  # alias for FIELD_MAPPING (Step 8 dashboard state)
    WATCHING = "watching"
    OBSERVING = "observing"  # actively observing the screen
    UNDERSTANDING = "understanding"  # parsing source record + fields
    ANALYZING = "analyzing"
    PLANNING = "planning"
    THINKING = "thinking"
    TYPING = "typing"
    CLICKING = "clicking"
    SCROLLING = "scrolling"
    WAITING = "waiting"  # waiting for a new record
    WAITING_NEXT_RECORD = "waiting_next_record"  # alias for WAITING (Step 8 dashboard state)
    VERIFYING = "verifying"
    UPLOADING = "uploading"  # submitting/uploading the current record
    COMPLETED = "completed"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERY = "recovery"


#: User-initiated states that may be entered from any running state.
_INTERRUPT_STATES = {AgentState.PAUSED, AgentState.STOPPED}

#: States in which the agent is actively executing work.
_ACTIVE_STATES = {
    AgentState.ATTACHING,
    AgentState.INSPECTING_UI,
    AgentState.WAITING_FOR_START_FIELD,
    AgentState.BUILD_UI_TREE,
    AgentState.BUILDING_TREE,
    AgentState.SCREEN_MODEL,
    AgentState.RECORD_EXTRACTION,
    AgentState.READING_RECORD,
    AgentState.FIELD_MAPPING,
    AgentState.MAPPING_FIELDS,
    AgentState.WATCHING,
    AgentState.OBSERVING,
    AgentState.UNDERSTANDING,
    AgentState.ANALYZING,
    AgentState.PLANNING,
    AgentState.THINKING,
    AgentState.TYPING,
    AgentState.CLICKING,
    AgentState.SCROLLING,
    AgentState.UPLOADING,
    AgentState.WAITING,
    AgentState.WAITING_NEXT_RECORD,
    AgentState.VERIFYING,
    AgentState.RECOVERY,
}


class InvalidTransitionError(RuntimeError):
    """Raised when a state transition is not allowed."""


class StateMachine:
    """Small, explicit transition state machine.

    Valid transitions:

    IDLE -> WAITING_ATTACH -> ATTACHING
        -> {WAITING_FOR_START_FIELD -> FIELD_MAPPING, BUILD_UI_TREE}
        -> SCREEN_MODEL -> RECORD_EXTRACTION -> FIELD_MAPPING -> WATCHING
        -> ANALYZING -> PLANNING
        -> THINKING -> {TYPING, CLICKING, SCROLLING} -> VERIFYING -> WATCHING
    WATCHING/VERIFYING -> COMPLETED -> WAITING -> {WATCHING, SCREEN_MODEL, RECORD_EXTRACTION}
    <active> -> PAUSED -> <prior active>
    <any running> -> STOPPED (terminal)
    <any> -> ERROR -> <recovery/active/stopped>
    """

    def __init__(self, initial: AgentState = AgentState.IDLE) -> None:
        self._state = initial
        self._on_change: list[Callable[[AgentState], None]] = []
        self._resume_state: AgentState | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    def is_active(self) -> bool:
        return self._state in _ACTIVE_STATES

    def can_pause(self) -> bool:
        return self.is_active() or self._state == AgentState.COMPLETED

    def on_change(self, listener: Callable[[AgentState], None]) -> None:
        """Register a listener notified with the new state on every change."""
        self._on_change.append(listener)

    def transition(self, new_state: AgentState) -> AgentState:
        """Attempt a transition, enforcing the transition table."""
        current = self._state
        if new_state == current:
            return current
        if not self._is_valid(current, new_state):
            raise InvalidTransitionError(f"{current.value} -> {new_state.value} not allowed")
        if new_state in _INTERRUPT_STATES:
            if current in _ACTIVE_STATES or current == AgentState.COMPLETED:
                self._resume_state = current
            else:
                self._resume_state = None
        elif current in _INTERRUPT_STATES and new_state == AgentState.PAUSED:
            pass  # resume handled below
        self._state = new_state
        for listener in self._on_change:
            try:
                listener(new_state)
            except Exception:
                pass
        return self._state

    def force(self, state: AgentState) -> AgentState:
        """Set the state directly, bypassing the transition table.

        Used only by the workflow loop to recover its position after a reset or
        an unexpected error; never for user-initiated transitions.
        """
        self._state = state
        self._resume_state = None
        for listener in self._on_change:
            try:
                listener(state)
            except Exception:
                pass
        return self._state

    def resume(self) -> AgentState:
        """Return to the state before pause."""
        if self._state == AgentState.PAUSED:
            target = self._resume_state or AgentState.WATCHING
            self._resume_state = None
            return self.force(target)
        return self._state

    def reset(self) -> None:
        self._resume_state = None
        self.force(AgentState.IDLE)

    # Alias states are interchangeable with their base states for transitions.
    _STATE_ALIASES: dict[AgentState, AgentState] = {
        AgentState.BUILDING_TREE: AgentState.BUILD_UI_TREE,
        AgentState.READING_RECORD: AgentState.RECORD_EXTRACTION,
        AgentState.MAPPING_FIELDS: AgentState.FIELD_MAPPING,
        AgentState.WAITING_NEXT_RECORD: AgentState.WAITING,
    }
    _ALIAS_TO_BASE: dict[AgentState, AgentState] = {}
    for _alias, _base in _STATE_ALIASES.items():
        _ALIAS_TO_BASE[_alias] = _base
        _ALIAS_TO_BASE[_base] = _base
    del _alias, _base

    @staticmethod
    def _normalize(state: AgentState) -> AgentState:
        """Map alias states to their base state for transition checks."""
        return StateMachine._ALIAS_TO_BASE.get(state, state)

    @staticmethod
    def _is_valid(current: AgentState, new_state: AgentState) -> bool:
        if new_state in _INTERRUPT_STATES:
            return True  # pause/stop always allowed
        # Normalize alias states to their base for transition checks.
        current = StateMachine._normalize(current)
        new_state = StateMachine._normalize(new_state)
        if current == AgentState.IDLE:
            return new_state == AgentState.WAITING_ATTACH
        if current == AgentState.WAITING_ATTACH:
            return new_state in {AgentState.ATTACHING, AgentState.ERROR, AgentState.IDLE}
        if current == AgentState.ATTACHING:
            return new_state in {
                AgentState.INSPECTING_UI,
                AgentState.WAITING_FOR_START_FIELD,
                AgentState.BUILD_UI_TREE,
                AgentState.WATCHING,
                AgentState.ERROR,
                AgentState.IDLE,
            }
        if current == AgentState.INSPECTING_UI:
            return new_state in {
                AgentState.WAITING_FOR_START_FIELD,
                AgentState.BUILD_UI_TREE,
                AgentState.WATCHING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.IDLE,
            }
        if current == AgentState.WAITING_FOR_START_FIELD:
            return new_state in {AgentState.FIELD_MAPPING, AgentState.ERROR, AgentState.IDLE}
        if current == AgentState.BUILD_UI_TREE:
            return new_state in {
                AgentState.SCREEN_MODEL,
                AgentState.WATCHING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.IDLE,
            }
        if current == AgentState.SCREEN_MODEL:
            return new_state in {
                AgentState.RECORD_EXTRACTION,
                AgentState.WATCHING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.IDLE,
            }
        if current == AgentState.RECORD_EXTRACTION:
            return new_state in {
                AgentState.FIELD_MAPPING,
                AgentState.PLANNING,
                AgentState.WATCHING,
                AgentState.WAITING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.IDLE,
            }
        if current == AgentState.FIELD_MAPPING:
            return new_state in {
                AgentState.WATCHING,
                AgentState.PLANNING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.IDLE,
            }
        if current == AgentState.WATCHING:
            return new_state in {
                AgentState.OBSERVING,
                AgentState.ANALYZING,
                AgentState.BUILD_UI_TREE,
                AgentState.SCREEN_MODEL,
                AgentState.RECORD_EXTRACTION,
                AgentState.COMPLETED,
                AgentState.WAITING,
                AgentState.ERROR,
            }
        if current == AgentState.OBSERVING:
            return new_state in {AgentState.UNDERSTANDING, AgentState.ANALYZING, AgentState.ERROR, AgentState.RECOVERY}
        if current == AgentState.UNDERSTANDING:
            return new_state in {AgentState.PLANNING, AgentState.ANALYZING, AgentState.ERROR, AgentState.RECOVERY}
        if current == AgentState.ANALYZING:
            return new_state in {AgentState.PLANNING, AgentState.UNDERSTANDING, AgentState.ERROR, AgentState.RECOVERY}
        if current == AgentState.PLANNING:
            return new_state in {
                AgentState.THINKING,
                AgentState.ERROR,
                AgentState.RECOVERY,
            }
        if current == AgentState.THINKING:
            return new_state in {
                AgentState.TYPING,
                AgentState.CLICKING,
                AgentState.SCROLLING,
                AgentState.UPLOADING,
                AgentState.ERROR,
                AgentState.RECOVERY,
            }
        if current in {AgentState.TYPING, AgentState.CLICKING, AgentState.SCROLLING, AgentState.UPLOADING}:
            return new_state in {
                AgentState.VERIFYING,
                AgentState.ERROR,
                AgentState.RECOVERY,
                AgentState.WATCHING,
            }
        if current == AgentState.VERIFYING:
            return new_state in {
                AgentState.WATCHING,
                AgentState.COMPLETED,
                AgentState.ERROR,
                AgentState.RECOVERY,
            }
        if current == AgentState.COMPLETED:
            return new_state in {AgentState.WAITING, AgentState.WATCHING, AgentState.IDLE}
        if current == AgentState.WAITING:
            return new_state in {
                AgentState.WATCHING,
                AgentState.OBSERVING,
                AgentState.SCREEN_MODEL,
                AgentState.RECORD_EXTRACTION,
                AgentState.ERROR,
                AgentState.STOPPED,
            }
        if current == AgentState.ERROR:
            return new_state in {
                AgentState.RECOVERY,
                AgentState.WATCHING,
                AgentState.IDLE,
                AgentState.STOPPED,
            }
        if current == AgentState.RECOVERY:
            return new_state in {
                AgentState.WATCHING,
                AgentState.OBSERVING,
                AgentState.ANALYZING,
                AgentState.BUILD_UI_TREE,
                AgentState.SCREEN_MODEL,
                AgentState.RECORD_EXTRACTION,
                AgentState.IDLE,
                AgentState.ERROR,
            }
        return False


def states_prioritized() -> Iterable[AgentState]:
    """Ordered states for display in debug UIs."""
    return list(AgentState)
