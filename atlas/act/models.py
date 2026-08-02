"""Action model shared by the planner, executor and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from atlas.vision.models import BBox


class ActionType(str, Enum):
    """All actions the agent can execute."""

    MOVE_MOUSE = "move_mouse"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    SCROLL = "scroll"
    TYPE = "type"
    CLEAR = "clear"
    SELECT = "select"
    CHOOSE_DATE = "choose_date"
    OPEN_DROPDOWN = "open_dropdown"
    TOGGLE = "toggle"  # checkbox / radio
    TAB = "tab"
    PRESS_ENTER = "press_enter"
    PRESS_ESCAPE = "press_escape"
    PASTE = "paste"
    UPLOAD_FILE = "upload_file"
    CTRL_A = "ctrl_a"
    WAIT = "wait"
    VERIFY = "verify"
    SUBMIT = "submit"
    CAPTURE = "capture"
    ANALYZE = "analyze"
    STOP = "stop"


@dataclass
class Action:
    """A single planned action.

    ``bbox`` is in absolute screen coordinates (client area + screen offset)
    so the executor can act on it directly.
    """

    type: ActionType
    reason: str = ""
    field_id: str | None = None
    value: str | None = None
    bbox: BBox | None = None
    confidence: float = 1.0
    options: list[str] = field(default_factory=list)
    wait_seconds: float = 0.5
    scroll_amount: int = 3
    expected: str | None = None  # expected observable value after the action
    max_retries: int | None = None  # per-action retry budget (overrides default)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "reason": self.reason,
            "field_id": self.field_id,
            "value": self.value,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "options": list(self.options),
            "wait_seconds": self.wait_seconds,
            "scroll_amount": self.scroll_amount,
            "expected": self.expected,
            "max_retries": self.max_retries,
        }


@dataclass
class ActionResult:
    """Outcome of executing one action."""

    action: Action
    success: bool
    verified: bool = False
    message: str = ""
    retries: int = 0
    duration_ms: float = 0.0
    verification_evidence: str = ""

    @property
    def ok(self) -> bool:
        return self.success and (self.verified or not self._requires_verification)

    @property
    def _requires_verification(self) -> bool:
        return self.action.type in {
            ActionType.TYPE,
            ActionType.SELECT,
            ActionType.TOGGLE,
            ActionType.CHOOSE_DATE,
            ActionType.CLEAR,
        }

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "verified": self.verified,
            "message": self.message,
            "retries": self.retries,
            "duration_ms": self.duration_ms,
            "verification_evidence": self.verification_evidence,
        }


#: Actions that require explicit verification after execution.
VERIFYABLE_ACTIONS = {
    ActionType.TYPE,
    ActionType.SELECT,
    ActionType.TOGGLE,
    ActionType.CHOOSE_DATE,
    ActionType.CLEAR,
    ActionType.PASTE,
    ActionType.UPLOAD_FILE,
}


__all__ = ["Action", "ActionType", "ActionResult", "VERIFYABLE_ACTIONS"]
