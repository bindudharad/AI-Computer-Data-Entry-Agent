"""Tests for the action executor (execution + verification + recovery)."""

from __future__ import annotations

from atlas.act.controls import ControlInterface, ControlOutcome
from atlas.act.executor import ActionExecutor
from atlas.act.models import Action, ActionType
from atlas.act.verify import FieldVerifier
from atlas.core.events import EventType, get_event_bus
from atlas.mapping.mapper import FieldMapping, MappingResult
from atlas.reason.planner import ActionPlanner
from atlas.reason.recovery import RecoveryPlanner
from atlas.understanding.fields import EditableField
from atlas.understanding.source import SourceRecord
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement


class FakeControls(ControlInterface):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def _record(self, name: str, field_id: str | None, value: str | None = None) -> ControlOutcome:
        self.calls.append((name, field_id, value))
        return ControlOutcome(ok=True, evidence=f"fake {name}")

    def focus(self, bbox, field_id=None): return self._record("focus", field_id)
    def click_field(self, bbox, field_id=None): return self._record("click", field_id)
    def type_value(self, bbox, value, field_id=None): return self._record("type", field_id, value)
    def clear(self, bbox, field_id=None): return self._record("clear", field_id)
    def select_option(self, bbox, value, options=None, field_id=None): return self._record("select", field_id, value)
    def toggle(self, bbox, value, field_id=None): return self._record("toggle", field_id, value)
    def choose_date(self, bbox, value, date_format=None, field_id=None): return self._record("date", field_id, value)
    def press_tab(self): return ControlOutcome(ok=True)
    def press_enter(self): return ControlOutcome(ok=True)
    def press_escape(self): return ControlOutcome(ok=True)
    def scroll(self, direction, amount=3): return ControlOutcome(ok=True)
    def paste(self, value, field_id=None): return self._record("paste", field_id, value)


class StubMouse:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    def move_to(self, x, y): pass
    def click(self, x, y): self.clicks.append((x, y))
    def double_click(self, x, y): pass
    def right_click(self, x, y): pass
    def hover(self, x, y): pass
    def scroll(self, direction, amount=3): pass


class StubKeyboard:
    driver = None


class AlwaysPassVerifier(FieldVerifier):
    name = "always-pass"

    def verify(self, bbox, expected, field_id=None):
        return True, "ok"


class AlwaysFailVerifier(FieldVerifier):
    name = "always-fail"

    def verify(self, bbox, expected, field_id=None):
        return False, "deliberate failure"


def _build_executor(controls, verifier, recovery, max_retries=3):
    return ActionExecutor(
        mouse=StubMouse(),
        keyboard=StubKeyboard(),
        controls=controls,
        verifier=verifier,
        recovery=recovery,
        verify_after_action=True,
        max_retries=max_retries,
        retry_delay=0.0,
    )


def test_successful_typed_action() -> None:
    controls = FakeControls()
    executor = _build_executor(controls, AlwaysPassVerifier(), RecoveryPlanner())
    action = Action(type=ActionType.TYPE, field_id="f0", value="Ravi", bbox=BBox(0, 0, 10, 10))
    result = executor.execute(action)
    assert result.ok is True
    assert result.verified is True
    assert ("type", "f0", "Ravi") in controls.calls


def test_failed_verification_recovers_then_skips() -> None:
    get_event_bus().clear()
    controls = FakeControls()
    recovery = RecoveryPlanner(max_retries=1, max_refocus=1, max_analyze=1, skip_after_exhaust=True)
    executor = _build_executor(controls, AlwaysFailVerifier(), recovery, max_retries=6)
    action = Action(type=ActionType.TYPE, field_id="f0", value="Ravi", bbox=BBox(0, 0, 10, 10))
    result = executor.execute(action)
    assert result.ok is False
    assert result.verified is False
    assert len(get_event_bus().history(EventType.ACTION_FAILED)) >= 1
    assert len(get_event_bus().history(EventType.RECOVERY)) >= 1


def test_non_verifyable_action_passes_through() -> None:
    controls = FakeControls()
    executor = _build_executor(controls, AlwaysFailVerifier(), RecoveryPlanner())
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(0, 0, 10, 10))
    result = executor.execute(action)
    assert result.ok is True
    assert ("click", "f0", None) in controls.calls


def test_execute_plan_ends_with_submit() -> None:
    controls = FakeControls()
    executor = _build_executor(controls, AlwaysPassVerifier(), RecoveryPlanner())

    field = EditableField(
        element=ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(0, 0, 10, 10)),
        offset=(0, 0),
    )
    record = SourceRecord(pairs={"Name": "Ravi"}, ordered_labels=["Name"])
    mapping = MappingResult(mappings=[FieldMapping("Name", "Ravi", field, 0.98, "exact")])
    scene = SceneDescription(elements=[
        field.element,
        ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save", bbox=BBox(0, 50, 10, 10)),
    ])
    plan = ActionPlanner().plan_fill(record, mapping, scene, "b0")
    results = executor.execute_plan(plan)
    assert results[-1].action.type == ActionType.CLICK
    assert all(r.ok for r in results)
