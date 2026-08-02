"""Tests for the action executor (execution + verification + recovery)."""

from __future__ import annotations

from atlas.act.controls import ControlInterface, ControlOutcome
from atlas.act.executor import ActionExecutor
from atlas.act.models import Action, ActionType
from atlas.act.sandbox import ExecutionSandbox, SandboxConfig, TargetInfo
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
    def scroll(self, direction, amount=3):
        self.calls.append(("scroll", None, f"{direction}:{amount}"))
        return ControlOutcome(ok=True, evidence=f"fake scroll {direction}")
    def scroll_by_keys(self, direction, amount=3):
        self.calls.append(("scroll_by_keys", None, f"{direction}:{amount}"))
        return ControlOutcome(ok=True, evidence=f"fake keys {direction}")
    def scroll_bar(self, direction, amount=3):
        self.calls.append(("scroll_bar", None, f"{direction}:{amount}"))
        return ControlOutcome(ok=True, evidence=f"fake bar {direction}")
    def paste(self, value, field_id=None): return self._record("paste", field_id, value)
    def upload_file(self, bbox, path, field_id=None): return self._record("upload", field_id, path)


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


def test_submit_is_never_retried_even_on_failure() -> None:
    """The spec forbids double submission: an upload action must be executed
    exactly once, never re-attempted through the recovery/retry loop."""
    from atlas.core.events import get_event_bus

    get_event_bus().clear()

    class SubmitFailsControls(FakeControls):
        def press_enter(self):
            self.calls.append(("enter", None, None))
            return ControlOutcome(ok=False, evidence="submit failed once")

        def click_field(self, bbox, field_id=None):
            self.calls.append(("click", field_id, None))
            return ControlOutcome(ok=False, evidence="submit failed once")

    controls = SubmitFailsControls()
    # max_retries high enough that a retry WOULD happen if the guard were absent.
    executor = _build_executor(controls, AlwaysPassVerifier(), RecoveryPlanner(max_retries=5), max_retries=5)
    action = Action(type=ActionType.SUBMIT, field_id="b0", bbox=BBox(0, 0, 10, 10))
    result = executor.execute(action)
    assert result.ok is False
    assert result.retries == 0
    # Only one dispatch attempt for the submit, regardless of the retry budget.
    assert sum(1 for c in controls.calls if c[0] in {"click", "enter"}) == 1
    # No recovery decision was issued for the submit.
    assert len(get_event_bus().history(EventType.RECOVERY)) == 0


# -- scroll-into-view ----------------------------------------------------------

def _sandbox_with_rect(rect) -> ExecutionSandbox:
    target = TargetInfo(
        handle=1000, pid=42, tid=1, class_name="MPF",
        title="MPF (Download and Upload Form)", exe_name="mpf.exe",
        client_rect=rect,
    )
    sandbox = ExecutionSandbox(SandboxConfig(check_keyboard=False, check_mouse=False))
    sandbox.attach(target)
    return sandbox


def test_off_viewport_field_scrolls_before_click() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    scene = SceneDescription(
        screen_offset=(0, 0),
        elements=[ScreenElement(
            element_id="f0", type=ElementType.TEXTBOX, label="Name",
            bbox=BBox(100, 400, 40, 20),  # inside viewport after scroll
        )],
    )
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
    )
    # Field center (120, 610) is below the viewport bottom (500).
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 600, 40, 20))
    result = executor.execute(action)
    assert result.ok is True
    assert any(call[0] == "scroll" and call[2].startswith("down") for call in controls.calls)
    sandbox.detach()


def test_in_viewport_field_no_scroll() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    scene = SceneDescription(elements=[ScreenElement(
        element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 100, 40, 20),
    )])
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
    )
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 100, 40, 20))
    result = executor.execute(action)
    assert result.ok is True
    assert not any(call[0] == "scroll" for call in controls.calls)
    sandbox.detach()


def test_scroll_refreshes_bbox_from_reobserve() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    # First re-observe still shows the field below the fold; second shows it inside.
    scenes = iter([
        SceneDescription(screen_offset=(0, 0), elements=[ScreenElement(
            element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 700, 40, 20),
        )]),
        SceneDescription(screen_offset=(0, 0), elements=[ScreenElement(
            element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 400, 40, 20),
        )]),
    ])
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: next(scenes), sandbox=sandbox,
    )
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 700, 40, 20))
    result = executor.execute(action)
    assert result.ok is True
    scrolls = [call for call in controls.calls if call[0] == "scroll"]
    # First wheel attempt makes no progress, so the executor escalates to
    # keyboard scrolling, whose re-observe brings the field into view.
    assert scrolls
    assert any(call[0] == "scroll_by_keys" for call in controls.calls)
    assert len(scrolls) >= 1
    sandbox.detach()


def test_off_viewport_field_skipped_when_scroll_fails() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    # Re-observe never brings the field into view.
    scene = SceneDescription(screen_offset=(0, 0), elements=[ScreenElement(
        element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 700, 40, 20),
    )])
    recovery = RecoveryPlanner(max_retries=0, max_refocus=0, max_analyze=0, skip_after_exhaust=True)
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysFailVerifier(), recovery=recovery,
        verify_after_action=True, max_retries=1, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
        max_scroll_attempts=3,
    )
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 700, 40, 20))
    result = executor.execute(action)
    # Bounded scrolling: at most max_scroll_attempts scrolls, then gives up.
    scrolls = [call for call in controls.calls if call[0] == "scroll"]
    assert len(scrolls) <= 3


def test_upload_file_dispatches_to_controls() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    scene = SceneDescription(elements=[ScreenElement(
        element_id="f0", type=ElementType.FILE_UPLOAD, label="Attachment", bbox=BBox(100, 100, 40, 20),
    )])
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
    )
    action = Action(type=ActionType.UPLOAD_FILE, field_id="f0", value="C:/docs/x.pdf", bbox=BBox(100, 100, 40, 20))
    result = executor.execute(action)
    assert result.ok is True
    uploads = [call for call in controls.calls if call[0] == "upload"]
    assert len(uploads) == 1
    assert uploads[0][2] == "C:/docs/x.pdf"
    sandbox.detach()


def test_upload_file_missing_path_fails_fast() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: SceneDescription(), sandbox=sandbox,
    )
    action = Action(type=ActionType.UPLOAD_FILE, field_id="f0", value=None, bbox=BBox(100, 100, 40, 20))
    result = executor.execute(action)
    assert result.ok is False
    assert not any(call[0] == "upload" for call in controls.calls)
    sandbox.detach()
    sandbox.detach()


def test_scroll_escalates_wheel_to_keys_when_no_progress() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    # Re-observe always returns the field below the fold: the wheel scrolls but
    # the nested pane does not move, so the executor must escalate to keys.
    scene = SceneDescription(
        screen_offset=(0, 0),
        elements=[ScreenElement(
            element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 700, 40, 20),
        )],
    )
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysPassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
        max_scroll_attempts=6,
    )
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 700, 40, 20))
    result = executor.execute(action)
    assert result.ok is True
    wheels = [c for c in controls.calls if c[0] == "scroll"]
    keys = [c for c in controls.calls if c[0] == "scroll_by_keys"]
    bars = [c for c in controls.calls if c[0] == "scroll_bar"]
    # Wheel ran, made no progress, then keys took over.
    assert wheels
    assert keys
    assert bars  # keys also made no progress, escalated to scroll-bar
    sandbox.detach()


def test_scroll_stops_when_strategies_exhausted() -> None:
    controls = FakeControls()
    sandbox = _sandbox_with_rect((0, 0, 500, 500))
    scene = SceneDescription(
        screen_offset=(0, 0),
        elements=[ScreenElement(
            element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(100, 700, 40, 20),
        )],
    )
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=AlwaysFailVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=1, retry_delay=0.0,
        reobserve=lambda: scene, sandbox=sandbox,
        max_scroll_attempts=6,
    )
    action = Action(type=ActionType.CLICK, field_id="f0", bbox=BBox(100, 700, 40, 20))
    result = executor.execute(action)
    # All three strategies ran, then the executor gave up (action still acted,
    # but sandbox/blocks keep it safe; here the fake always passes).
    strategies = [
        c[0] for c in controls.calls
        if c[0] in {"scroll", "scroll_by_keys", "scroll_bar"}
    ]
    assert strategies[0] == "scroll"
    assert "scroll_by_keys" in strategies
    assert "scroll_bar" in strategies
    # No more than max_scroll_attempts total scroll attempts.
    assert len(strategies) <= 6
    sandbox.detach()
