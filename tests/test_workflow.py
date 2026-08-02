"""Tests for the workflow loop against a fake target."""

from __future__ import annotations

from atlas.act.controls import ControlInterface, ControlOutcome
from atlas.act.executor import ActionExecutor
from atlas.act.models import ActionType
from atlas.act.verify import FieldVerifier
from atlas.mapping.mapper import SemanticMapper
from atlas.reason.planner import ActionPlanner
from atlas.reason.recovery import RecoveryPlanner
from atlas.target.base import TargetAdapter, TargetInfo
from atlas.understanding.source import SourceReader
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement
from atlas.vision.scene import SceneAnalysis
from atlas.workflow.loop import AgentLoop


class RecordingControls(ControlInterface):
    def __init__(self) -> None:
        self.typed: list[str] = []

    def focus(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def click_field(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def type_value(self, bbox, value, field_id=None):
        self.typed.append(value)
        return ControlOutcome(ok=True)
    def clear(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def select_option(self, bbox, value, options=None, field_id=None): return ControlOutcome(ok=True)
    def toggle(self, bbox, value, field_id=None): return ControlOutcome(ok=True)
    def choose_date(self, bbox, value, date_format=None, field_id=None): return ControlOutcome(ok=True)
    def press_tab(self): return ControlOutcome(ok=True)
    def press_enter(self): return ControlOutcome(ok=True)
    def press_escape(self): return ControlOutcome(ok=True)
    def scroll(self, direction, amount=3): return ControlOutcome(ok=True)
    def paste(self, value, field_id=None): return ControlOutcome(ok=True)
    def upload_file(self, bbox, path, field_id=None): return ControlOutcome(ok=True)


class StubMouse:
    def move_to(self, x, y): pass
    def click(self, x, y): pass
    def double_click(self, x, y): pass
    def right_click(self, x, y): pass
    def hover(self, x, y): pass
    def scroll(self, direction, amount=3): pass


class StubKeyboard:
    driver = None


class PassVerifier(FieldVerifier):
    name = "pass"

    def verify(self, bbox, expected, field_id=None):
        return True, "ok"


class FakeTarget(TargetAdapter):
    name = "fake"

    def __init__(self, scenes: list[SceneDescription]) -> None:
        self._scenes = list(scenes)
        self._idx = 0
        self._info = TargetInfo(name="fake", title="Fake Window")

    def attach(self, hint: str | None = None) -> TargetInfo:
        return self._info

    def detach(self) -> None:
        pass

    def observe(self) -> SceneAnalysis | None:
        if self._idx < len(self._scenes):
            scene = self._scenes[self._idx]
            self._idx += 1
            return SceneAnalysis(scene=scene)
        return None

    def is_alive(self) -> bool:
        return True

    def read_field_value(self, field_id: str) -> str | None:
        return None


def make_scene(record_no: str, name: str, agree: str, pan_required: bool = False) -> SceneDescription:
    elements = [
        ScreenElement(element_id="s0", type=ElementType.LABEL, label="Application No", value=record_no, bbox=BBox(10, 10, 120, 16)),
        ScreenElement(element_id="s1", type=ElementType.LABEL, label="Applicant Name", value=name, bbox=BBox(10, 30, 120, 16)),
        ScreenElement(element_id="s2", type=ElementType.LABEL, label="Agree", value=agree, bbox=BBox(10, 50, 120, 16)),
        ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Applicant Name", bbox=BBox(200, 40, 120, 20)),
        ScreenElement(element_id="f1", type=ElementType.CHECKBOX, label="Agree", bbox=BBox(200, 80, 20, 20)),
        ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save", bbox=BBox(200, 120, 60, 24)),
    ]
    if pan_required:
        elements.append(ScreenElement(
            element_id="f2", type=ElementType.TEXTBOX, label="PAN Number", required=True, bbox=BBox(200, 100, 120, 20),
        ))
    return SceneDescription(window_title="Fake Window", elements=elements, screen_offset=(0, 0))


def _build_loop(target, controls, max_records=2, timeout=1.0):
    executor = ActionExecutor(
        mouse=StubMouse(),
        keyboard=StubKeyboard(),
        controls=controls,
        verifier=PassVerifier(),
        recovery=RecoveryPlanner(),
        verify_after_action=True,
        max_retries=2,
        retry_delay=0.0,
    )
    return AgentLoop(
        target=target,
        source_reader=SourceReader(),
        mapper=SemanticMapper(),
        planner=ActionPlanner(verify_after_action=True),
        executor=executor,
        max_records=max_records,
        next_record_timeout=timeout,
        next_record_poll=0.05,
    )


def test_loop_processes_multiple_records() -> None:
    target = FakeTarget([
        make_scene("1001", "Ravi Kumar", "Yes"),
        make_scene("1002", "Sita Devi", "No"),
    ])
    controls = RecordingControls()
    loop = _build_loop(target, controls)
    summary = loop.run()
    assert summary.completed == 2
    assert summary.failed == 0
    assert controls.typed == ["Ravi Kumar", "Sita Devi"]
    assert loop.state.value == "stopped"


def test_loop_waits_when_no_record() -> None:
    """No records must NOT terminate the loop: it retries and waits for the
    next record; only an explicit stop ends the run."""
    import threading
    import time

    target = FakeTarget([])
    loop = _build_loop(target, controls=RecordingControls(), timeout=0.3)

    def _stop() -> None:
        time.sleep(0.2)
        loop.stop()

    stopper = threading.Thread(target=_stop, daemon=True)
    stopper.start()
    summary = loop.run()
    stopper.join()
    assert summary.records == []
    assert summary.stopped_reason == "stopped by user"


def test_loop_marks_unmapped_required_field() -> None:
    target = FakeTarget([
        make_scene("1003", "Ravi", "Yes", pan_required=True),
    ])
    loop = _build_loop(target, controls=RecordingControls(), max_records=1, timeout=0.3)
    summary = loop.run()
    assert summary.completed == 1
    record = summary.records[0]
    assert "PAN Number" in record.incomplete_fields
    assert record.success is True


def test_loop_max_records() -> None:
    target = FakeTarget([
        make_scene("2001", "A", "Yes"),
        make_scene("2002", "B", "No"),
        make_scene("2003", "C", "Yes"),
    ])
    loop = _build_loop(target, controls=RecordingControls(), max_records=2, timeout=0.3)
    summary = loop.run()
    assert len(summary.records) == 2
    assert summary.stopped_reason == "max_records reached (2)"


def test_record_summary_contains_actions() -> None:
    target = FakeTarget([make_scene("3001", "Ravi", "Yes")])
    loop = _build_loop(target, controls=RecordingControls(), max_records=1, timeout=0.3)
    summary = loop.run()
    record = summary.records[0]
    types = [a.action.type for a in record.actions]
    assert ActionType.TYPE in types
    assert ActionType.TOGGLE in types
    assert types[-1] == ActionType.CLICK  # submit


def test_loop_captures_before_and_after_fill_screenshots(tmp_path) -> None:
    target = FakeTarget([make_scene("4001", "Ravi", "Yes")])
    saved: list[str] = []

    def capture(path) -> bool:
        saved.append(str(path))
        return True

    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=RecordingControls(),
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
    )
    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=SemanticMapper(),
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=1, next_record_timeout=0.3, next_record_poll=0.05,
        debug_dir=tmp_path, capture_callback=capture,
    )
    summary = loop.run()
    assert summary.completed == 1
    assert len(saved) >= 2, saved
    assert any(p.endswith("-before-fill.png") for p in saved)
    assert any(p.endswith("-after-fill.png") for p in saved)
    assert any(p.endswith("-after-upload.png") for p in saved)


class _SequenceTarget(FakeTarget):
    """Fake target whose observe() returns a fresh scene each call."""

    def observe(self) -> SceneAnalysis | None:
        if self._idx < len(self._scenes):
            scene = self._scenes[self._idx]
            self._idx += 1
            return SceneAnalysis(scene=scene)
        return None


def test_reobserve_scene_refreshes_after_ui_change() -> None:
    """Self-healing: reobserve_scene must return the NEW scene (window moved,
    layout changed, fields re-added) rather than the cached one."""
    from atlas.understanding.fields import EditableField
    from atlas.vision.models import ScreenElement

    initial = make_scene("5001", "Ravi", "Yes")
    # The UI changes: the form field moves (as if the window/layout changed).
    moved = SceneDescription(
        window_title="Fake Window",
        elements=[
            ScreenElement(element_id="s0", type=ElementType.LABEL, label="Application No",
                          value="5001", bbox=BBox(10, 10, 120, 16)),
            ScreenElement(element_id="s1", type=ElementType.LABEL, label="Applicant Name",
                          value="Ravi", bbox=BBox(10, 30, 120, 16)),
            ScreenElement(element_id="s2", type=ElementType.LABEL, label="Agree",
                          value="Yes", bbox=BBox(10, 50, 120, 16)),
            ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Applicant Name",
                          bbox=BBox(200, 40, 120, 20)),
            ScreenElement(element_id="f1", type=ElementType.CHECKBOX, label="Agree",
                          bbox=BBox(200, 80, 20, 20)),
            ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save",
                          bbox=BBox(200, 120, 60, 24)),
        ],
        screen_offset=(500, 300),  # window moved on screen
    )

    target = _SequenceTarget([initial, moved])
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=RecordingControls(),
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=2, retry_delay=0.0,
    )
    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=SemanticMapper(),
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=0, next_record_timeout=0.3, next_record_poll=0.05,
    )
    # First observation caches the initial scene.
    first = loop.reobserve_scene()
    assert first is not None and first.screen_offset == (0, 0)
    # A later re-observe (after a UI change) must see the moved window.
    second = loop.reobserve_scene()
    assert second is not None and second.screen_offset == (500, 300)
    fields = [e for e in second.elements if e.element_id == "f0"]
    assert fields and fields[0].bbox.x == 200
