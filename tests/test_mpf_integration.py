"""End-to-end MPF integration test.

Proves the full data-entry loop works as a real AI operator would expect:

    observe (MPF window) -> read LEFT source panel -> map to RIGHT form
    -> plan -> execute (type/select/date) -> verify -> click Upload Details
    -> wait for next record -> repeat

This test wires the REAL ``MpfPlugin`` (its ``refine_scene`` hook and seeded
aliases) into the REAL ``AgentLoop``. The target is a fake window that yields
three successive MPF scenes (one per record). The plugin must tag left/right
sections and the upload button from raw geometry - no coordinates are
hardcoded anywhere.
"""

from __future__ import annotations

from pathlib import Path

from atlas.act.controls import ControlInterface, ControlOutcome
from atlas.act.executor import ActionExecutor
from atlas.act.models import ActionType
from atlas.act.verify import FieldVerifier
from atlas.config import load_config
from atlas.core.events import Event, EventType, get_event_bus
from atlas.mapping.mapper import SemanticMapper
from atlas.plugins.manager import PluginManager
from atlas.reason.planner import ActionPlanner
from atlas.reason.recovery import RecoveryPlanner
from atlas.target.base import TargetAdapter, TargetInfo
from atlas.understanding.source import SourceReader
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement
from atlas.vision.scene import SceneAnalysis
from atlas.workflow.loop import AgentLoop

from plugins.mpf.plugin import MpfPlugin

# ---------------------------------------------------------------------------
# Fakes (target + controls + verifier)
# ---------------------------------------------------------------------------


class RecordingControls(ControlInterface):
    """Records every typed/selected value like a real control engine would."""

    def __init__(self) -> None:
        self.typed: list[str] = []
        self.selected: list[str] = []
        self.dates: list[str] = []
        self.uploads_clicked = 0

    def focus(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def click_field(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def type_value(self, bbox, value, field_id=None):
        self.typed.append(value)
        return ControlOutcome(ok=True)
    def clear(self, bbox, field_id=None): return ControlOutcome(ok=True)
    def select_option(self, bbox, value, options=None, field_id=None):
        self.selected.append(value)
        return ControlOutcome(ok=True)
    def toggle(self, bbox, value, field_id=None): return ControlOutcome(ok=True)
    def choose_date(self, bbox, value, date_format=None, field_id=None):
        self.dates.append(value)
        return ControlOutcome(ok=True)
    def press_tab(self): return ControlOutcome(ok=True)
    def press_enter(self): return ControlOutcome(ok=True)
    def press_escape(self): return ControlOutcome(ok=True)
    def scroll(self, direction, amount=3): return ControlOutcome(ok=True)
    def paste(self, value, field_id=None):
        self.typed.append(value)
        return ControlOutcome(ok=True)


class StubMouse:
    def move_to(self, x, y): pass
    def click(self, x, y): self.clicked = getattr(self, "clicked", 0) + 1
    def double_click(self, x, y): pass
    def right_click(self, x, y): pass
    def hover(self, x, y): pass
    def scroll(self, direction, amount=3): pass


class StubKeyboard:
    driver = None


class PassVerifier(FieldVerifier):
    name = "pass"

    def verify(self, bbox, expected, field_id=None):
        return True, f"matched '{expected}'"


class MpfFakeTarget(TargetAdapter):
    """A fake MPF window yielding one un-refined scene per record."""

    name = "fake-mpf"

    def __init__(self, scenes: list[SceneDescription]) -> None:
        self._scenes = list(scenes)
        self._idx = 0
        self._info = TargetInfo(name="fake-mpf", title="MPF (Download and Upload Form)")

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


# ---------------------------------------------------------------------------
# Scene builders (RAW - no section tags; the plugin must derive them)
# ---------------------------------------------------------------------------


def _raw_mpf_scene(record_no: str, name: str, gender: str, dob: str) -> SceneDescription:
    """Build an un-tagged MPF window: LEFT data panel + RIGHT form + upload."""
    elements: list[ScreenElement] = [
        # LEFT source panel (x < 400)
        ScreenElement(element_id="src_app", type=ElementType.LABEL, label="Application Number",
                      value=record_no, bbox=BBox(20, 20, 150, 18), confidence=0.95),
        ScreenElement(element_id="src_name", type=ElementType.LABEL, label="Full Name",
                      value=name, bbox=BBox(20, 50, 150, 18), confidence=0.95),
        ScreenElement(element_id="src_gender", type=ElementType.LABEL, label="Gender",
                      value=gender, bbox=BBox(20, 80, 150, 18), confidence=0.95),
        ScreenElement(element_id="src_dob", type=ElementType.LABEL, label="DOB",
                      value=dob, bbox=BBox(20, 110, 150, 18), confidence=0.95),
        # RIGHT form fields (x >= 500)
        ScreenElement(element_id="fld_name", type=ElementType.TEXTBOX, label="Full Name",
                      bbox=BBox(500, 40, 200, 24), confidence=0.9, required=True),
        ScreenElement(element_id="fld_gender", type=ElementType.TEXTBOX, label="Gender",
                      bbox=BBox(500, 80, 200, 24), confidence=0.9, required=True),
        ScreenElement(element_id="fld_dob", type=ElementType.TEXTBOX, label="Date Of Birth",
                      bbox=BBox(500, 120, 200, 24), confidence=0.9, required=True),
        # BOTTOM upload button
        ScreenElement(element_id="btn_upload", type=ElementType.BUTTON,
                      label="Upload Details", bbox=BBox(500, 300, 140, 32), confidence=0.95),
    ]
    return SceneDescription(
        window_title="MPF (Download and Upload Form)",
        layout_summary="un-tagged window",
        elements=elements,
        screen_offset=(0, 0),
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _run_three_records() -> tuple[AgentLoop, RecordingControls, list[Event]]:
    """Wire the real plugin into the real loop and run 3 MPF records."""
    scenes = [
        _raw_mpf_scene("MPF-001", "KRISHNA", "Male", "21 March 1996"),
        _raw_mpf_scene("MPF-002", "RAVI KUMAR", "Female", "05 August 1990"),
        _raw_mpf_scene("MPF-003", "SITA DEVI", "Male", "14 December 1988"),
    ]
    target = MpfFakeTarget(scenes)
    controls = RecordingControls()

    plugin = MpfPlugin()  # real plugin: reads field_mapping.json next to it

    executor = ActionExecutor(
        mouse=StubMouse(),
        keyboard=StubKeyboard(),
        controls=controls,
        verifier=PassVerifier(),
        recovery=RecoveryPlanner(),
        verify_after_action=True,
        max_retries=3,
        retry_delay=0.0,
    )

    # Seed aliases just like Assistant._setup_plugins does.
    mapper = SemanticMapper()
    for variant, canonical in plugin._config.get("aliases", {}).items():
        mapper.aliases.learn(variant, canonical)

    loop = AgentLoop(
        target=target,
        source_reader=SourceReader(),
        mapper=mapper,
        planner=ActionPlanner(verify_after_action=True),
        executor=executor,
        max_records=3,
        next_record_timeout=2.0,
        next_record_poll=0.05,
        scene_hook=plugin.refine_scene,  # the plugin refines every scene
    )

    history: list[Event] = []
    get_event_bus().subscribe_all(history.append)

    summary = loop.run()
    return loop, controls, history, summary  # type: ignore[return-value]


def test_mpf_plugin_refines_raw_scenes() -> None:
    """The plugin must tag LEFT data + RIGHT form + bottom upload from geometry."""
    plugin = MpfPlugin()
    scene = _raw_mpf_scene("MPF-001", "KRISHNA", "Male", "21 March 1996")
    refined = plugin.refine_scene(scene)

    sections = {e.section for e in refined.elements if e.section}
    assert sections == {"source", "form", "actions"}, sections

    upload = plugin.detector.find_upload_button(refined)
    assert upload is not None and upload.element_id == "btn_upload"

    for element in refined.elements:
        if element.element_id.startswith("src_"):
            assert element.section == "source"
        elif element.element_id.startswith("fld_"):
            assert element.section == "form"
        elif element.element_id == "btn_upload":
            assert element.section == "actions"


def test_mpf_full_loop_three_records() -> None:
    """3 records: read LEFT, fill RIGHT, upload, repeat; STOP after 3."""
    loop, controls, history, summary = _run_three_records()

    # All three records completed with verification.
    assert summary.completed == 3, f"completed={summary.completed} records"
    assert summary.failed == 0
    assert len(summary.records) == 3

    # Every record processed the correct source values.
    assert controls.typed == ["KRISHNA", "RAVI KUMAR", "SITA DEVI"]
    assert controls.selected == ["Male", "Female", "Male"]
    assert controls.dates == ["21 March 1996", "05 August 1990", "14 December 1988"]

    # Upload was clicked once per record.
    upload_events = [e for e in history if e.type == EventType.UPLOAD_COMPLETED]
    assert len(upload_events) == 3, "expected one upload per record"

    # Loop ended in the stopped state.
    assert loop.state.value == "stopped"


def test_mpf_loop_stops_on_stop_flag() -> None:
    """STOP stops execution safely between records."""
    scenes = [
        _raw_mpf_scene("MPF-001", "KRISHNA", "Male", "21 March 1996"),
        _raw_mpf_scene("MPF-002", "RAVI KUMAR", "Female", "05 August 1990"),
        _raw_mpf_scene("MPF-003", "SITA DEVI", "Male", "14 December 1988"),
    ]
    target = MpfFakeTarget(scenes)
    controls = RecordingControls()

    plugin = MpfPlugin()
    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=3, retry_delay=0.0,
    )
    mapper = SemanticMapper()
    for variant, canonical in plugin._config.get("aliases", {}).items():
        mapper.aliases.learn(variant, canonical)

    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=mapper,
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=0, next_record_timeout=2.0, next_record_poll=0.05,
        scene_hook=plugin.refine_scene,
    )

    # Stop after the first record completes.
    def _stop_after_first(event: Event) -> None:
        if event.type == EventType.RECORD_COMPLETED:
            loop.stop()

    get_event_bus().subscribe_all(_stop_after_first)
    summary = loop.run()

    assert len(summary.records) == 1
    assert summary.completed == 1
    assert controls.typed == ["KRISHNA"]
    assert loop.state.value == "stopped"


def test_mpf_field_mapping_json_loads() -> None:
    """field_mapping.json must contain the documented example fields."""
    path = Path(__file__).parent.parent / "plugins" / "mpf" / "field_mapping.json"
    plugin = MpfPlugin(path)
    fields = plugin._config.get("fields", {})
    assert "Full Name" in fields
    assert "Gender" in fields
    assert "Date Of Birth" in fields
    assert "Mobile Number" in fields
