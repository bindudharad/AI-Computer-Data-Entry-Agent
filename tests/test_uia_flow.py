"""Tests for the UIA-anchored data-entry flow.

Covers the new pieces without touching a real UI: the new state-machine
transitions, the source-pair OCR parser, the field-map JSON round-trip, the
upload-button picker, and the end-to-end AgentLoop run against a synthetic
``UiaFieldMap`` (exact form geometry injected into an otherwise empty scene).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atlas.act.controls import ControlInterface, ControlOutcome
from atlas.act.executor import ActionExecutor
from atlas.core.states import AgentState, StateMachine
from atlas.core.events import EventType, get_event_bus
from atlas.mapping.mapper import SemanticMapper
from atlas.mapping.uia_map import UiaFieldMap, UiaFieldMapBuilder, pair_source_pairs
from atlas.observe.uia import UiaNode
from atlas.reason.planner import ActionPlanner
from atlas.reason.recovery import RecoveryPlanner
from atlas.target.base import TargetAdapter, TargetInfo
from atlas.understanding.source import SourceReader
from atlas.vision.models import BBox, ElementType, OcrText, SceneDescription
from atlas.vision.scene import SceneAnalysis
from atlas.workflow.loop import AgentLoop

from tests.test_mpf_integration import PassVerifier, RecordingControls, StubKeyboard, StubMouse

from plugins.mpf.plugin import MpfPlugin


# ---------------------------------------------------------------------------
# New state machine
# ---------------------------------------------------------------------------


def test_state_machine_new_states() -> None:
    sm = StateMachine()
    sm.force(AgentState.ATTACHING)
    sm.transition(AgentState.WAITING_FOR_START_FIELD)
    assert sm.state == AgentState.WAITING_FOR_START_FIELD
    sm.transition(AgentState.FIELD_MAPPING)
    assert sm.state == AgentState.FIELD_MAPPING
    sm.transition(AgentState.WATCHING)
    assert sm.state == AgentState.WATCHING


def test_state_machine_new_states_active() -> None:
    sm = StateMachine()
    for state in (AgentState.WAITING_FOR_START_FIELD, AgentState.FIELD_MAPPING):
        sm.force(state)
        assert sm.is_active()


# ---------------------------------------------------------------------------
# Source pair parsing
# ---------------------------------------------------------------------------


def _ocr(texts: list[str]) -> list[OcrText]:
    return [OcrText(text=t, bbox=BBox(0, i * 20, 300, 16)) for i, t in enumerate(texts)]


def test_pair_source_pairs_colon_lines() -> None:
    lines = _ocr([
        "Application Number : MPF-100",
        "Full Name : KRISHNA",
        "DOB : 21 March 1996",
    ])
    pairs = pair_source_pairs(lines, [])
    assert dict(pairs) == {
        "Application Number": "MPF-100",
        "Full Name": "KRISHNA",
        "DOB": "21 March 1996",
    }


def test_pair_source_pairs_uia_labels_fill_in() -> None:
    lines = _ocr(["Application Number : MPF-100"])
    labels = [UiaNode(name="Full Name", control_type="Text"), UiaNode(name="Application Number", control_type="Text")]
    pairs = dict(pair_source_pairs(lines, labels))
    assert pairs["Application Number"] == "MPF-100"
    assert "Full Name" in pairs  # known label from UIA, no OCR value


# ---------------------------------------------------------------------------
# Field map serialization
# ---------------------------------------------------------------------------


def _sample_field_map() -> UiaFieldMap:
    start = UiaNode(name="Full Name", control_type="Edit", handle=1001, rect=BBox(500, 40, 200, 24))
    left = [
        UiaNode(name="Application Number", control_type="Text", rect=BBox(20, 20, 150, 18)),
        UiaNode(name="Full Name", control_type="Text", rect=BBox(20, 50, 150, 18)),
    ]
    right = [
        UiaNode(name="Full Name", control_type="Edit", handle=2001, rect=BBox(500, 40, 200, 24)),
        UiaNode(name="Gender", control_type="ComboBox", handle=2002, rect=BBox(500, 80, 200, 24), options=["Male", "Female"]),
        UiaNode(name="Date Of Birth", control_type="Edit", handle=2003, rect=BBox(500, 120, 200, 24),
                type_override=ElementType.DATE_PICKER),
    ]
    upload = UiaNode(name="Upload Details", control_type="Button", handle=3001, rect=BBox(500, 300, 140, 32))
    return UiaFieldMap(
        start_control=start,
        left_labels=left,
        right_fields=right,
        upload_button=upload,
        left_rect=BBox(10, 10, 300, 200),
        right_rect=BBox(480, 30, 300, 320),
        mappings=[{"source": "Full Name", "target": "Full Name", "confidence": 0.98}],
        client_origin=(0, 0),
        client_size=(1024, 768),
    )


def test_field_map_json_round_trip(tmp_path: Path) -> None:
    field_map = _sample_field_map()
    path = tmp_path / "field_map.json"
    field_map.save(path)
    loaded = UiaFieldMap.load(path)
    assert loaded is not None
    assert loaded.start_control is not None and loaded.start_control.name == "Full Name"
    assert len(loaded.left_labels) == 2
    assert len(loaded.right_fields) == 3
    assert loaded.right_fields[1].options == ["Male", "Female"]
    assert loaded.upload_button is not None and loaded.upload_button.name == "Upload Details"
    assert loaded.left_rect == BBox(10, 10, 300, 200)


def test_builder_attaches_declared_type_and_options() -> None:
    builder = UiaFieldMapBuilder(declared_fields={
        "gender": {"type": "combobox", "options": ["Male", "Female"]},
        "date of birth": {"type": "date_picker"},
    })
    node = UiaNode(name="Gender", control_type="Edit", handle=5)
    node = builder._attach_declared(node)
    assert node.type_override == ElementType.COMBOBOX
    assert node.options == ["Male", "Female"]

    dob = builder._attach_declared(UiaNode(name="Date Of Birth", control_type="Edit"))
    assert dob.type_override == ElementType.DATE_PICKER


def test_upload_button_picker_word_boundary() -> None:
    buttons = [
        UiaNode(name="Blue Book DSA", control_type="Button", rect=BBox(180, 700, 100, 24)),
        UiaNode(name="Upload Details", control_type="Button", rect=BBox(500, 300, 140, 32)),
        UiaNode(name="OK", control_type="Button", rect=BBox(10, 10, 40, 20)),
    ]
    picked = UiaFieldMapBuilder._pick_upload_button(buttons)
    assert picked is not None and picked.name == "Upload Details"


# ---------------------------------------------------------------------------
# End-to-end loop against a synthetic field map
# ---------------------------------------------------------------------------


class FieldMapFakeTarget(TargetAdapter):
    name = "fake-fieldmap"

    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self._idx = 0
        self._info = TargetInfo(name="fake-fieldmap", title="MPF (Download and Upload Form)")

    def attach(self, hint: str | None = None) -> TargetInfo:
        return self._info

    def detach(self) -> None:
        pass

    def observe(self) -> SceneAnalysis | None:
        if self._idx < len(self._records):
            self._idx += 1
            return SceneAnalysis(scene=SceneDescription(
                window_title="MPF (Download and Upload Form)",
                layout_summary="empty scene",
                screen_offset=(0, 0),
            ))
        return None

    def is_alive(self) -> bool:
        return True

    def read_field_value(self, field_id: str) -> str | None:
        return None

    def current(self) -> dict:
        return self._records[min(max(self._idx - 1, 0), len(self._records) - 1)]


def _run_anchored_records(tmp_path: Path) -> tuple[AgentLoop, RecordingControls, dict]:
    get_event_bus().clear()  # event bus is a process singleton; drop prior history
    records = [
        {"key": "MPF-100", "name": "KRISHNA", "gender": "Male", "dob": "21 March 1996"},
        {"key": "MPF-200", "name": "RAVI KUMAR", "gender": "Female", "dob": "05 August 1990"},
        {"key": "MPF-300", "name": "SITA DEVI", "gender": "Male", "dob": "14 December 1988"},
    ]
    target = FieldMapFakeTarget(records)
    controls = RecordingControls()

    def ocr_callback(bbox: BBox) -> list[OcrText]:
        rec = target.current()
        return _ocr([
            f"Application Number : {rec['key']}",
            f"Full Name : {rec['name']}",
            f"Gender : {rec['gender']}",
            f"DOB : {rec['dob']}",
        ])

    plugin = MpfPlugin()
    mapper = SemanticMapper()
    for variant, canonical in plugin._config.get("aliases", {}).items():
        mapper.aliases.learn(variant, canonical)

    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=3, retry_delay=0.0,
    )

    field_map = _sample_field_map()
    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=mapper,
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=3, next_record_timeout=2.0, next_record_poll=0.05,
        scene_hook=plugin.refine_scene,
        field_map=field_map,
        ocr_callback=ocr_callback,
        debug_dir=tmp_path,
    )
    summary = loop.run()
    return loop, controls, summary.to_dict()


def test_anchored_loop_three_records(tmp_path: Path) -> None:
    loop, controls, summary = _run_anchored_records(tmp_path)

    assert summary["completed"] == 3, summary
    assert summary["failed"] == 0
    assert controls.typed == ["KRISHNA", "RAVI KUMAR", "SITA DEVI"]
    assert controls.selected == ["Male", "Female", "Male"]
    assert controls.dates == ["21 March 1996", "05 August 1990", "14 December 1988"]
    assert loop.state.value == "stopped"

    uploads = get_event_bus().history(EventType.UPLOAD_COMPLETED)
    assert len(uploads) == 3


def test_anchored_loop_writes_debug_artifacts(tmp_path: Path) -> None:
    _run_anchored_records(tmp_path)
    for name in ("planner.json", "execution.json", "verification.json"):
        assert (tmp_path / name).exists(), f"missing {name}"
    # Per-record session artifacts, including the extracted record.
    for name in ("record.json", "timeline.json"):
        assert (tmp_path / "session" / name).exists(), f"missing session/{name}"
    import json

    session = json.loads((tmp_path / "session" / "record.json").read_text(encoding="utf-8"))
    assert session["key"] == "MPF-300"
    # No failures -> no failure.json for a clean run.
    assert not (tmp_path / "failure.json").exists()


def test_anchored_loop_writes_failure_json_on_empty_source(tmp_path: Path) -> None:
    """With OCR finding nothing, the loop never self-terminates; it reports a
    no-record condition, keeps waiting, and only a user stop ends the run and
    writes failure.json."""
    import threading
    import time

    target = FieldMapFakeTarget([{"key": "MPF-100", "name": "KRISHNA", "gender": "Male", "dob": "21 March 1996"}])
    controls = RecordingControls()

    def ocr_callback(bbox: BBox) -> list[OcrText]:
        return []  # OCR finds nothing -> no source record

    plugin = MpfPlugin()
    mapper = SemanticMapper()
    for variant, canonical in plugin._config.get("aliases", {}).items():
        mapper.aliases.learn(variant, canonical)

    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=3, retry_delay=0.0,
    )
    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=mapper,
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=0, next_record_timeout=0.3, next_record_poll=0.02,
        scene_hook=plugin.refine_scene,
        field_map=_sample_field_map(),
        ocr_callback=ocr_callback,
        debug_dir=tmp_path,
    )

    def _stop() -> None:
        time.sleep(0.4)
        loop.stop()

    stopper = threading.Thread(target=_stop, daemon=True)
    stopper.start()
    summary = loop.run()
    stopper.join()

    assert summary.to_dict()["records"] == []
    no_record = tmp_path / "no_record.json"
    assert no_record.exists()
    failure = tmp_path / "failure.json"
    assert failure.exists()
    import json

    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["stopped_reason"] == "stopped by user"
    assert payload["no_record_reason"]
    # The NO_RECORD event surfaced the condition (no silent records=0).
    assert len(get_event_bus().history(EventType.NO_RECORD)) > 0


def test_auto_build_records_from_ocr_without_vlm(tmp_path: Path) -> None:
    """The auto (non-anchored) path: a start_control-free field map + OCR
    source pairs yield a real Record even though the VLM scene is empty."""
    from atlas.core.record_builder import RecordBuilder

    target = FieldMapFakeTarget([
        {"key": "MPF-900", "name": "ANIL", "gender": "Female", "dob": "01 January 1999"},
    ])
    controls = RecordingControls()
    get_event_bus().clear()

    def ocr_callback(bbox: BBox) -> list[OcrText]:
        rec = target.current()
        return _ocr([
            f"Application Number : {rec['key']}",
            f"Full Name : {rec['name']}",
            f"Gender : {rec['gender']}",
            f"DOB : {rec['dob']}",
        ])

    plugin = MpfPlugin()
    mapper = SemanticMapper()
    for variant, canonical in plugin._config.get("aliases", {}).items():
        mapper.aliases.learn(variant, canonical)

    declared = plugin._config.get("fields", {})
    record_builder = RecordBuilder(declared_fields=declared, aliases=plugin._config.get("aliases", {}))

    executor = ActionExecutor(
        mouse=StubMouse(), keyboard=StubKeyboard(), controls=controls,
        verifier=PassVerifier(), recovery=RecoveryPlanner(),
        verify_after_action=True, max_retries=3, retry_delay=0.0,
    )
    loop = AgentLoop(
        target=target, source_reader=SourceReader(), mapper=mapper,
        planner=ActionPlanner(verify_after_action=True), executor=executor,
        max_records=1, next_record_timeout=2.0, next_record_poll=0.05,
        scene_hook=plugin.refine_scene,
        field_map=_sample_field_map(),
        ocr_callback=ocr_callback,
        debug_dir=tmp_path,
        session_dir=tmp_path / "session",
        record_builder=record_builder,
    )
    summary = loop.run()

    assert summary.completed == 1
    record = summary.records[0].record
    assert record.pairs["Application Number"] == "MPF-900"
    assert record.pairs["Full Name"] == "ANIL"
    assert record.pairs["Gender"] == "Female"
    session = tmp_path / "session" / "record.json"
    assert session.exists()
    import json

    payload = json.loads(session.read_text(encoding="utf-8"))
    assert payload["key"] == "MPF-900"


# ---------------------------------------------------------------------------
# Assistant orchestration (fakes only - no live window)
# ---------------------------------------------------------------------------


class _FakeListener:
    """Stand-in for MouseClickListener: returns a fixed click point."""

    def __init__(self, click: tuple[int, int]) -> None:
        self._click = click

    def wait_for_click(self, timeout: float = 0.0) -> tuple[int, int] | None:
        return self._click

    def stop(self) -> None:
        pass


class _FakeBackend:
    """Stand-in for UiaBackend: resolves the click to an editable node."""

    def __init__(self, node: UiaNode) -> None:
        self._node = node

    def client_origin(self, hwnd: int) -> tuple[int, int]:
        return (0, 29)

    def client_size(self, hwnd: int) -> tuple[int, int]:
        return (800, 600)

    def element_at(self, x: int, y: int) -> UiaNode:
        return self._node

    def focused(self) -> UiaNode:
        return self._node

    def editable_fields(self, hwnd: int) -> list[UiaNode]:
        return [self._node]

    def text_nodes(self, hwnd: int) -> list[UiaNode]:
        return []

    def buttons(self, hwnd: int) -> list[UiaNode]:
        return []

    def scroll_into_view(self, node: UiaNode) -> UiaNode:
        return node

    def dump_tree(self, hwnd: int) -> list[dict]:
        return []


def test_assistant_captures_start_control_and_builds_map(tmp_path: Path, monkeypatch) -> None:
    from atlas.assistant.assistant import Assistant
    from atlas.observe.uia import UiaBackend
    from atlas.target.desktop import DesktopTarget
    from atlas.vision.capture import ScreenGrabber

    start = UiaNode(
        name="Full Name", control_type="Edit", handle=2001,
        rect=BBox(500, 40, 200, 24), enabled=True,
    )
    backend = _FakeBackend(start)
    monkeypatch.setattr(UiaBackend, "_instance", backend)
    monkeypatch.setattr(
        UiaBackend,
        "instance",
        classmethod(lambda cls: backend),
    )

    class _FakeTarget(DesktopTarget):
        class _Info:
            handle = 2000
            title = "MPF (Download and Upload Form)"

        def __init__(self) -> None:
            self.__dict__["_info"] = self._Info()

    class _FakeGrabber(ScreenGrabber):
        def grab_rect(self, x, y, width, height):  # noqa: ARG002
            from PIL import Image
            return np.zeros((height, width, 3), dtype=np.uint8)

    assistant = object.__new__(Assistant)
    assistant._bus = get_event_bus()
    assistant._target = _FakeTarget()
    assistant._grabber = _FakeGrabber()

    out = tmp_path / "anchored"
    out.mkdir(parents=True, exist_ok=True)

    node = assistant._capture_start_control(_FakeListener((520, 52)), out, timeout=1.0)
    assert node.handle == 2001
    assert (out / "start_control.json").exists()

    field_map = assistant._build_field_map(2000, node, out)
    assert field_map.has_form
    assert (out / "field_map.json").exists()
    assert (out / "uia_tree.json").exists()
