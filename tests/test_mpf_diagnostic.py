"""MPF diagnostic and workflow tests.

Tests that the MPF plugin correctly:
1. Detects MPF windows
2. Refines scenes with left/right panel tagging
3. Maps source labels to form fields
4. Creates proper fill plans
5. Handles record lifecycle (upload, next record, stop)
"""

from __future__ import annotations

from pathlib import Path

from atlas.mapping.mapper import SemanticMapper, normalize_label
from atlas.reason.planner import ActionPlanner, ActionType
from atlas.understanding.fields import discover_fields, EditableField
from atlas.understanding.source import SourceReader
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement

from plugins.mpf.mpf_detector import MpfDetector, load_field_mapping
from plugins.mpf.mpf_workflow import MpfWorkflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mpf_scene(
    window_title: str = "MPF (Download and Upload Form)",
    left_pairs: list[tuple[str, str]] | None = None,
    right_fields: list[tuple[str, ElementType, bool]] | None = None,
    has_upload: bool = True,
) -> SceneDescription:
    """Build a synthetic MPF scene with left panel + right form + upload button."""
    left_pairs = left_pairs or [
        ("Full Name", "KRISHNA"),
        ("Gender", "Male"),
        ("DOB", "21 March 1996"),
        ("Mobile Number", "9876543210"),
    ]
    right_fields = right_fields or [
        ("Full Name", ElementType.TEXTBOX, True),
        ("Gender", ElementType.COMBOBOX, True),
        ("Date Of Birth", ElementType.DATE_PICKER, True),
        ("Mobile Number", ElementType.TEXTBOX, True),
    ]

    elements: list[ScreenElement] = []
    # Left panel source data (labels with values)
    y = 20
    for label, value in left_pairs:
        elements.append(ScreenElement(
            element_id=f"src_{normalize_label(label)}",
            type=ElementType.LABEL,
            label=label,
            value=value,
            bbox=BBox(10, y, 150, 20),
            confidence=0.95,
            section="source",
        ))
        y += 30

    # Right panel form fields (editable controls)
    y = 20
    for label, etype, required in right_fields:
        elements.append(ScreenElement(
            element_id=f"fld_{normalize_label(label)}",
            type=etype,
            label=label,
            bbox=BBox(300, y, 180, 24),
            confidence=0.9,
            section="form",
            required=required,
        ))
        y += 35

    # Upload button
    if has_upload:
        elements.append(ScreenElement(
            element_id="btn_upload",
            type=ElementType.BUTTON,
            label="Upload Details",
            bbox=BBox(300, 300, 120, 30),
            confidence=0.95,
            section="actions",
        ))

    scene = SceneDescription(
        window_title=window_title,
        layout_summary="mpf(left=source,right=form)",
        elements=elements,
        screen_offset=(0, 0),
        confidence=0.9,
    )
    scene.sections = [
        type("Section", (), {"name": "source", "to_dict": lambda: {"name": "source"}})(),
        type("Section", (), {"name": "form", "to_dict": lambda: {"name": "form"}})(),
    ]
    return scene


def _make_detector() -> MpfDetector:
    config_path = Path(__file__).parent.parent / "plugins" / "mpf" / "field_mapping.json"
    mapping = load_field_mapping(config_path)
    return MpfDetector(
        window_keywords=mapping.get("window_keywords", ["mpf"]),
        upload_labels=mapping.get("upload_button_labels", ["upload"]),
        field_map=mapping.get("fields", {}),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mpf_detector_identifies_window() -> None:
    detector = _make_detector()
    assert detector.is_mpf_window("MPF (Download and Upload Form)")
    assert detector.is_mpf_window("MPF - Upload Form")
    assert not detector.is_mpf_window("Notepad")
    assert not detector.is_mpf_window("")


def test_mpf_detector_refines_scene() -> None:
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    # Check section tagging
    for element in refined.elements:
        if element.element_id.startswith("src_"):
            assert element.section == "source", f"{element.element_id} should be source"
        elif element.element_id.startswith("fld_"):
            assert element.section == "form", f"{element.element_id} should be form"
        elif element.element_id == "btn_upload":
            assert element.section == "actions", f"{element.element_id} should be actions"

    # Check upload button detection
    upload = detector.find_upload_button(refined)
    assert upload is not None
    assert upload.element_id == "btn_upload"


def test_mpf_source_reader_extracts_pairs() -> None:
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    reader = SourceReader()
    record = reader.read(refined)
    assert record.pairs is not None
    assert "Full Name" in record.pairs
    assert record.pairs["Full Name"] == "KRISHNA"
    assert "Gender" in record.pairs
    assert record.pairs["Gender"] == "Male"
    assert "DOB" in record.pairs
    assert record.pairs["DOB"] == "21 March 1996"
    assert "Mobile Number" in record.pairs
    assert record.pairs["Mobile Number"] == "9876543210"


def test_mpf_field_discovery() -> None:
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    fields = discover_fields(refined)
    assert len(fields) >= 4
    field_labels = [f.label for f in fields]
    assert "Full Name" in field_labels
    assert "Gender" in field_labels
    assert "Date Of Birth" in field_labels
    assert "Mobile Number" in field_labels

    # Check types
    for f in fields:
        if f.label == "Gender":
            assert f.element.type == ElementType.COMBOBOX
        elif f.label == "Date Of Birth":
            assert f.element.type in {ElementType.DATE_PICKER, ElementType.TEXTBOX}


def test_mpf_semantic_mapping() -> None:
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    reader = SourceReader()
    record = reader.read(refined)
    fields = discover_fields(refined)

    mapper = SemanticMapper()
    mapping = mapper.map(record, fields)

    # Check that source labels mapped to target fields
    mapped_labels = [m.source_label for m in mapping.mappings]
    assert "Full Name" in mapped_labels, f"Full Name not mapped: {mapped_labels}"
    assert "Gender" in mapped_labels, f"Gender not mapped: {mapped_labels}"
    assert "DOB" in mapped_labels, f"DOB not mapped: {mapped_labels}"
    assert "Mobile Number" in mapped_labels, f"Mobile Number not mapped: {mapped_labels}"

    # Check values carried through
    for m in mapping.mappings:
        if m.source_label == "Full Name":
            assert m.source_value == "KRISHNA"
        elif m.source_label == "Gender":
            assert m.source_value == "Male"


def test_mpf_action_plan() -> None:
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    reader = SourceReader()
    record = reader.read(refined)
    fields = discover_fields(refined)

    mapper = SemanticMapper()
    mapping = mapper.map(record, fields)

    upload = detector.find_upload_button(refined)
    planner = ActionPlanner(verify_after_action=True)
    plan = planner.plan_fill(record, mapping, refined, upload.element_id if upload else None)

    assert plan.actions, "Plan should have actions"

    # Check that the plan has value-producing actions with correct expected values
    type_actions = [a for a in plan.actions if a.type == ActionType.TYPE]
    assert len(type_actions) > 0
    assert any(a.value == "KRISHNA" for a in type_actions)

    # Check verification actions exist
    verify_actions = [a for a in plan.actions if a.type == ActionType.VERIFY]
    assert len(verify_actions) > 0

    # Check submit/click at end
    assert plan.actions[-1].type in {ActionType.CLICK, ActionType.SUBMIT}
    assert plan.actions[-1].reason == "click submit button"


def test_mpf_workflow_tracking() -> None:
    detector = _make_detector()
    workflow = MpfWorkflow(detector)

    assert workflow.completed == 0
    assert workflow.failed == 0

    # Simulate events
    from atlas.core.events import Event, EventType

    workflow.on_event(Event(EventType.UPLOAD_COMPLETED))
    assert workflow.completed == 1

    workflow.on_event(Event(EventType.RECORD_FAILED))
    assert workflow.failed == 1

    summary = workflow.summary()
    assert summary["completed"] == 1
    assert summary["failed"] == 1


def test_mpf_record_uploaded_detection() -> None:
    detector = _make_detector()

    # Scene with data still present -> not uploaded
    scene = _make_mpf_scene(has_upload=True)
    refined = detector.refine(scene)
    assert not detector.record_uploaded(refined), "Should not detect upload when data present"

    # Scene without upload button -> uploaded
    scene2 = _make_mpf_scene(has_upload=False)
    refined2 = detector.refine(scene2)
    assert detector.record_uploaded(refined2), "Should detect upload when button gone"


def test_mpf_full_workflow_pipeline() -> None:
    """End-to-end: scene -> source -> mapping -> plan -> all fields accounted for."""
    detector = _make_detector()
    scene = _make_mpf_scene()
    refined = detector.refine(scene)

    reader = SourceReader()
    record = reader.read(refined)
    fields = discover_fields(refined)

    mapper = SemanticMapper()
    mapping = mapper.map(record, fields)

    upload = detector.find_upload_button(refined)
    planner = ActionPlanner(verify_after_action=True)
    plan = planner.plan_fill(record, mapping, refined, upload.element_id if upload else None)

    # Verify all source values made it to the plan
    plan_values = set()
    for a in plan.actions:
        if a.value:
            plan_values.add(a.value)

    assert "KRISHNA" in plan_values
    assert "Male" in plan_values
    assert "21 March 1996" in plan_values
    assert "9876543210" in plan_values

    # Verify no unmapped required fields
    assert not mapping.unmatched_fields, f"Unmatched fields: {mapping.unmatched_fields}"