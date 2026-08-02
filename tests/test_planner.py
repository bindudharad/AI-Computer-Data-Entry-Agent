"""Tests for the action planner."""

from __future__ import annotations

from atlas.act.models import ActionType
from atlas.mapping.mapper import FieldMapping, MappingResult
from atlas.reason.planner import ActionPlanner
from atlas.understanding.fields import EditableField
from atlas.understanding.source import SourceRecord
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement


def _field(element_id: str, label: str, type_: ElementType, bbox: BBox) -> EditableField:
    return EditableField(
        element=ScreenElement(element_id=element_id, type=type_, label=label, bbox=bbox),
        offset=(0, 0),
    )


def _scene(submit_id: str = "b0") -> SceneDescription:
    return SceneDescription(elements=[
        ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=BBox(0, 0, 100, 20)),
        ScreenElement(element_id=submit_id, type=ElementType.BUTTON, label="Save", bbox=BBox(0, 100, 60, 24)),
    ])


def test_text_field_plan() -> None:
    planner = ActionPlanner(verify_after_action=True)
    record = SourceRecord(pairs={"Name": "Ravi"}, ordered_labels=["Name"])
    field = _field("f0", "Name", ElementType.TEXTBOX, BBox(0, 0, 100, 20))
    mapping = MappingResult(mappings=[
        FieldMapping("Name", "Ravi", field, 0.98, "exact"),
    ])
    scene = _scene()
    plan = planner.plan_fill(record, mapping, scene, "b0")
    types = [a.type for a in plan.actions]
    assert types == [
        ActionType.CLICK,
        ActionType.CLEAR,
        ActionType.TYPE,
        ActionType.VERIFY,
        ActionType.CLICK,  # submit
    ]
    assert plan.actions[2].value == "Ravi"
    assert plan.actions[3].field_id == "f0"
    assert plan.actions[-1].field_id == "b0"


def test_empty_value_skips_type() -> None:
    planner = ActionPlanner(verify_after_action=True)
    record = SourceRecord(pairs={"Name": ""}, ordered_labels=["Name"])
    field = _field("f0", "Name", ElementType.TEXTBOX, BBox(0, 0, 100, 20))
    mapping = MappingResult(mappings=[FieldMapping("Name", "", field, 0.98, "exact")])
    plan = planner.plan_fill(record, mapping, _scene(), "b0")
    types = [a.type for a in plan.actions]
    assert ActionType.TYPE not in types
    assert ActionType.VERIFY not in types


def test_checkbox_plan() -> None:
    planner = ActionPlanner(verify_after_action=True)
    record = SourceRecord(pairs={"Agree": "Yes"}, ordered_labels=["Agree"])
    field = _field("f0", "Agree", ElementType.CHECKBOX, BBox(0, 0, 20, 20))
    mapping = MappingResult(mappings=[FieldMapping("Agree", "Yes", field, 0.98, "exact")])
    plan = planner.plan_fill(record, mapping, _scene(), "b0")
    assert plan.actions[0].type == ActionType.TOGGLE
    assert plan.actions[0].value == "Yes"
    assert plan.actions[1].type == ActionType.VERIFY


def test_submit_without_element_id() -> None:
    planner = ActionPlanner()
    plan = planner.plan_fill(SourceRecord(), MappingResult(), _scene(), None)
    assert plan.actions[-1].type == ActionType.SUBMIT


def test_field_order_top_to_bottom() -> None:
    planner = ActionPlanner(verify_after_action=False)
    record = SourceRecord(pairs={"A": "1", "B": "2"}, ordered_labels=["A", "B"])
    a = _field("a", "A", ElementType.TEXTBOX, BBox(0, 200, 100, 20))
    b = _field("b", "B", ElementType.TEXTBOX, BBox(0, 20, 100, 20))
    mapping = MappingResult(mappings=[
        FieldMapping("A", "1", a, 0.9, "exact"),
        FieldMapping("B", "2", b, 0.9, "exact"),
    ])
    plan = planner.plan_fill(record, mapping, _scene(), "b0")
    # B (top) is focused/typed before A (bottom)
    b_index = next(i for i, a_ in enumerate(plan.actions) if a_.field_id == "b")
    a_index = next(i for i, a_ in enumerate(plan.actions) if a_.field_id == "a")
    assert b_index < a_index


def test_file_upload_plan() -> None:
    planner = ActionPlanner(verify_after_action=True)
    record = SourceRecord(pairs={"Attachment": "C:/docs/x.pdf"}, ordered_labels=["Attachment"])
    field = _field("f0", "Attachment", ElementType.FILE_UPLOAD, BBox(0, 0, 100, 20))
    mapping = MappingResult(mappings=[
        FieldMapping("Attachment", "C:/docs/x.pdf", field, 0.98, "exact"),
    ])
    plan = planner.plan_fill(record, mapping, _scene(), "b0")
    types = [a.type for a in plan.actions]
    assert types == [
        ActionType.CLICK,
        ActionType.UPLOAD_FILE,
        ActionType.VERIFY,
        ActionType.CLICK,  # submit
    ]
    assert plan.actions[0].field_id == "f0"
    assert plan.actions[1].value == "C:/docs/x.pdf"
    assert plan.actions[1].field_id == "f0"


def test_file_upload_empty_value_skips_upload() -> None:
    planner = ActionPlanner(verify_after_action=True)
    record = SourceRecord(pairs={"Attachment": ""}, ordered_labels=["Attachment"])
    field = _field("f0", "Attachment", ElementType.FILE_UPLOAD, BBox(0, 0, 100, 20))
    mapping = MappingResult(mappings=[FieldMapping("Attachment", "", field, 0.98, "exact")])
    plan = planner.plan_fill(record, mapping, _scene(), "b0")
    types = [a.type for a in plan.actions]
    # CLICK focus retained, but no UPLOAD_FILE/VERIFY for an empty path
    assert types == [ActionType.CLICK, ActionType.CLICK]
