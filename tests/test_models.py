"""Tests for the action / vision data models."""

from __future__ import annotations

from atlas.act.models import VERIFYABLE_ACTIONS, Action, ActionResult, ActionType
from atlas.act.verify import normalize_for_compare
from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement


def test_bbox_geometry() -> None:
    box = BBox(10, 20, 100, 50)
    assert box.left == 10
    assert box.top == 20
    assert box.right == 110
    assert box.bottom == 70
    assert box.center == (60, 45)
    assert box.contains(60, 45)
    assert not box.contains(5, 5)
    assert box.shifted(5, 5) == BBox(15, 25, 100, 50)
    assert BBox.from_dict(box.to_dict()) == box


def test_scene_element_editable() -> None:
    box = BBox(0, 0, 10, 10)
    field = ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Name", bbox=box)
    assert field.editable is True
    field.disabled = True
    assert field.editable is False
    button = ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save", bbox=box)
    assert button.editable is False


def test_scene_description_editable_fields() -> None:
    box = BBox(0, 0, 10, 10)
    scene = SceneDescription(elements=[
        ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="A", bbox=box),
        ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save", bbox=box),
    ])
    assert len(scene.editable_fields) == 1
    assert scene.buttons[0].element_id == "b0"
    assert scene.element("f0").label == "A"


def test_action_serialization_roundtrip() -> None:
    action = Action(type=ActionType.TYPE, field_id="f0", value="hello", reason="fill")
    data = action.to_dict()
    assert data["type"] == "type"
    assert data["field_id"] == "f0"
    assert data["value"] == "hello"


def test_action_result_ok() -> None:
    action = Action(type=ActionType.CLICK)
    ok = ActionResult(action=action, success=True, verified=True)
    assert ok.ok is True
    unverified = ActionResult(action=action, success=True, verified=False)
    assert unverified.ok is True  # click needs no verification
    type_action = Action(type=ActionType.TYPE)
    bad = ActionResult(action=type_action, success=True, verified=False)
    assert bad.ok is False


def test_verifyable_actions() -> None:
    assert ActionType.TYPE in VERIFYABLE_ACTIONS
    assert ActionType.SELECT in VERIFYABLE_ACTIONS
    assert ActionType.CLICK not in VERIFYABLE_ACTIONS


def test_normalize_for_compare() -> None:
    assert normalize_for_compare("  Hello   World ") == "hello world"
    assert normalize_for_compare("123-4567") == "123 4567"
    for truthy in ("Yes", "yes", "CHECKED", "on", "1", "true", "X"):
        assert normalize_for_compare(truthy) == "1", truthy
    for falsy in ("No", "unchecked", "off", "0", "false", ""):
        assert normalize_for_compare(falsy) == "0", falsy
