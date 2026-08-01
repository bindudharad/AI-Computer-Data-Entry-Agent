"""Action planning.

Turns the source record + mapping result into an ordered list of executable
actions, each followed by a verification step. The planner is deterministic and
safe by construction; the LLM advisor is consulted only for ambiguous or failed
situations (see ``RecoveryPlanner``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.act.models import Action, ActionType
from atlas.core.logging import logger
from atlas.mapping.mapper import MappingResult
from atlas.understanding.fields import EditableField
from atlas.understanding.source import SourceRecord
from atlas.vision.models import ElementType, SceneDescription

#: Field types that receive a text value via typing.
TEXT_TYPES = {
    ElementType.TEXTBOX,
    ElementType.PASSWORD,
    ElementType.TEXTAREA,
    ElementType.SEARCH_BOX,
    ElementType.UNKNOWN,
}


@dataclass
class FillPlan:
    """A complete per-record plan."""

    actions: list[Action] = field(default_factory=list)
    source_record: SourceRecord | None = None
    mapping_result: MappingResult | None = None

    def to_dict(self) -> dict:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "source_record": self.source_record.to_dict() if self.source_record else None,
            "mapping": self.mapping_result.to_dict() if self.mapping_result else None,
        }


class ActionPlanner:
    """Builds the fill plan for one record.

    Strategy:
    * Mapped fields are filled in visual order (left-to-right, top-to-bottom).
    * Each value-producing action is followed by a VERIFY action.
    * Unmapped required fields are flagged for review (no blind guesses).
    * The plan ends with clicking the submit button when all mapped fields
      have been processed.
    """

    def __init__(self, verify_after_action: bool = True) -> None:
        self._verify = verify_after_action

    def plan_fill(
        self,
        record: SourceRecord,
        mapping_result: MappingResult,
        scene: SceneDescription | None = None,
        submit_element_id: str | None = None,
    ) -> FillPlan:
        actions: list[Action] = []
        ordered = sorted(
            mapping_result.mappings,
            key=lambda m: _field_order(m.target),
        )

        for mapping in ordered:
            target = mapping.target
            field_type = target.type
            value = mapping.source_value
            box = target.screen_bbox
            confidence = mapping.confidence

            if field_type in TEXT_TYPES:
                actions.append(Action(
                    type=ActionType.CLICK,
                    field_id=target.element_id,
                    value=None,
                    bbox=box,
                    confidence=confidence,
                    reason=f"focus '{mapping.source_label}'",
                ))
                if value:
                    actions.append(Action(
                        type=ActionType.CLEAR,
                        field_id=target.element_id,
                        bbox=box,
                        confidence=confidence,
                        expected="",
                        reason="clear existing value",
                    ))
                    actions.append(Action(
                        type=ActionType.TYPE,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"fill '{mapping.source_label}' = {value!r}",
                    ))
                if self._verify and value:
                    actions.append(Action(
                        type=ActionType.VERIFY,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"verify '{mapping.source_label}'",
                    ))
            elif field_type in {ElementType.COMBOBOX, ElementType.LISTBOX}:
                actions.append(Action(
                    type=ActionType.CLICK,
                    field_id=target.element_id,
                    bbox=box,
                    confidence=confidence,
                    reason=f"open '{mapping.source_label}' dropdown",
                ))
                actions.append(Action(
                    type=ActionType.SELECT,
                    field_id=target.element_id,
                    value=value,
                    bbox=box,
                    options=list(target.element.options),
                    confidence=confidence,
                    expected=value,
                    reason=f"select {value!r} in '{mapping.source_label}'",
                ))
                if self._verify:
                    actions.append(Action(
                        type=ActionType.VERIFY,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"verify '{mapping.source_label}'",
                    ))
            elif field_type == ElementType.CHECKBOX:
                actions.append(Action(
                    type=ActionType.TOGGLE,
                    field_id=target.element_id,
                    value=value,
                    bbox=box,
                    confidence=confidence,
                    expected=value,
                    reason=f"set checkbox '{mapping.source_label}' to {value!r}",
                ))
                if self._verify:
                    actions.append(Action(
                        type=ActionType.VERIFY,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"verify '{mapping.source_label}'",
                    ))
            elif field_type == ElementType.RADIO:
                actions.append(Action(
                    type=ActionType.TOGGLE,
                    field_id=target.element_id,
                    value=value,
                    bbox=box,
                    confidence=confidence,
                    expected=value,
                    reason=f"select radio '{mapping.source_label}' = {value!r}",
                ))
                if self._verify:
                    actions.append(Action(
                        type=ActionType.VERIFY,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"verify '{mapping.source_label}'",
                    ))
            elif field_type in {ElementType.DATE_PICKER, ElementType.CALENDAR}:
                actions.append(Action(
                    type=ActionType.CLICK,
                    field_id=target.element_id,
                    bbox=box,
                    confidence=confidence,
                    reason=f"open date control '{mapping.source_label}'",
                ))
                actions.append(Action(
                    type=ActionType.CHOOSE_DATE,
                    field_id=target.element_id,
                    value=value,
                    bbox=box,
                    confidence=confidence,
                    expected=value,
                    reason=f"set date '{mapping.source_label}' = {value!r}",
                ))
                if self._verify:
                    actions.append(Action(
                        type=ActionType.VERIFY,
                        field_id=target.element_id,
                        value=value,
                        bbox=box,
                        confidence=confidence,
                        expected=value,
                        reason=f"verify '{mapping.source_label}'",
                    ))
            else:
                logger.info("no action for field type {}", field_type)

        # Unmapped required-ish fields: note them, do not blind-fill.
        for unmatched in mapping_result.unmatched_fields:
            if unmatched.element.required:
                logger.warning("unmapped required field left empty: '{}'", unmatched.label)

        # Submit the form.
        if submit_element_id:
            submit_element = scene.element(submit_element_id) if scene else None
            submit_box = (
                submit_element.bbox.shifted(*scene.screen_offset)
                if submit_element is not None and submit_element.bbox is not None and scene is not None
                else None
            )
            actions.append(Action(
                type=ActionType.CLICK,
                field_id=submit_element_id,
                bbox=submit_box,
                confidence=1.0,
                expected="clicked submit",
                reason="click submit button",
            ))
        else:
            actions.append(Action(
                type=ActionType.SUBMIT,
                confidence=1.0,
                expected="submitted",
                reason="submit the form",
            ))

        plan = FillPlan(actions=actions, source_record=record, mapping_result=mapping_result)
        logger.debug("plan created: {} actions", len(actions))
        return plan


def _field_order(field: EditableField) -> tuple[int, int]:
    bbox = field.element.bbox
    if bbox is None:
        return (10**9, 10**9)
    return (bbox.top, bbox.left)


__all__ = ["ActionPlanner", "FillPlan", "TEXT_TYPES"]
