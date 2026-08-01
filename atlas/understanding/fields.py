"""Editable field discovery.

Wraps scene elements with the geometry needed to act on them, deduplicates the
raw VLM element list, and orders fields in the natural reading order
(top-to-bottom, then left-to-right).
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.vision.models import BBox, ElementType, SceneDescription, ScreenElement


@dataclass
class EditableField:
    """A field the agent can fill, with absolute screen coordinates."""

    element: ScreenElement
    offset: tuple[int, int]  # client-area origin on the physical screen

    @property
    def element_id(self) -> str:
        return self.element.element_id

    @property
    def label(self) -> str:
        return self.element.label or self.element.name or self.element_id

    @property
    def type(self) -> ElementType:
        return self.element.type

    @property
    def bbox(self) -> BBox | None:
        return self.element.bbox

    @property
    def screen_bbox(self) -> BBox | None:
        bbox = self.element.bbox
        if bbox is None:
            return None
        return bbox.shifted(*self.offset)

    @property
    def click_point(self) -> tuple[int, int] | None:
        bbox = self.screen_bbox
        if bbox is None:
            return None
        return bbox.center

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "label": self.label,
            "type": self.type.value,
            "bbox": self.element.bbox.to_dict() if self.element.bbox else None,
            "screen_bbox": self.screen_bbox.to_dict() if self.screen_bbox else None,
            "confidence": self.element.confidence,
            "section": self.element.section,
        }


def discover_fields(scene: SceneDescription) -> list[EditableField]:
    """Return the deduplicated editable fields of a scene in reading order."""
    elements = dedupe(scene.elements)
    editable = [e for e in elements if e.editable]
    fields = [EditableField(element=e, offset=scene.screen_offset) for e in editable]
    return order_fields(fields)


def dedupe(elements: list[ScreenElement]) -> list[ScreenElement]:
    """Remove near-duplicate elements (same type overlapping the same label).

    Elements without a bounding box are kept - they may be valid web-DOM
    controls whose geometry is unknown to the vision model.
    """
    unique: list[ScreenElement] = []
    for element in elements:
        duplicate = False
        for existing in unique:
            if existing.type != element.type:
                continue
            if existing.bbox is None or element.bbox is None:
                continue  # nothing to compare geometrically
            overlap = _overlap_ratio(existing.bbox, element.bbox)
            if overlap > 0.85 and (
                existing.label == element.label or not existing.label or not element.label
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(element)
    return unique


def order_fields(fields: list[EditableField]) -> list[EditableField]:
    """Sort fields top-to-bottom, then left-to-right (reading order)."""
    keyed = []
    for field in fields:
        bbox = field.element.bbox
        key = (bbox.top, bbox.left) if bbox else (10**9, 10**9)
        keyed.append((key, field))
    keyed.sort(key=lambda pair: (pair[0][0], pair[0][1]))
    return [f for _, f in keyed]


def to_screen(scene: SceneDescription, bbox: BBox) -> BBox:
    """Translate a capture-relative box to absolute screen coordinates."""
    dx, dy = scene.screen_offset
    return bbox.shifted(dx, dy)


def _overlap_ratio(a: BBox, b: BBox) -> float:
    x = max(0, min(a.right, b.right) - max(a.left, b.left))
    y = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = x * y
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


__all__ = ["EditableField", "discover_fields", "dedupe", "order_fields", "to_screen"]
