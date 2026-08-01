"""MPF scene detector.

Recognises the MPF (Download and Upload Form) desktop window from its title,
splits the perceived scene into the LEFT source panel and RIGHT form panel by
relative geometry (never fixed coordinates), tags the "Upload Details" button,
and reports when the current record has been uploaded.

The tags drive the rest of the pipeline: the SourceReader prefers ``section ==
"source"`` elements, field discovery + mapping operate on the ``"form"``
section, and the loop's submit detection lands on the ``"actions"`` button.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.core.logging import logger
from atlas.mapping.mapper import normalize_label
from atlas.vision.models import ElementType, SceneDescription, ScreenElement

SOURCE_SECTIONS = {"source", "left", "data"}
FORM_SECTIONS = {"form", "right", "target"}
ACTION_SECTION = "actions"

#: Element types that are value-producing (they carry label/value pairs).
_DATA_TYPES = {
    ElementType.LABEL,
    ElementType.TEXTBOX,
    ElementType.SECTION,
    ElementType.UNKNOWN,
    ElementType.STATUS_BAR,
}


class MpfDetector:
    """Structural + semantic understanding of an MPF scene."""

    def __init__(self, window_keywords: list[str], upload_labels: list[str], field_map: dict[str, Any]) -> None:
        self._window_keywords = [w.lower() for w in window_keywords]
        self._upload_labels = [u.lower() for u in upload_labels]
        self._field_map: dict[str, dict[str, Any]] = {normalize_label(k): v for k, v in field_map.items()}

    # -- window identity ------------------------------------------------------

    def is_mpf_window(self, title: str | None) -> bool:
        if not title:
            return False
        lowered = title.lower()
        return any(keyword in lowered for keyword in self._window_keywords)

    # -- layout ---------------------------------------------------------------

    def refine(self, scene: SceneDescription) -> SceneDescription:
        """Annotate every element with a section based on relative geometry.

        Segments the window into:
          * left data panel  (source)  - x < mid_x
          * right editable form (form) - x >= mid_x
          * bottom action area (actions) - upload/submit buttons by label
        Never uses absolute coordinates - everything is relative to the
        captured client area.
        """
        boxed = [e for e in scene.elements if e.bbox is not None]
        if not boxed:
            return scene
        max_right = max(e.bbox.right for e in boxed)
        max_bottom = max(e.bbox.bottom for e in boxed)
        if max_right <= 0:
            return scene
        mid_x = max_right // 2
        # Bottom action band: lowest 20% of the window height.
        action_band_top = max_bottom * 0.8 if max_bottom > 0 else 10**9

        for element in scene.elements:
            if element.bbox is None:
                element.section = None
                continue
            if self._is_upload_button(element):
                element.section = ACTION_SECTION
                if element.type in {ElementType.UNKNOWN, ElementType.TOOLBAR}:
                    element.type = ElementType.BUTTON
                element.disabled = False
                continue
            # Bottom action area: buttons in the lowest band.
            if element.type == ElementType.BUTTON and element.bbox.top >= action_band_top:
                element.section = ACTION_SECTION
                continue
            element.section = "source" if element.bbox.center[0] < mid_x else "form"

        self._correct_field_types(scene)
        scene.layout_summary = "mpf(left=source,right=form,bottom=actions)"
        return scene

    def _correct_field_types(self, scene: SceneDescription) -> None:
        """Upgrade generic elements in the form to their declared widget type."""
        for element in scene.elements:
            if element.section != "form" or not element.editable:
                continue
            declared = self._field_map.get(normalize_label(element.label or ""))
            if not declared:
                continue
            declared_type = _parse_element_type(declared.get("type", ""))
            if declared_type is None or declared_type == element.type:
                continue
            if _more_specific(declared_type, element.type):
                logger.debug("mpf: retyped '{}' {} -> {}", element.label, element.type.value, declared_type.value)
                element.type = declared_type
                if declared.get("required"):
                    element.required = True

    # -- buttons --------------------------------------------------------------

    def is_upload_button(self, element: ScreenElement) -> bool:
        return self._is_upload_button(element)

    def _is_upload_button(self, element: ScreenElement) -> bool:
        if element.type not in {ElementType.BUTTON, ElementType.UNKNOWN, ElementType.TOOLBAR}:
            return False
        text = normalize_label(element.label or element.name or "")
        return any(token in text for token in self._upload_labels)

    def find_upload_button(self, scene: SceneDescription) -> ScreenElement | None:
        """Best candidate for the 'Upload Details' button."""
        tagged = [e for e in scene.elements if e.section == ACTION_SECTION and e.bbox is not None]
        if tagged:
            tagged.sort(key=lambda e: (e.bbox.top, e.bbox.left))
            return tagged[0]
        candidates = [e for e in scene.elements if self._is_upload_button(e) and e.bbox is not None]
        if candidates:
            candidates.sort(key=lambda e: (e.bbox.top, e.bbox.left))
            return candidates[0]
        return None

    # -- record lifecycle -----------------------------------------------------

    def record_uploaded(self, scene: SceneDescription) -> bool:
        """True when the source panel no longer holds a fresh record.

        Used to decide that uploading finished and the app is waiting for the
        next record (e.g. an empty panel or a success message).
        """
        if self.find_upload_button(scene) is None:
            return True
        for element in scene.elements:
            if element.section == "source" and element.bbox is not None:
                if element.label and element.value:
                    return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_keywords": self._window_keywords,
            "upload_labels": self._upload_labels,
            "field_count": len(self._field_map),
        }


def _parse_element_type(name: str) -> ElementType | None:
    try:
        return ElementType(name.strip().lower())
    except ValueError:
        return None


def _more_specific(candidate: ElementType, current: ElementType) -> bool:
    """Declared types beat generic types; never downgrade a specific type."""
    generic = {ElementType.UNKNOWN, ElementType.TEXTBOX, ElementType.SECTION}
    if current in generic and candidate not in generic:
        return True
    if candidate == ElementType.COMBOBOX and current == ElementType.TEXTBOX:
        return True
    if candidate in {ElementType.DATE_PICKER, ElementType.CALENDAR} and current in {ElementType.TEXTBOX, ElementType.UNKNOWN}:
        return True
    if candidate == ElementType.TEXTAREA and current == ElementType.TEXTBOX:
        return True
    return False


def load_field_mapping(path: str | Path) -> dict[str, Any]:
    """Load the MPF field/alias configuration file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["MpfDetector", "load_field_mapping"]
