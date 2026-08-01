"""Source record reading (the "left data panel").

Given a scene, identify the source data region (the panel that shows the
current record - typically the left panel) and extract label/value pairs.
These pairs become the source record that is mapped onto the destination
form's editable fields.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from atlas.core.logging import logger
from atlas.vision.models import ElementType, SceneDescription, ScreenElement

#: Label shapes that identify the record key used for next-record detection.
APP_NUMBER_PATTERNS = re.compile(
    r"(application|app|appl|case|file|record|serial|reference)\s*(no|number|#|id)", re.IGNORECASE
)


@dataclass
class SourceRecord:
    """A source record: an ordered mapping of label -> value."""

    pairs: dict[str, str] = field(default_factory=dict)
    ordered_labels: list[str] = field(default_factory=list)
    title: str = ""

    @property
    def record_key(self) -> str | None:
        """The value of the application/record number field, if present."""
        for label in self.ordered_labels:
            if APP_NUMBER_PATTERNS.search(label):
                value = self.pairs.get(label)
                if value:
                    return value
        for label, value in self.pairs.items():
            if APP_NUMBER_PATTERNS.search(label) and value:
                return value
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def get(self, label: str, default: str | None = None) -> str | None:
        return self.pairs.get(label, default)

    def to_dict(self) -> dict:
        return {
            "pairs": dict(self.pairs),
            "ordered_labels": list(self.ordered_labels),
            "title": self.title,
            "record_key": self.record_key,
        }


class SourceReader:
    """Extracts a SourceRecord from a scene using the VLM-provided pairs.

    The VLM already annotates each element with ``label`` and ``value``. The
    source panel is the section that contains non-editable, value-bearing
    elements (typically the left panel). Elements are collected in visual
    order so the record preserves field order.
    """

    def __init__(self) -> None:
        self._cache: dict[str, SourceRecord] = {}

    def read(self, scene: SceneDescription) -> SourceRecord:
        if not scene.elements:
            return SourceRecord(title=scene.window_title)

        key = self._scene_key(scene)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        pairs: dict[str, str] = {}
        ordered: list[str] = []

        # 1) Prefer label/value pairs on non-editable elements in the source
        #    panel (tagged by a plugin, e.g. the MPF left panel).
        source_elements = [
            e
            for e in sorted(scene.elements, key=lambda e: _element_order(e))
            if not e.editable and e.section in {"source", "left", "data"}
        ]
        for element in source_elements:
            label = self._clean_label(element)
            value = self._clean_value(element)
            if label and value is not None and label not in pairs:
                pairs[label] = value
                ordered.append(label)

        # 2) General fallback: any non-editable label/value element.
        if not pairs:
            for element in sorted(scene.elements, key=lambda e: _element_order(e)):
                if element.editable:
                    continue
                label = self._clean_label(element)
                value = self._clean_value(element)
                if label and value is not None and label not in pairs:
                    pairs[label] = value
                    ordered.append(label)

        # 3) Final fallback: LABEL elements that were assigned a value by the VLM.
        if not pairs:
            for element in sorted(scene.elements, key=lambda e: _element_order(e)):
                if element.type == ElementType.LABEL and element.value:
                    label = self._clean_label(element)
                    if label and label not in pairs:
                        pairs[label] = element.value.strip()
                        ordered.append(label)

        record = SourceRecord(pairs=pairs, ordered_labels=ordered, title=scene.window_title)
        self._cache[key] = record
        if record.record_key:
            logger.debug("source record key: {}", record.record_key)
        return record

    @staticmethod
    def _clean_label(element: ScreenElement) -> str:
        label = element.label or element.name or ""
        label = re.sub(r"[:：\s]+$", "", label).strip()
        return label

    @staticmethod
    def _clean_value(element: ScreenElement) -> str | None:
        value = element.value
        if value is None:
            return None
        value = str(value).strip()
        return value if value else None

    @staticmethod
    def _scene_key(scene: SceneDescription) -> str:
        digest = hashlib.md5()
        for element in sorted(scene.elements, key=lambda e: e.element_id):
            digest.update(
                f"{element.element_id}|{element.type.value}|{element.label}|{element.value}|{element.bbox}".encode(
                    "utf-8", errors="replace"
                )
            )
        return digest.hexdigest()


def _element_order(element: ScreenElement) -> tuple[int, int]:
    bbox = element.bbox
    if bbox is None:
        return (10**9, 10**9)
    return (bbox.top, bbox.left)


__all__ = ["SourceRecord", "SourceReader", "APP_NUMBER_PATTERNS"]
