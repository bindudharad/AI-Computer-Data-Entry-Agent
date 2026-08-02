"""Vision data models.

The ``SceneDescription`` produced by a Vision Language Model is the primary
perception output of the agent. Coordinates are relative to the captured
client-area image unless the model carries an explicit ``screen_offset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ElementType(str, Enum):
    """UI widget taxonomy understood by the agent."""

    TEXTBOX = "textbox"
    PASSWORD = "password"
    TEXTAREA = "textarea"
    COMBOBOX = "combobox"
    LISTBOX = "listbox"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DATE_PICKER = "date_picker"
    CALENDAR = "calendar"
    BUTTON = "button"
    TOOLBAR = "toolbar"
    TAB = "tab"
    POPUP = "popup"
    DIALOG = "dialog"
    TREE_VIEW = "tree_view"
    GRID = "grid"
    TABLE = "table"
    MENU = "menu"
    STATUS_BAR = "status_bar"
    SEARCH_BOX = "search_box"
    NAVIGATION = "navigation"
    SCROLLABLE = "scrollable"
    FILE_UPLOAD = "file_upload"
    LABEL = "label"
    SECTION = "section"
    UNKNOWN = "unknown"


EDITABLE_TYPES = {
    ElementType.TEXTBOX,
    ElementType.PASSWORD,
    ElementType.TEXTAREA,
    ElementType.COMBOBOX,
    ElementType.LISTBOX,
    ElementType.CHECKBOX,
    ElementType.RADIO,
    ElementType.DATE_PICKER,
    ElementType.CALENDAR,
    ElementType.SEARCH_BOX,
    ElementType.FILE_UPLOAD,
}


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in capture (client-area) coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def shifted(self, dx: int, dy: int) -> BBox:
        return BBox(self.x + dx, self.y + dy, self.width, self.height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> BBox:
        return cls(
            int(data["x"]),
            int(data["y"]),
            int(data.get("width", 0)),
            int(data.get("height", 0)),
        )

    def __repr__(self) -> str:
        return f"BBox({self.x},{self.y},{self.width}x{self.height})"


@dataclass
class OcrText:
    """A single OCR result line with its box."""

    text: str
    bbox: BBox
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": self.bbox.to_dict(), "confidence": self.confidence}


@dataclass
class ScreenElement:
    """A discovered UI element on the captured screen."""

    element_id: str
    type: ElementType
    label: str = ""
    name: str = ""
    bbox: BBox | None = None
    confidence: float = 1.0
    value: str | None = None
    required: bool | None = None
    disabled: bool | None = None
    section: str | None = None
    options: list[str] = field(default_factory=list)
    hint: str | None = None

    @property
    def editable(self) -> bool:
        return self.type in EDITABLE_TYPES and not self.disabled

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "type": self.type.value,
            "label": self.label,
            "name": self.name,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "value": self.value,
            "required": self.required,
            "disabled": self.disabled,
            "section": self.section,
            "options": list(self.options),
            "hint": self.hint,
        }


@dataclass
class Section:
    """A labelled region of the interface (group of fields)."""

    name: str
    bbox: BBox | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {"name": self.name, "bbox": self.bbox.to_dict() if self.bbox else None}


@dataclass
class SceneDescription:
    """The agent's structured understanding of one screen capture."""

    window_title: str = ""
    url: str | None = None
    layout_summary: str = ""
    sections: list[Section] = field(default_factory=list)
    elements: list[ScreenElement] = field(default_factory=list)
    confidence: float = 1.0
    provider: str = ""
    screen_offset: tuple[int, int] = (0, 0)  # (dx, dy) of capture origin on the physical screen

    @property
    def editable_fields(self) -> list[ScreenElement]:
        return [e for e in self.elements if e.editable]

    @property
    def buttons(self) -> list[ScreenElement]:
        return [e for e in self.elements if e.type == ElementType.BUTTON]

    def element(self, element_id: str) -> ScreenElement | None:
        for e in self.elements:
            if e.element_id == element_id:
                return e
        return None

    def to_dict(self) -> dict:
        return {
            "window_title": self.window_title,
            "url": self.url,
            "layout_summary": self.layout_summary,
            "sections": [s.to_dict() for s in self.sections],
            "elements": [e.to_dict() for e in self.elements],
            "confidence": self.confidence,
            "provider": self.provider,
            "screen_offset": list(self.screen_offset),
        }
