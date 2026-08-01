"""Developer-mode visual debug.

Renders detected fields, buttons and field names onto a screenshot so a human
can audit what the agent perceives. Colours follow the convention:

* green  - detected editable field
* blue   - current field being filled
* yellow - next field to be filled
* purple - buttons
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from atlas.core.logging import logger
from atlas.vision.models import BBox, SceneDescription

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

GREEN = (46, 204, 113)
BLUE = (52, 152, 219)
YELLOW = (241, 196, 15)
PURPLE = (155, 89, 182)
RED = (231, 76, 60)
WHITE = (255, 255, 255)


class SceneRenderer:
    """Draws annotated boxes and labels over a screenshot."""

    def __init__(self, font_path: str | None = None, font_size: int = 12) -> None:
        self._font_path = font_path
        self._font_size = font_size
        self._font: Any = None
        self._small_font: Any = None
        self._load_fonts()

    def _load_fonts(self) -> None:
        if ImageFont is None:
            return
        try:
            self._font = ImageFont.load_default()
            if hasattr(ImageFont, "truetype"):
                path = self._font_path
                if path is None:
                    for candidate in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
                        try:
                            self._font = ImageFont.truetype(candidate, self._font_size)
                            break
                        except OSError:
                            continue
        except Exception:  # pragma: no cover
            self._font = None

    def render(
        self,
        image: np.ndarray,
        scene: SceneDescription,
        current_id: str | None = None,
        next_id: str | None = None,
    ) -> np.ndarray:
        """Return a copy of the image annotated with all scene elements."""
        if Image is None or ImageDraw is None:
            return image.copy()
        pil = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(pil, "RGBA")

        for element in scene.elements:
            bbox = element.bbox
            if bbox is None:
                continue
            if element.element_id == current_id:
                colour = BLUE
            elif element.element_id == next_id:
                colour = YELLOW
            elif element.type.value in {"button", "radio"}:
                colour = PURPLE
            elif element.editable:
                colour = GREEN
            else:
                colour = RED if element.confidence < 0.4 else (128, 128, 128)
            self._draw_box(draw, bbox, colour)
            label = element.label or element.type.value
            self._draw_label(draw, bbox, label, colour)

        for section in scene.sections:
            if section.bbox:
                self._draw_box(draw, section.bbox, (52, 152, 219, 40), width=1, dashed=True)
        return np.array(pil)

    def _draw_box(self, draw, bbox: BBox, colour, width: int = 2, dashed: bool = False) -> None:
        if dashed:
            self._draw_dashed(draw, bbox, colour)
            return
        draw.rectangle(
            [bbox.left, bbox.top, bbox.right, bbox.bottom],
            outline=colour + (255,),
            width=width,
        )

    def _draw_dashed(self, draw, bbox: BBox, colour) -> None:
        dash = 6
        for x in range(bbox.left, bbox.right, dash * 2):
            draw.line([(x, bbox.top), (min(x + dash, bbox.right), bbox.top)], fill=colour + (255,), width=1)
            draw.line([(x, bbox.bottom), (min(x + dash, bbox.right), bbox.bottom)], fill=colour + (255,), width=1)
        for y in range(bbox.top, bbox.bottom, dash * 2):
            draw.line([(bbox.left, y), (bbox.left, min(y + dash, bbox.bottom))], fill=colour + (255,), width=1)
            draw.line([(bbox.right, y), (bbox.right, min(y + dash, bbox.bottom))], fill=colour + (255,), width=1)

    def _draw_label(self, draw, bbox: BBox, text: str, colour) -> None:
        text = (text or "")[:40]
        if not text:
            return
        font = self._font
        box = draw.textbbox((0, 0), text, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        bg = colour + (230,)
        x = bbox.left
        y = bbox.top - th - 2
        if y < 0:
            y = bbox.bottom + 2
        draw.rectangle([x, y, x + tw + 4, y + th + 2], fill=bg)
        draw.text((x + 2, y), text, fill=WHITE + (255,), font=font)

    def save(self, image: np.ndarray, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        logger.debug("debug render saved to {}", path)
        return path


__all__ = ["SceneRenderer", "GREEN", "BLUE", "YELLOW", "PURPLE", "RED"]
