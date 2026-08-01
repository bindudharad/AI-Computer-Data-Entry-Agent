"""Vision providers.

The Vision Language Model (VLM) is the PRIMARY perception channel. It turns a
screenshot of the attached window's client area into a structured
``SceneDescription`` (elements, labels, types, layout, sections, options).

Providers
---------
* ``OpenAIVisionProvider`` - any OpenAI-compatible vision chat-completions API
  (OpenAI, DeepSeek-VL, Ollama, vLLM, ...).
* ``GeminiVisionProvider`` - Google Gemini (REST ``generateContent``).
* ``RuleVisionProvider``  - deterministic OpenCV fallback (no network, no key).
  Used only when no VLM endpoint is configured; may optionally call a local OCR
  engine to read text inside detected boxes. This is the *degraded* mode.
* ``MockVisionProvider``  - deterministic synthetic scenes (tests / offline dev).

OCR is never part of ``describe`` for VLM providers; it is invoked only via the
explicit ``read_text`` channel (see ``atlas.vision.ocr``).
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from atlas.config import VisionConfig
from atlas.core.logging import logger
from atlas.vision.models import BBox, ElementType, OcrText, SceneDescription, ScreenElement, Section

try:
    import cv2
except ImportError:  # pragma: no cover - optional
    cv2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(image: np.ndarray, fmt: str = "png") -> str:
    """Encode an RGB numpy image to a base64 data URL."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    ok, buf = cv2.imencode(f".{fmt}", image[:, :, ::-1])  # RGB -> BGR for cv2
    if not ok:
        raise ValueError("failed to encode screenshot")
    return "data:image/{};base64,{}".format(fmt, base64.b64encode(buf.tobytes()).decode("ascii"))


def _safe_json(text: str) -> dict | None:
    """Parse JSON from an LLM response, tolerating ```json fences and trailing text."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _to_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _element_type(raw: Any) -> ElementType:
    if raw is None:
        return ElementType.UNKNOWN
    text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return ElementType(text)
    except ValueError:
        aliases = {
            "text_box": ElementType.TEXTBOX,
            "input": ElementType.TEXTBOX,
            "text_field": ElementType.TEXTBOX,
            "password_box": ElementType.PASSWORD,
            "select": ElementType.COMBOBOX,
            "dropdown": ElementType.COMBOBOX,
            "drop_down": ElementType.COMBOBOX,
            "date": ElementType.DATE_PICKER,
            "datepicker": ElementType.DATE_PICKER,
            "check_box": ElementType.CHECKBOX,
            "radio_button": ElementType.RADIO,
            "radios": ElementType.RADIO,
            "submit": ElementType.BUTTON,
            "push_button": ElementType.BUTTON,
        }
        return aliases.get(text, ElementType.UNKNOWN)


def _parse_scene(raw: dict, provider: str, offset: tuple[int, int]) -> SceneDescription:
    """Build a SceneDescription from a (possibly ragged) VLM JSON dict."""
    elements: list[ScreenElement] = []
    for i, item in enumerate(raw.get("elements", raw.get("fields", []) or [])):
        if not isinstance(item, dict):
            continue
        box = item.get("bbox") or item.get("box") or item.get("bounds")
        bbox = None
        if isinstance(box, dict) and "x" in box:
            bbox = BBox(
                _to_int(box.get("x")),
                _to_int(box.get("y")),
                _to_int(box.get("width") or box.get("w")),
                _to_int(box.get("height") or box.get("h")),
            )
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            bbox = BBox(*[int(v) for v in box])
        elif isinstance(box, dict) and "left" in box:
            bbox = BBox(
                _to_int(box.get("left")),
                _to_int(box.get("top")),
                _to_int(box.get("right")) - _to_int(box.get("left")),
                _to_int(box.get("bottom")) - _to_int(box.get("top")),
            )
        element = ScreenElement(
            element_id=str(item.get("id", item.get("element_id", f"e{i}"))),
            type=_element_type(item.get("type")),
            label=str(item.get("label", "") or ""),
            name=str(item.get("name", "") or ""),
            bbox=bbox,
            confidence=_to_float(item.get("confidence", 1.0), 1.0),
            value=(str(item["value"]) if item.get("value") is not None else None),
            required=item.get("required") if item.get("required") is not None else None,
            disabled=item.get("disabled") if item.get("disabled") is not None else None,
            section=str(item.get("section", "") or None),
            options=[str(o) for o in (item.get("options") or [])],
            hint=str(item.get("hint", "") or None),
        )
        if bbox is not None and bbox.width > 0 and bbox.height > 0:
            elements.append(element)

    sections = [
        Section(
            name=str(s.get("name", s.get("title", "section")) or "section"),
            bbox=BBox(_to_int(s.get("x")), _to_int(s.get("y")),
                      _to_int(s.get("width") or s.get("w")),
                      _to_int(s.get("height") or s.get("h")))
            if isinstance(s, dict) and s.get("x") is not None
            else None,
            confidence=_to_float(s.get("confidence", 1.0), 1.0),
        )
        for s in (raw.get("sections") or [])
        if isinstance(s, dict)
    ]

    return SceneDescription(
        window_title=str(raw.get("window_title", raw.get("title", "")) or ""),
        url=str(raw.get("url")) or None,
        layout_summary=str(raw.get("layout_summary", raw.get("layout", "")) or ""),
        sections=sections,
        elements=elements,
        confidence=_to_float(raw.get("confidence", 1.0), 1.0),
        provider=provider,
        screen_offset=offset,
    )


# ---------------------------------------------------------------------------
# Vision provider interface
# ---------------------------------------------------------------------------

class VisionProvider(ABC):
    """Interface for turning screenshots into structured scene descriptions."""

    name = "abstract"

    @abstractmethod
    def describe(self, image: np.ndarray, window_title: str = "", url: str | None = None) -> SceneDescription:
        """Analyse a screenshot and return the structured scene."""

    def read_text(self, image: np.ndarray) -> list[OcrText]:
        """Read all visible text in the image (explicit request only)."""
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# OpenAI-compatible VLM
# ---------------------------------------------------------------------------

class OpenAIVisionProvider(VisionProvider):
    """Any OpenAI-compatible vision chat-completions endpoint."""

    name = "openai"

    SCENE_PROMPT = """You are the perception engine of a desktop automation agent.
Analyse the screenshot of a desktop application's client area and return STRICT JSON
with this exact shape:
{
  "window_title": string,
  "layout_summary": "one sentence describing the layout",
  "sections": [{"name": string, "x": int, "y": int, "width": int, "height": int, "confidence": number}],
  "elements": [
    {
      "id": string, "type": string, "label": string, "name": string,
      "x": int, "y": int, "width": int, "height": int,
      "confidence": number, "value": string|null, "required": boolean|null,
      "disabled": boolean|null, "section": string|null,
      "options": [string], "hint": string|null
    }
  ],
  "confidence": number
}
Rules:
- Coordinates are pixel offsets relative to the top-left of the image.
- element types: textbox, password, textarea, combobox, listbox, checkbox, radio,
  date_picker, calendar, button, toolbar, tab, dialog, tree_view, grid, table, menu,
  status_bar, search_box, navigation, label.
- 'label' is the visible text next to/above a control (e.g. "Date of Birth").
- For combobox/listbox include plausible 'options' if visible.
- Report ONLY elements you can actually see. Never invent fields.
- Include all buttons with their visible text as 'label' and type 'button'.
- Return valid JSON only, no markdown, no commentary."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._base = (config.api_base or "https://api.openai.com/v1").rstrip("/")
        self._model = config.model or "gpt-4o-mini"

    def describe(self, image: np.ndarray, window_title: str = "", url: str | None = None) -> SceneDescription:
        prompt = self.SCENE_PROMPT + (
            f"\nKnown window title: {window_title!r}."
            if window_title else ""
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
        ]
        raw = self._chat(content)
        data = _safe_json(raw) or {}
        if not data:
            logger.warning("VLM returned unparseable scene JSON")
        return _parse_scene(data, self.name, (0, 0))

    def read_text(self, image: np.ndarray) -> list[OcrText]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "List every visible text line as JSON: "
                                     '{"lines":[{"text":string,"x":int,"y":int,"width":int,"height":int}]}. '
                                     "Return valid JSON only."},
            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
        ]
        raw = self._chat(content)
        data = _safe_json(raw) or {}
        out: list[OcrText] = []
        for item in data.get("lines") or []:
            if isinstance(item, dict) and item.get("text"):
                out.append(OcrText(
                    text=str(item["text"]),
                    bbox=BBox(_to_int(item.get("x")), _to_int(item.get("y")),
                              _to_int(item.get("width", 0)), _to_int(item.get("height", 0))),
                    confidence=_to_float(item.get("confidence", 1.0), 1.0),
                ))
        return out

    def _chat(self, content: list[dict[str, Any]]) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        resp = requests.post(
            f"{self._base}/chat/completions", headers=headers, json=payload, timeout=self._config.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Google Gemini VLM
# ---------------------------------------------------------------------------

class GeminiVisionProvider(VisionProvider):
    """Google Gemini vision via the REST generateContent endpoint."""

    name = "gemini"

    SCENE_PROMPT = OpenAIVisionProvider.SCENE_PROMPT

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._model = config.model or "gemini-2.0-flash"

    def describe(self, image: np.ndarray, window_title: str = "", url: str | None = None) -> SceneDescription:
        if not self._config.api_key:
            raise RuntimeError("GeminiVisionProvider requires VISION_API_KEY")
        import requests

        prompt = self.SCENE_PROMPT + (f"\nKnown window title: {window_title!r}." if window_title else "")
        b64 = _encode_image(image).split(",", 1)[1]
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
        }
        base = (self._config.api_base or "https://generativelanguage.googleapis.com").rstrip("/")
        resp = requests.post(
            f"{base}/v1beta/models/{self._model}:generateContent",
            headers={"x-goog-api-key": self._config.api_key},
            json=payload,
            timeout=self._config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        scene = _parse_scene(_safe_json(text) or {}, self.name, (0, 0))
        return scene


# ---------------------------------------------------------------------------
# Deterministic OpenCV fallback (no network / no key)
# ---------------------------------------------------------------------------

class RuleVisionProvider(VisionProvider):
    """Heuristic scene understanding used when no VLM is configured.

    Detects text lines and bordered input regions with morphology, groups them
    into label/value pairs, and classifies obvious buttons. Optionally enriches
    text via the injected OCR reader. Deliberately reports lower confidence.
    """

    name = "rule"

    def __init__(self, ocr_reader: Any | None = None) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for the rule vision provider")
        self._ocr = ocr_reader

    def describe(self, image: np.ndarray, window_title: str = "", url: str | None = None) -> SceneDescription:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Text lines: close horizontal gaps then dilate vertically.
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)
        text_blocks = self._contours(lines)

        # Bordered input boxes: white interior with dark border.
        edged = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        boxes = self._contours(dilated)
        inputs = [b for b in boxes if self._looks_like_input(b, image.shape)]

        elements: list[ScreenElement] = []
        idx = 0
        for bbox in inputs:
            x, y, w, h = bbox
            elements.append(ScreenElement(
                element_id=f"field{idx}",
                type=ElementType.TEXTBOX,
                label="",
                bbox=BBox(x, y, w, h),
                confidence=0.5,
            ))
            idx += 1

        # Buttons: filled blocks with text.
        for bbox in self._contours(cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))):
            x, y, w, h = bbox
            if not (10 <= w <= 300 and 12 <= h <= 80):
                continue
            region = gray[y : y + h, x : x + w]
            if region.size == 0:
                continue
            if float(np.mean(region)) < 210:
                elements.append(ScreenElement(
                    element_id=f"btn{idx}", type=ElementType.BUTTON, label="",
                    bbox=BBox(x, y, w, h), confidence=0.45,
                ))
                idx += 1

        # Text line boxes become labels if no input overlaps them.
        input_boxes = [e.bbox for e in elements if e.bbox]
        for bbox in text_blocks:
            x, y, w, h = bbox
            if w < 30 or h < 8:
                continue
            if any(self._overlaps(BBox(x, y, w, h), ib) for ib in input_boxes):
                continue
            elements.append(ScreenElement(
                element_id=f"label{idx}", type=ElementType.LABEL, label="",
                bbox=BBox(x, y, w, h), confidence=0.35,
            ))
            idx += 1

        if self._ocr is not None:
            try:
                words = self._ocr.read_image(image)
                self._attach_text(elements, words)
            except Exception as exc:
                logger.debug("rule vision text enrichment failed: {}", exc)

        scene = SceneDescription(
            window_title=window_title,
            layout_summary="Heuristic (no VLM configured) scene understanding.",
            elements=[e for e in elements if e.bbox and e.bbox.area > 0],
            confidence=0.45,
            provider=self.name,
        )
        scene.sections = [Section(name="main", bbox=BBox(0, 0, image.shape[1], image.shape[0]))]
        return scene

    def _attach_text(self, elements: list[ScreenElement], words: list[OcrText]) -> None:
        for word in words:
            best = None
            best_overlap = 0.0
            for e in elements:
                if e.bbox is None:
                    continue
                inter = self._intersection(e.bbox, word.bbox)
                if inter and inter.area > best_overlap:
                    best_overlap = inter.area
                    best = e
            if best is not None:
                best.label = (best.label + " " + word.text).strip()

    @staticmethod
    def _contours(binary) -> list[tuple[int, ...]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [tuple(cv2.boundingRect(c)) for c in contours]

    @staticmethod
    def _looks_like_input(bbox: tuple[int, ...], shape: tuple[int, ...]) -> bool:
        x, y, w, h = bbox
        height, width = shape[:2]
        if not (8 <= w <= int(width * 0.6) and 10 <= h <= 60):
            return False
        if not (0 <= x <= width and 0 <= y <= height):
            return False
        return w / max(h, 1) > 1.2

    @staticmethod
    def _overlaps(a: BBox, b: BBox) -> bool:
        return not (a.right < b.left or b.right < a.left or a.bottom < b.top or b.bottom < a.top)

    @staticmethod
    def _intersection(a: BBox, b: BBox) -> BBox | None:
        x = max(a.left, b.left)
        y = max(a.top, b.top)
        w = min(a.right, b.right) - x
        h = min(a.bottom, b.bottom) - y
        if w <= 0 or h <= 0:
            return None
        return BBox(x, y, w, h)


# ---------------------------------------------------------------------------
# Synthetic provider (tests / offline development)
# ---------------------------------------------------------------------------

class MockVisionProvider(VisionProvider):
    """Deterministic scene provider backed by a registry of synthetic scenes.

    Tests register a scene per window title; ``describe`` returns the registered
    scene with coordinates offset to match the synthetic canvas.
    """

    name = "mock"

    def __init__(self) -> None:
        self._registry: dict[str, SceneDescription] = {}

    def register(self, window_title: str, scene: SceneDescription) -> None:
        self._registry[window_title.lower()] = scene

    def clear(self) -> None:
        self._registry.clear()

    def describe(self, image: np.ndarray, window_title: str = "", url: str | None = None) -> SceneDescription:
        scene = self._registry.get(window_title.lower())
        if scene is None:
            return SceneDescription(
                window_title=window_title,
                layout_summary="No synthetic scene registered.",
                confidence=0.0,
                provider=self.name,
            )
        return scene


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vision_provider(config: VisionConfig, ocr_reader: Any | None = None) -> VisionProvider:
    """Instantiate the configured vision provider.

    ``auto`` selects the first available provider in order: configured VLM
    provider -> rule-based fallback. ``mock`` returns the synthetic provider.
    """
    provider = config.provider.lower()
    if provider == "mock":
        return MockVisionProvider()
    if provider in {"openai", "auto"} and (config.api_key or config.api_base):
        return OpenAIVisionProvider(config)
    if provider in {"gemini"} and config.api_key:
        return GeminiVisionProvider(config)
    if provider in {"local"}:
        return RuleVisionProvider(ocr_reader)
    if provider == "auto":
        logger.info("No VLM endpoint configured - using rule-based vision fallback")
        return RuleVisionProvider(ocr_reader)
    raise ValueError(f"Unknown vision provider: {provider}")


def wait_until_provider_ready(provider: VisionProvider, timeout: float = 0.0) -> VisionProvider:
    """Placeholder hook for providers that need async warm-up."""
    return provider


__all__ = [
    "VisionProvider",
    "OpenAIVisionProvider",
    "GeminiVisionProvider",
    "RuleVisionProvider",
    "MockVisionProvider",
    "create_vision_provider",
    "_parse_scene",
]
