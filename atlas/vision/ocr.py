"""OCR fallback.

OCR is used ONLY when the agent explicitly requests reading tiny printed text
(e.g. small captions, confirmation codes) and never as part of the primary
scene understanding channel. Two backends are supported behind one interface:
PaddleOCR (default) and Tesseract.

Robustness contract: the vision pipeline must keep running even when an OCR
backend is broken or unavailable. ``read_image`` never raises - on any failure
it logs and returns an empty result. Engine construction caches failures so a
broken backend is retried at most once per process, never every frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from atlas.config import OcrConfig
from atlas.core.logging import logger, ocr_logger
from atlas.vision.models import BBox, OcrText
from atlas.vision.preprocess import preprocess as _preprocess


class OcrReader(ABC):
    """Interface for the local OCR fallback engine."""

    name = "abstract"

    @abstractmethod
    def read_image(self, image: np.ndarray) -> list[OcrText]:
        """Return OCR lines with bounding boxes and confidence."""

    def read_region(self, image: np.ndarray, region: BBox) -> list[OcrText]:
        """OCR a sub-region; returned boxes are offset to the full image."""
        crop = image[region.top : region.bottom, region.left : region.right]
        if crop.size == 0:
            return []
        lines = self.read_image(crop)
        for line in lines:
            line.bbox = line.bbox.shifted(region.left, region.top)
        ocr_logger.debug(
            "ocr_region {}x{}@({},{}) -> {} line(s) [{}]",
            region.width, region.height, region.left, region.top, len(lines), self.name,
        )
        return lines

    def close(self) -> None:
        pass

    def _maybe_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply OCR preprocessing unless the config disables it."""
        if not getattr(self, "_preprocess", True):
            return image
        try:
            return _preprocess(image)
        except Exception as exc:
            logger.debug("ocr preprocess skipped: {}", exc)
            return image


class PaddleOcrReader(OcrReader):
    """PaddleOCR backend.

    PaddleOCR 2.9.x forwards constructor kwargs to the paddle engine. Paddle
    3.0 removed ``show_log`` (raising ``Unknown argument: show_log``), so this
    reader never passes it. Construction tries progressively simpler argument
    sets, and inference falls back from the deprecated ``ocr()`` API to the
    v3 ``predict()`` API.
    """

    name = "paddle"

    def __init__(self, config: OcrConfig | None = None) -> None:
        self._config = config or OcrConfig()
        self._engine: Any = None
        self._engine_failed: bool = False
        self._available: bool | None = None
        self._preprocess: bool = self._config.preprocess

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_failed:
            raise RuntimeError("PaddleOCR unavailable after previous failure")
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            self._engine_failed = True
            self._available = False
            raise RuntimeError(
                "PaddleOCR not installed - run `pip install -r requirements-optional.txt`"
            )
        # PaddleOCR 3.x rejects unknown kwargs (show_log, use_angle_cls, etc.)
        # and requires a valid engine. Try multiple engine configurations:
        # 1. onnxruntime (no paddlepaddle needed)
        # 2. Default (no engine specified, may auto-detect)
        # 3. Minimal args
        engines_to_try = ["onnxruntime", None]
        last_error: Exception | None = None
        for engine in engines_to_try:
            kwargs: dict[str, Any] = {"lang": self._config.lang}
            if engine is not None:
                kwargs["engine"] = engine
            try:
                self._engine = PaddleOCR(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - backend-specific failures
                last_error = exc
                continue
        if self._engine is None:
            # Final attempt: completely empty init
            try:
                self._engine = PaddleOCR()
            except Exception as exc:
                last_error = exc
        if self._engine is None:
            self._engine_failed = True
            self._available = False
            logger.warning("PaddleOCR initialisation failed: {}", last_error)
            raise RuntimeError(f"could not initialise PaddleOCR: {last_error}")
        self._available = True
        logger.debug("paddleocr engine ready (lang={})", self._config.lang)
        return self._engine

    @property
    def available(self) -> bool:
        if self._available is None and not self._engine_failed:
            try:
                self._ensure_engine()
            except Exception:
                self._available = False
        return bool(self._available)

    def read_image(self, image: np.ndarray) -> list[OcrText]:
        """OCR an image; never raises (empty result on failure)."""
        try:
            engine = self._ensure_engine()
        except Exception as exc:
            logger.debug("paddleocr unavailable: {}", exc)
            return []
        try:
            prepared = self._maybe_preprocess(image)
        except Exception:
            prepared = image
        if prepared.ndim == 2:
            prepared = np.stack([prepared] * 3, axis=-1)
        bgr = prepared[:, :, ::-1].copy()
        result: Any = None
        try:
            result = engine.ocr(bgr, cls=True)
        except (AttributeError, TypeError):
            pass
        except Exception as exc:
            logger.debug("paddleocr ocr() failed, trying predict(): {}", exc)
            result = None
        if result is None:
            try:
                result = engine.predict(bgr)
            except Exception as exc:
                logger.debug("paddleocr predict() failed: {}", exc)
                return []
        return self._parse(result)

    def _parse(self, result: Any) -> list[OcrText]:
        lines: list[OcrText] = []
        if not result:
            return lines

        # PaddleOCR v3 predict(): list[dict] (one dict per image).
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return self._parse_dict(result[0])

        # PaddleOCR v3 ocr()/predict() single dict.
        if isinstance(result, dict):
            return self._parse_dict(result)

        # PaddleOCR v2 list-of-pages format.
        page = result[0]
        if not page:
            return lines
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            quad, text_conf = item[0], item[1]
            text, score = text_conf
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            if xs and ys:
                box = BBox(int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
                lines.append(OcrText(text=str(text), bbox=box, confidence=float(score)))
        return lines

    @staticmethod
    def _parse_dict(data: dict) -> list[OcrText]:
        lines: list[OcrText] = []
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polys = data.get("rec_polys") or []
        for text, score, poly in zip(texts, scores, polys, strict=False):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if xs and ys:
                box = BBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                lines.append(OcrText(text=str(text), bbox=box, confidence=float(score)))
        return lines


class TesseractOcrReader(OcrReader):
    """Tesseract backend (requires pytesseract + binary)."""

    name = "tesseract"

    def __init__(self, config: OcrConfig | None = None) -> None:
        self._config = config or OcrConfig()
        self._available: bool | None = None
        self._ensure_failed: bool = False
        self._preprocess: bool = self._config.preprocess

    def _ensure(self) -> Any:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            self._ensure_failed = True
            self._available = False
            raise RuntimeError("pytesseract not installed") from exc
        self._available = True
        return pytesseract

    @property
    def available(self) -> bool:
        if self._available is None and not self._ensure_failed:
            try:
                self._ensure()
            except Exception:
                self._available = False
        return bool(self._available)

    def read_image(self, image: np.ndarray) -> list[OcrText]:
        from PIL import Image

        try:
            pytesseract = self._ensure()
            prepared = self._maybe_preprocess(image)
            pil = Image.fromarray(prepared)
            data = pytesseract.image_to_data(pil, lang=self._config.lang, output_type="dict")
        except Exception as exc:  # noqa: BLE001 - backend-specific failures
            logger.debug("tesseract read failed: {}", exc)
            return []
        lines: list[OcrText] = []
        for text, conf, left, top, w, h in zip(
            data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"], strict=False
        ):
            text = (text or "").strip()
            if not text:
                continue
            try:
                score = float(conf) / 100.0
            except (TypeError, ValueError):
                score = 0.0
            if score < self._config.confidence_threshold:
                continue
            lines.append(OcrText(text=text, bbox=BBox(left, top, w, h), confidence=score))
        return lines


class NoopOcrReader(OcrReader):
    """No-op backend (OCR disabled)."""

    name = "none"

    def read_image(self, image: np.ndarray) -> list[OcrText]:
        return []


_READER_CACHE: dict[tuple, OcrReader] = {}


def create_ocr_reader(config: OcrConfig | None = None) -> OcrReader:
    """Factory for the configured OCR fallback.

    Readers are cached per (engine, lang) so model weights load once per
    process and are reused across captures.
    """
    config = config or OcrConfig()
    key = (config.engine, config.lang)
    cached = _READER_CACHE.get(key)
    if cached is not None:
        return cached
    if config.engine == "paddle":
        reader: OcrReader = PaddleOcrReader(config)
    elif config.engine == "tesseract":
        reader = TesseractOcrReader(config)
    else:
        reader = NoopOcrReader()
    _READER_CACHE[key] = reader
    return reader


__all__ = ["OcrReader", "PaddleOcrReader", "TesseractOcrReader", "create_ocr_reader"]
