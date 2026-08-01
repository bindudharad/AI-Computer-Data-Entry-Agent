"""Scene orchestration.

Coordinates the capture -> VLM analysis -> validation pipeline. Also provides
the explicit "read tiny text" OCR fallback channel that the agent can request.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from atlas.core.logging import logger
from atlas.vision.capture import ClientArea, WindowCapture
from atlas.vision.models import BBox, OcrText, SceneDescription
from atlas.vision.providers import VisionProvider


@dataclass
class SceneAnalysis:
    """A scene plus the capture it was derived from and its timestamp."""

    scene: SceneDescription
    capture: ClientArea | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "capture": {
                "left": self.capture.left,
                "top": self.capture.top,
                "width": self.capture.width,
                "height": self.capture.height,
            }
            if self.capture
            else None,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class SceneAnalyzer:
    """Analyses client-area captures into structured scenes.

    A short-lived cache keys scenes by a perceptual image hash so repeated
    captures of an unchanged screen do not re-invoke the VLM.
    """

    def __init__(self, provider: VisionProvider, cache_ttl: float = 2.0, cache_size: int = 32) -> None:
        self._provider = provider
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, SceneDescription]] = {}

    @property
    def provider(self) -> VisionProvider:
        return self._provider

    def analyze(self, capture: ClientArea, window_title: str = "", url: str | None = None) -> SceneAnalysis:
        start = time.perf_counter()
        key = self._hash(capture.image)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[0]) <= self._cache_ttl:
            scene = cached[1]
        else:
            scene = self._provider.describe(capture.image, window_title=window_title, url=url)
            scene.screen_offset = capture.offset
            self._cache[key] = (now, scene)
        duration = (time.perf_counter() - start) * 1000.0
        return SceneAnalysis(scene=scene, capture=capture, duration_ms=duration)

    def invalidate(self) -> None:
        self._cache.clear()

    def read_text(self, capture: ClientArea | np.ndarray, region: BBox | None = None) -> list[OcrText]:
        """Explicitly read tiny printed text (AI-requested OCR fallback).

        Never raises: a failing OCR backend degrades to an empty result so the
        agent can continue.
        """
        image = capture.image if isinstance(capture, ClientArea) else capture
        try:
            if region is not None:
                crop = image[region.top : region.bottom, region.left : region.right]
                if crop.size == 0:
                    return []
                lines = self._provider.read_text(crop)
                for line in lines:
                    line.bbox = line.bbox.shifted(region.left, region.top)
                return lines
            return self._provider.read_text(image)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the loop
            logger.warning("read_text failed: {}", exc)
            return []

    @staticmethod
    def _hash(image: np.ndarray) -> str:
        small = image[::8, ::8]
        return hashlib.md5(np.ascontiguousarray(small).tobytes()).hexdigest()


class WindowSceneSource:
    """Combines an attached WindowCapture with a SceneAnalyzer.

    This is the object the agent's observation loop drives: capture the client
    area of the attached window, then analyse it with the VLM.
    """

    def __init__(self, capture: WindowCapture, analyzer: SceneAnalyzer) -> None:
        self._capture = capture
        self._analyzer = analyzer

    @property
    def capture(self) -> WindowCapture:
        return self._capture

    @property
    def analyzer(self) -> SceneAnalyzer:
        return self._analyzer

    @property
    def attached(self) -> bool:
        return self._capture.attached

    def observe(self) -> SceneAnalysis | None:
        """Capture the client area and analyse it. Returns None if not attached."""
        area = self._capture.capture_until_nonempty()
        if area is None:
            return None
        analysis = self._analyzer.analyze(area, window_title=self._capture.title)
        return analysis

    def close(self) -> None:
        self._capture.close()


__all__ = ["SceneAnalyzer", "SceneAnalysis", "WindowSceneSource"]
