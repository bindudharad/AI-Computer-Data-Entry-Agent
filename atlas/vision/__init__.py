"""Vision engine: VLM-first perception with OCR as an explicit fallback."""

from atlas.vision.capture import ClientArea, ScreenGrabber, WindowCapture, WindowGeometry
from atlas.vision.models import BBox, ElementType, OcrText, SceneDescription, ScreenElement, Section
from atlas.vision.ocr import OcrReader, create_ocr_reader
from atlas.vision.providers import (
    GeminiVisionProvider,
    MockVisionProvider,
    OpenAIVisionProvider,
    RuleVisionProvider,
    VisionProvider,
    create_vision_provider,
)
from atlas.vision.scene import SceneAnalysis, SceneAnalyzer, WindowSceneSource

__all__ = [
    "ClientArea",
    "ScreenGrabber",
    "WindowCapture",
    "WindowGeometry",
    "BBox",
    "ElementType",
    "OcrText",
    "SceneDescription",
    "ScreenElement",
    "Section",
    "OcrReader",
    "create_ocr_reader",
    "VisionProvider",
    "OpenAIVisionProvider",
    "GeminiVisionProvider",
    "RuleVisionProvider",
    "MockVisionProvider",
    "create_vision_provider",
    "SceneAnalyzer",
    "SceneAnalysis",
    "WindowSceneSource",
]
