"""Tests for OCR image preprocessing (rotation/noise/contrast)."""

from __future__ import annotations

import numpy as np

from atlas.vision.preprocess import (
    deskew,
    denoise,
    enhance_contrast,
    preprocess,
    to_gray,
)


def _rgb_image(height: int = 60, width: int = 200) -> np.ndarray:
    """A synthetic RGB image with a dark text-like row on a light background."""
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    image[20:24, 40:140] = (10, 10, 10)  # a dark "text" bar
    return image


def test_to_gray_converts_rgb() -> None:
    image = _rgb_image()
    gray = to_gray(image)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray.shape == (60, 200)


def test_to_gray_passthrough_grayscale() -> None:
    gray = np.zeros((10, 10), dtype=np.uint8)
    assert to_gray(gray) is gray


def test_enhance_contrast_returns_same_shape() -> None:
    image = _rgb_image()
    out = enhance_contrast(image)
    assert out.shape == (60, 200)
    assert out.dtype == np.uint8
    # Contrast enhancement should produce at least one pure-dark pixel (text)
    # and keep the light background light.
    assert out.min() <= 60
    assert out.max() >= 200


def test_denoise_returns_same_shape() -> None:
    image = _rgb_image()
    out = denoise(image)
    assert out.shape == (60, 200)
    assert out.dtype == np.uint8


def test_deskew_keeps_upright_image_unaltered() -> None:
    # An upright grayscale image has no dominant skew -> returned unchanged.
    image = np.full((60, 200), 240, dtype=np.uint8)
    image[20:24, 40:140] = 10
    out = deskew(image)
    assert out.shape == (60, 200)
    assert np.array_equal(out, image)


def test_deskew_corrects_rotated_text() -> None:
    import numpy as _np

    # Build a rotated synthetic "text": 3px-thick black lines at a 5deg angle.
    h, w = 120, 240
    canvas = _np.full((h, w), 255, dtype=_np.uint8)
    for x in range(40, w - 20):
        y = int(60 + (x - 40) * 0.09)  # ~5.1 degrees
        for dy in (-1, 0, 1):
            yy = y + dy
            if 0 <= yy < h:
                canvas[yy, x] = 0
    out = deskew(canvas)
    assert out.shape == (120, 240)

    def _row_span(img: _np.ndarray) -> int:
        rows = {y for y in range(0, h) if int((img[y, :] == 0).sum()) > 0}
        return len(rows)

    # Deskew concentrates the diagonal line into far fewer rows (it becomes
    # horizontal). The span must drop meaningfully.
    assert _row_span(out) < _row_span(canvas)


def test_preprocess_full_pipeline() -> None:
    image = _rgb_image()
    out = preprocess(image)
    assert out.ndim == 2
    assert out.shape == (60, 200)


def test_preprocess_handles_none_and_garbage() -> None:
    assert preprocess(None) is None
    empty = np.zeros((0, 0), dtype=np.uint8)
    out = preprocess(empty)
    assert out.shape == (0, 0)


def test_ocr_config_preprocess_flag(monkeypatch) -> None:
    from atlas.config import OcrConfig

    monkeypatch.setenv("OCR_PREPROCESS", "false")
    config = OcrConfig()
    assert config.preprocess is False
    monkeypatch.setenv("OCR_PREPROCESS", "true")
    assert OcrConfig().preprocess is True
