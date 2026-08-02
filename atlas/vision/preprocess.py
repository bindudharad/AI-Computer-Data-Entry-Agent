"""OCR image preprocessing.

Improves OCR accuracy on noisy, rotated, or low-contrast captures without
changing the main vision pipeline. The transforms are cheap, deterministic, and
safe to apply before every OCR read:

- ``to_gray``: single-channel conversion.
- ``enhance_contrast``: adaptive contrast stretching (CLAHE) for faint text.
- ``denoise``: fast non-local means / median blur for sensor noise.
- ``deskew``: rotation correction via the Hough line angle, clamped to a sane
  range so an upright image is never distorted.
- ``preprocess``: combined pipeline in a sensible order.

All functions accept and return ``numpy.ndarray`` (grayscale or RGB) and never
raise - they return the input unchanged on failure.
"""

from __future__ import annotations

import numpy as np

from atlas.core.logging import logger


def _is_safe_shape(image: np.ndarray) -> bool:
    """True when the image has non-trivial dimensions we can actually process."""
    if image is None or getattr(image, "ndim", 0) < 2:
        return False
    h, w = image.shape[0], image.shape[1]
    return h >= 4 and w >= 4


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale (0..255 uint8)."""
    if not _is_safe_shape(image):
        return image
    try:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv_rgb_to_gray(image)
    except Exception as exc:
        logger.debug("to_gray failed: {}", exc)
    return image


def cv_rgb_to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE adaptive contrast enhancement for faint / uneven text."""
    if not _is_safe_shape(image):
        return image
    try:
        import cv2

        gray = to_gray(image)
        if gray is None or gray.ndim != 2:
            return image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        out = clahe.apply(gray)
        return out if out is not None else image
    except Exception as exc:
        logger.debug("enhance_contrast failed: {}", exc)
        return image


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Reduce sensor noise; falls back to a median blur if NLM is unavailable."""
    if not _is_safe_shape(image):
        return image
    try:
        import cv2

        gray = to_gray(image)
        if gray is None or gray.ndim != 2:
            return image
        try:
            out = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=strength, searchWindowSize=21)
        except cv2.error:
            out = cv2.medianBlur(gray, 3)
        return out if out is not None else image
    except Exception as exc:
        logger.debug("denoise failed: {}", exc)
        return image


def deskew(image: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """Rotate the image so text lines are horizontal (Hough-based).

    The rotation angle is clamped to ``max_angle`` degrees so a well-aligned
    capture is never needlessly distorted.
    """
    if not _is_safe_shape(image):
        return image
    try:
        import cv2

        gray = to_gray(image)
        if gray is None or gray.ndim != 2 or gray.shape[0] < 20 or gray.shape[1] < 20:
            return image
        angle = _estimate_angle(gray)
        if angle is None or abs(angle) < 0.05:
            return image
        if abs(angle) > max_angle:
            angle = max_angle if angle > 0 else -max_angle
        h, w = gray.shape
        center = (w / 2, h / 2)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, rotation, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.debug("deskew applied {:.2f}deg", angle)
        return rotated if rotated is not None else image
    except Exception as exc:
        logger.debug("deskew failed: {}", exc)
        return image


def _estimate_angle(gray: np.ndarray) -> float | None:
    """Return the dominant text-line angle in degrees, or None if unsure."""
    import cv2

    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=4)
        if lines is None:
            return None
        angles: list[float] = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (int(v) for v in line)
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            angles.append(angle)
        if not angles:
            return None
        from collections import Counter

        buckets = Counter(round(a) for a in angles)
        dominant = buckets.most_common(1)[0][0]
        # Lines near horizontal (0deg) or vertical (90deg) mean no skew.
        if abs(dominant) <= 2 or abs(abs(dominant) - 90) <= 2:
            return None
        return dominant if abs(dominant) <= 90 else dominant - 180.0
    except Exception:
        return None


def preprocess(image: np.ndarray, *, denoise_first: bool = True) -> np.ndarray:
    """Apply the full preprocessing pipeline in a safe, fixed order."""
    if image is None:
        return image
    gray = to_gray(image)
    if denoise_first:
        gray = denoise(gray)
        gray = enhance_contrast(gray)
        gray = deskew(gray)
    else:
        gray = enhance_contrast(gray)
        gray = denoise(gray)
        gray = deskew(gray)
    return gray


__all__ = ["to_gray", "enhance_contrast", "denoise", "deskew", "preprocess"]
