"""Verification of executed actions.

Every value-producing action is verified before the agent continues. Multiple
strategies are composed:

1. target adapter verification (e.g. DOM ``value`` for web targets),
2. clipboard read-back (select-all + copy on desktop text fields),
3. vision read-back (OCR the field region after the action).

Verification failure never continues blindly - the executor retries and, if it
still fails, the recovery planner decides what to do.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from atlas.act.clipboard import ClipboardEngine
from atlas.act.keyboard import HumanKeyboard
from atlas.core.logging import logger
from atlas.vision.models import BBox, OcrText


class FieldVerifier(ABC):
    """A verification strategy."""

    name = "abstract"

    @abstractmethod
    def verify(self, bbox: BBox | None, expected: str, field_id: str | None = None) -> tuple[bool, str]:
        """Return (matched, evidence)."""


#: Boolean value synonyms collapsed to "1"/"0" before comparison, so checkbox
#: and radio verification is robust across sources ("Yes"/"checked"/"on"/"true").
_TRUE_VALUES = {"yes", "y", "true", "t", "1", "on", "checked", "selected", "x"}
_FALSE_VALUES = {"no", "n", "false", "f", "0", "off", "unchecked", "unselected", ""}


def normalize_for_compare(value: str) -> str:
    """Normalize values for comparison: strip, collapse spaces, lowercase."""
    text = " ".join(str(value).strip().split())
    lowered = text.lower()
    if lowered in _TRUE_VALUES:
        return "1"
    if lowered in _FALSE_VALUES:
        return "0"
    return re.sub(r"[\s_\-]+", " ", lowered).strip()


def _file_match(actual: str, expected: str) -> bool:
    """Match file-upload read-backs against an expected path.

    Browsers expose file inputs as ``C:\\fakepath\\<basename>`` (and the
    separator can vary), so compare the basenames case-insensitively and also
    accept the expected path appearing anywhere in the read-back.
    """
    if not actual or not expected:
        return False
    expected_lower = expected.lower().replace("\\", "/")
    actual_lower = actual.lower().replace("\\", "/")
    if expected_lower in actual_lower:
        return True
    expected_base = expected_lower.rsplit("/", 1)[-1]
    actual_base = actual_lower.rsplit("/", 1)[-1]
    return bool(expected_base) and expected_base == actual_base


class TargetFieldVerifier(FieldVerifier):
    """Delegates verification to the target adapter (e.g. DOM value read)."""

    name = "target"

    def __init__(self, get_value: Callable[[Any], str | None]) -> None:
        self._get_value = get_value

    def verify(self, bbox: BBox | None, expected: str, field_id: str | None = None) -> tuple[bool, str]:
        try:
            actual = self._get_value(field_id)
        except Exception as exc:
            return False, f"target read failed: {exc}"
        if actual is None:
            return False, "no value available"
        return self._compare(actual, expected)

    @staticmethod
    def _compare(actual: str, expected: str) -> tuple[bool, str]:
        a, e = normalize_for_compare(actual), normalize_for_compare(expected)
        if a == e:
            return True, f"target read-back matched ({actual!r})"
        if e and e in a:
            return True, f"target read-back contains expected ({actual!r})"
        if _file_match(actual, expected):
            return True, f"target read-back file matched ({actual!r})"
        return False, f"target read-back mismatch: got {actual!r}, expected {expected!r}"


class ClipboardVerifier(FieldVerifier):
    """Select-all + copy and compare with the expected value."""

    name = "clipboard"

    def __init__(self, keyboard: HumanKeyboard, clipboard: ClipboardEngine) -> None:
        self._keyboard = keyboard
        self._clipboard = clipboard

    def verify(self, bbox: BBox | None, expected: str, field_id: str | None = None) -> tuple[bool, str]:
        try:
            actual = self._clipboard.read_focused()
        except Exception as exc:
            return False, f"clipboard read failed: {exc}"
        return self._compare(actual, expected)

    @staticmethod
    def _compare(actual: str, expected: str) -> tuple[bool, str]:
        a, e = normalize_for_compare(actual), normalize_for_compare(expected)
        if a == e:
            return True, f"clipboard matched ({actual!r})"
        if e and e in a:
            return True, f"clipboard contains expected ({actual!r})"
        return False, f"clipboard mismatch: got {actual!r}, expected {expected!r}"


class VisionVerifier(FieldVerifier):
    """Re-captures the field region and OCR-reads it for comparison.

    OCR is used here deliberately - this is the explicit "read what's in the
    field" request, not part of scene understanding.
    """

    name = "vision"

    def __init__(self, read_region: Callable[[BBox], list[OcrText]]) -> None:
        """``read_region(image_source, bbox)`` returns OCR lines for the region."""
        self._read_region = read_region

    def verify(self, bbox: BBox | None, expected: str, field_id: str | None = None) -> tuple[bool, str]:
        if bbox is None:
            return False, "no bbox for vision verification"
        try:
            lines = self._read_region(bbox)
        except Exception as exc:
            return False, f"vision read failed: {exc}"
        actual = " ".join(line.text for line in lines)
        if not actual:
            return False, "vision read empty"
        return self._compare(actual, expected)

    @staticmethod
    def _compare(actual: str, expected: str) -> tuple[bool, str]:
        a, e = normalize_for_compare(actual), normalize_for_compare(expected)
        if a == e:
            return True, f"vision matched ({actual!r})"
        if e and e in a:
            return True, f"vision contains expected ({actual!r})"
        return False, f"vision mismatch: got {actual!r}, expected {expected!r}"


class CompositeVerifier:
    """Runs verification strategies in order until one matches."""

    def __init__(self, verifiers: list[FieldVerifier] | None = None) -> None:
        self._verifiers: list[FieldVerifier] = verifiers or []

    def add(self, verifier: FieldVerifier) -> None:
        self._verifiers.append(verifier)

    @property
    def strategies(self) -> list[str]:
        return [v.name for v in self._verifiers]

    def verify(self, bbox: BBox | None, expected: str, field_id: str | None = None) -> tuple[bool, str]:
        if not self._verifiers:
            logger.debug("no verifier strategies configured - verification skipped")
            return False, "no verifier configured"
        failures: list[str] = []
        for verifier in self._verifiers:
            ok, evidence = verifier.verify(bbox, expected, field_id)
            if ok:
                return True, evidence
            failures.append(evidence)
        return False, " | ".join(failures)


__all__ = ["FieldVerifier", "CompositeVerifier", "ClipboardVerifier", "VisionVerifier", "TargetFieldVerifier", "normalize_for_compare", "_file_match"]
