"""MPF workflow helpers.

High-level hooks around the standard Atlas loop for the MPF app: record
bookkeeping, upload confirmation, and a summary of the data-entry session.
Kept deliberately thin - the heavy lifting (mapping, planning, verification,
retries) lives in the core Atlas pipeline.
"""

from __future__ import annotations

from typing import Any

from atlas.core.events import Event, EventType
from atlas.core.logging import logger

from plugins.mpf.mpf_detector import MpfDetector


class MpfWorkflow:
    """Session bookkeeping for the MPF data-entry run."""

    def __init__(self, detector: MpfDetector) -> None:
        self._detector = detector
        self._records_completed = 0
        self._records_failed = 0
        self._last_event: Event | None = None

    @property
    def completed(self) -> int:
        return self._records_completed

    @property
    def failed(self) -> int:
        return self._records_failed

    def on_event(self, event: Event) -> None:
        """Track upload progress from the event stream."""
        self._last_event = event
        if event.type == EventType.UPLOAD_COMPLETED:
            self._records_completed += 1
        elif event.type == EventType.RECORD_FAILED:
            self._records_failed += 1

    def on_record(self, record: Any) -> None:
        """Log a human-readable summary line for one finished record."""
        key = getattr(getattr(record, "record", None), "record_key", None)
        ok = bool(getattr(record, "success", False))
        index = getattr(record, "index", "?")
        actions = len(getattr(record, "actions", []))
        logger.info(
            "mpf: record {} ({}) {} - {} actions",
            index,
            key or "?",
            "uploaded" if ok else "FAILED",
            actions,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "completed": self._records_completed,
            "failed": self._records_failed,
        }


__all__ = ["MpfWorkflow"]
