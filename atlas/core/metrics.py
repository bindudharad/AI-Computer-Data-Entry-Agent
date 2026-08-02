"""Performance metrics and timers.

Lightweight timing utilities for logging every stage's duration and producing
per-record performance summaries.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class Timer:
    """Stopwatch context manager / one-shot timer."""

    def __init__(self) -> None:
        self._started: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._started is not None:
            self.elapsed = time.perf_counter() - self._started
            self._started = None
        return self.elapsed

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


@contextmanager
def timed(label: str, logger=None) -> Iterator[Timer]:
    """Context manager that logs the duration of a block."""
    timer = Timer().start()
    try:
        yield timer
    finally:
        timer.stop()
        if logger is not None:
            logger.debug("timed[{}] {:.3f}s", label, timer.elapsed)
        from atlas.core.logging import timing_logger

        timing_logger.debug("timed[{}] {:.3f}s", label, timer.elapsed)


@dataclass
class StageMetrics:
    """Timings per pipeline stage for one record."""

    stage_times: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.stage_times[stage] = self.stage_times.get(stage, 0.0) + seconds

    def total(self) -> float:
        return sum(self.stage_times.values())

    def to_dict(self) -> dict[str, float]:
        return dict(self.stage_times)
