"""Continuous observation loop.

Polls the attached window's client area and produces observations. The loop is
driven by the workflow controller; a callback is invoked on every observation
so the agent can re-analyse, verify previous actions, or detect the next record.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from atlas.core.logging import logger
from atlas.vision.scene import SceneAnalysis, WindowSceneSource

ObservationCallback = Callable[[SceneAnalysis], bool | None]


@dataclass
class Observation:
    """One observation tick result."""

    analysis: SceneAnalysis | None
    timestamp: float
    changed: bool = False

    @property
    def ok(self) -> bool:
        return self.analysis is not None


class Observer:
    """Background thread that polls the window scene source.

    Pausing is cooperative: when paused the thread sleeps and does not invoke
    the callback. Stop is a one-way transition.
    """

    def __init__(
        self,
        source: WindowSceneSource,
        callback: ObservationCallback,
        poll_interval: float = 0.8,
    ) -> None:
        self._source = source
        self._callback = callback
        self._interval = poll_interval
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_analysis: SceneAnalysis | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="atlas-observer", daemon=True)
        self._thread.start()
        logger.debug("observer started")

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.1)
                continue
            try:
                analysis = self._source.observe()
            except Exception as exc:
                logger.warning("observe failed: {}", exc)
                analysis = None
            self.last_analysis = analysis
            try:
                should_stop = self._callback(analysis) if analysis is not None else False
            except Exception:
                logger.exception("observation callback failed")
                should_stop = False
            if should_stop:
                break
            time.sleep(self._interval)
        logger.debug("observer stopped")

    def _changed(self, analysis: SceneAnalysis | None) -> bool:
        if analysis is None or self.last_analysis is None:
            return analysis is not None
        return (
            self.last_analysis.scene.layout_summary != analysis.scene.layout_summary
            or len(self.last_analysis.scene.elements) != len(analysis.scene.elements)
        )


__all__ = ["Observer", "Observation"]
