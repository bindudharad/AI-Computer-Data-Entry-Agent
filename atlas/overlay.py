"""Floating status overlay (optional).

A tiny always-on-top tkinter window that mirrors the agent's current state so
the operator can see progress without watching logs. Purely cosmetic: the agent
runs identically with or without it. Disabled by default for automation work
that must not cover the target window.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from atlas.config import OverlayConfig
from atlas.core.logging import logger

StateProvider = Callable[[], str]


class Overlay:
    """Always-on-top status label driven by a state provider."""

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self._config = config or OverlayConfig()
        self._provider: StateProvider = lambda: "idle"
        self._root: Any = None
        self._label: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def start(self, provider: StateProvider) -> None:
        if not self._config.enabled:
            return
        try:
            import tkinter as tk
        except ImportError:
            logger.warning("tkinter unavailable - overlay disabled")
            return
        self._provider = provider
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(tk,),
            name="atlas-overlay",
            daemon=True,
        )
        self._thread.start()

    def _run(self, tk: Any) -> None:
        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)
            self._root.attributes("-topmost", True)
            self._root.configure(bg="#101418")
            self._label = tk.Label(
                self._root,
                text="ATLAS AI",
                fg="#7dd3fc",
                bg="#101418",
                font=("Consolas", 11, "bold"),
                padx=10,
                pady=4,
            )
            self._label.pack()
            self._poll()
            self._root.mainloop()
        except Exception as exc:
            logger.debug("overlay stopped: {}", exc)

    def _poll(self) -> None:
        if self._stop.is_set():
            try:
                self._root.after(50, self._root.destroy)
            except Exception:
                pass
            return
        try:
            self._label.configure(text=f"ATLAS AI  [{self._provider()}]")
        except Exception:
            pass
        try:
            self._root.after(200, self._poll)
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


__all__ = ["Overlay"]
