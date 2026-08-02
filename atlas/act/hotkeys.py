"""Global hotkey manager for emergency control.

Registers system-wide hotkeys:

  ESC           - immediate stop
  Ctrl+Shift+S  - pause
  Ctrl+Shift+R  - resume
  Ctrl+Shift+Q  - quit safely

Uses a low-level keyboard hook so the hotkeys work from anywhere, even when
the target application does not have focus.
"""

from __future__ import annotations

import threading
from typing import Callable

from atlas.core.logging import logger

try:
    import win32api
    import win32con
    import win32gui
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


class HotkeyManager:
    """Registers global hotkeys and dispatches callbacks."""

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._registered: list[int] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False

    def register(self, name: str, callback: Callable[[], None]) -> None:
        """Register a named hotkey callback."""
        self._callbacks[name] = callback

    def start(self) -> None:
        """Start the hotkey listener thread."""
        if not _HAS_WIN32 or self._running:
            return
        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("hotkey manager started (ESC=stop, Ctrl+Shift+S=pause, Ctrl+Shift+R=resume, Ctrl+Shift+Q=quit)")

    def stop(self) -> None:
        """Stop the hotkey listener thread."""
        self._stop.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._unregister_all()

    def _unregister_all(self) -> None:
        for hotkey_id in self._registered:
            try:
                win32gui.UnregisterHotKey(None, hotkey_id)
            except Exception:
                pass
        self._registered.clear()

    def _run(self) -> None:
        """Message loop that dispatches registered hotkeys."""
        # Register hotkeys
        # VK codes: ESC=0x1B, S=0x53, R=0x52, Q=0x51
        hotkey_map = {
            1: ("stop", 0x1B, 0),
            2: ("pause", 0x53, win32con.MOD_CONTROL | win32con.MOD_SHIFT),
            3: ("resume", 0x52, win32con.MOD_CONTROL | win32con.MOD_SHIFT),
            4: ("quit", 0x51, win32con.MOD_CONTROL | win32con.MOD_SHIFT),
        }
        for hotkey_id, (name, vk, modifiers) in hotkey_map.items():
            try:
                win32gui.RegisterHotKey(None, hotkey_id, modifiers, vk)
                self._registered.append(hotkey_id)
            except Exception as exc:
                logger.debug("failed to register hotkey {}: {}", name, exc)

        import time
        try:
            while not self._stop.is_set():
                try:
                    # Use PeekMessage so stop() can interrupt the loop.
                    # Returns (bGotMsg, hwnd, message, wParam, lParam, time, pt).
                    msg = win32gui.PeekMessage(None, 0, 0, win32con.PM_REMOVE)
                    if msg and msg[0] and msg[2] == win32con.WM_HOTKEY:
                        hotkey_id = msg[3]
                        name = hotkey_map.get(hotkey_id, (None,))[0]
                        if name and name in self._callbacks:
                            try:
                                self._callbacks[name]()
                            except Exception as exc:
                                logger.debug("hotkey {} callback failed: {}", name, exc)
                    else:
                        time.sleep(0.05)
                except Exception:
                    break
        finally:
            self._unregister_all()


__all__ = ["HotkeyManager"]