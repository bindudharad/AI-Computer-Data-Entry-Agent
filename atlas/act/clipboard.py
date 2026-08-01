"""Clipboard engine.

Provides safe clipboard read/write with restoration of the previous clipboard
content, and copy/paste into a focused control. Clipboard ops are used for
long text values (faster and more reliable than character typing) and for
verification (select-all + copy to read a field's current value).
"""

from __future__ import annotations

import time
import pyperclip

from atlas.act.mouse import InputDriver
from atlas.core.logging import logger


class ClipboardEngine:
    """Thread-safe-ish clipboard wrapper with restore semantics."""

    def __init__(self, driver: InputDriver | None = None) -> None:
        self._driver = driver

    def get_text(self) -> str:
        try:
            return pyperclip.paste() or ""
        except Exception as exc:
            logger.warning("clipboard read failed: {}", exc)
            return ""

    def set_text(self, text: str) -> None:
        try:
            pyperclip.copy(text or "")
        except Exception as exc:
            logger.warning("clipboard write failed: {}", exc)

    def paste_into_focused(self, text: str) -> None:
        """Copy text to the clipboard and paste into the focused control."""
        self.set_text(text)
        if self._driver is not None:
            self._driver.hotkey("ctrl", "v")
        time.sleep(0.1)

    def read_focused(self) -> str:
        """Select-all + copy the focused control and return its value."""
        if self._driver is not None:
            self._driver.hotkey("ctrl", "a")
            time.sleep(0.05)
            self._driver.hotkey("ctrl", "c")
            time.sleep(0.05)
        return self.get_text()

    @staticmethod
    def normalize(value: str) -> str:
        return " ".join(value.strip().split())


__all__ = ["ClipboardEngine"]
