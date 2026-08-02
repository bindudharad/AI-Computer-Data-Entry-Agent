"""Human-like keyboard control.

Variable-speed typing with human pause patterns, TAB navigation, Ctrl+A,
clipboard paste and backspace correction. Optionally simulates typos that are
immediately corrected (configurable, off by default for data entry safety).
"""

from __future__ import annotations

import random
import time

from atlas.act.mouse import InputDriver
from atlas.config import TypingConfig

#: pyautogui key names for common keys.
KEY_NAMES = {
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "space": "space",
    "escape": "esc",
    "esc": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "pageup": "pageup",
    "pagedown": "pagedown",
}


class HumanKeyboard:
    """Human-like typing on top of an :class:`InputDriver`."""

    def __init__(self, driver: InputDriver, config: TypingConfig | None = None) -> None:
        self._driver = driver
        self._cfg = config or TypingConfig()

    @property
    def driver(self) -> InputDriver:
        return self._driver

    def type_text(self, text: str) -> None:
        """Type text with variable per-character delays and human pauses."""
        for char in text:
            delay = random.uniform(self._cfg.min_delay, self._cfg.max_delay)
            self._press_char(char)
            # occasional word-boundary pause
            if char == " " and random.random() < 0.3:
                delay += random.uniform(0.05, 0.2)
            time.sleep(delay)
        time.sleep(self._cfg.pause_after)

    def press(self, key: str, presses: int = 1) -> None:
        name = KEY_NAMES.get(str(key).lower(), str(key))
        for _ in range(presses):
            self._driver.press(name)
            time.sleep(random.uniform(0.03, 0.09))

    def tab(self, times: int = 1) -> None:
        self.press("tab", times)

    def enter(self) -> None:
        self.press("enter")

    def escape(self) -> None:
        self.press("escape")

    def backspace(self, count: int = 1) -> None:
        self.press("backspace", count)

    def select_all(self) -> None:
        self._driver.hotkey("ctrl", "a")

    def copy(self) -> None:
        self._driver.hotkey("ctrl", "c")

    def paste(self) -> None:
        self._driver.hotkey("ctrl", "v")

    def undo(self) -> None:
        self._driver.hotkey("ctrl", "z")

    def release(self) -> None:
        """Release any held modifiers on emergency stop."""
        try:
            self._driver.release_all()
        except Exception as exc:
            from atlas.core.logging import logger

            logger.debug("keyboard release failed: {}", exc)

    def clear_field(self) -> None:
        """Select all and delete the current field content."""
        self.select_all()
        time.sleep(random.uniform(0.03, 0.08))
        self.backspace()
        time.sleep(random.uniform(0.03, 0.08))

    def type_with_correction(self, text: str, typo_rate: float = 0.0) -> None:
        """Type text, optionally simulating a typo that is corrected.

        Typo simulation is disabled by default (``TYPING_SIMULATE_TYPOS=false``)
        because this agent performs data entry where correctness matters.
        """
        if typo_rate <= 0 or len(text) < 6:
            self.type_text(text)
            return
        if random.random() < typo_rate:
            pos = random.randint(1, len(text) - 2)
            wrong = random.choice("abcdefghijklmnopqrstuvwxyz")
            self.type_text(text[:pos] + wrong)
            time.sleep(random.uniform(0.1, 0.25))
            self.backspace()
            time.sleep(random.uniform(0.05, 0.15))
            self.type_text(text[pos:])
        else:
            self.type_text(text)

    def _press_char(self, char: str) -> None:
        self._driver.type_char(char)


__all__ = ["HumanKeyboard", "KEY_NAMES"]
