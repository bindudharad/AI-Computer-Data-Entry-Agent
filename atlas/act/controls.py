"""Field control actions.

Defines the ``ControlInterface`` implemented by both the desktop control engine
(mouse/keyboard/clipboard) and the web DOM control engine (Playwright). The
action executor depends only on the interface, so swapping a target never
changes the executor.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from atlas.act.keyboard import HumanKeyboard
from atlas.act.mouse import HumanMouse
from atlas.config import TypingConfig
from atlas.vision.models import BBox


@dataclass
class ControlOutcome:
    """Result of a control operation."""

    ok: bool
    evidence: str = ""


class ControlInterface(ABC):
    """Operations the executor can perform on a single field."""

    @abstractmethod
    def focus(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def click_field(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def type_value(self, bbox: BBox | None, value: str, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def clear(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def select_option(
        self, bbox: BBox | None, value: str, options: list[str] | None = None, field_id: str | None = None
    ) -> ControlOutcome: ...

    @abstractmethod
    def toggle(self, bbox: BBox | None, value: str, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def choose_date(
        self, bbox: BBox | None, value: str, date_format: str | None = None, field_id: str | None = None
    ) -> ControlOutcome: ...

    @abstractmethod
    def press_tab(self) -> ControlOutcome: ...

    @abstractmethod
    def press_enter(self) -> ControlOutcome: ...

    @abstractmethod
    def press_escape(self) -> ControlOutcome: ...

    @abstractmethod
    def scroll(self, direction: str, amount: int = 3) -> ControlOutcome: ...

    def scroll_by_keys(self, direction: str, amount: int = 3) -> ControlOutcome:
        """Scroll via keyboard (PageUp/PageDown/Home/End).

        Default implementation reports it is unsupported so the executor can
        fall through to the scroll-bar strategy. Engines that can scroll a
        focused container by keys override this.
        """
        return ControlOutcome(ok=False, evidence="keyboard scroll not supported")

    def scroll_bar(self, direction: str, amount: int = 3) -> ControlOutcome:
        """Scroll by dragging / clicking the window's scroll bar.

        Default implementation reports it is unsupported so the executor can
        give up cleanly. Engines that can reach a scroll bar override this.
        """
        return ControlOutcome(ok=False, evidence="scroll-bar scroll not supported")

    @abstractmethod
    def paste(self, value: str, field_id: str | None = None) -> ControlOutcome: ...

    @abstractmethod
    def upload_file(self, bbox: BBox | None, path: str, field_id: str | None = None) -> ControlOutcome: ...


class ControlEngine(ControlInterface):
    """Desktop control engine: mouse, keyboard and clipboard."""

    def __init__(
        self,
        mouse: HumanMouse,
        keyboard: HumanKeyboard,
        typing_config: TypingConfig | None = None,
        clipboard_use_long: bool = True,
        clipboard_min_length: int = 25,
    ) -> None:
        self._mouse = mouse
        self._keyboard = keyboard
        self._typing = typing_config or TypingConfig()
        self._clipboard_long = clipboard_use_long
        self._clipboard_min = clipboard_min_length

    def focus(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome:
        if bbox is None:
            return ControlOutcome(ok=True, evidence="focus skipped (no bbox)")
        x, y = bbox.center
        self._mouse.click(x, y)
        time.sleep(0.15)
        return ControlOutcome(ok=True, evidence=f"focused ({x},{y})")

    def click_field(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome:
        if bbox is None:
            return ControlOutcome(ok=False, evidence="no bbox for click")
        x, y = bbox.center
        self._mouse.click(x, y)
        time.sleep(0.1)
        return ControlOutcome(ok=True, evidence=f"clicked ({x},{y})")

    def type_value(self, bbox: BBox | None, value: str, field_id: str | None = None) -> ControlOutcome:
        self._keyboard.clear_field()
        time.sleep(0.1)
        if self._clipboard_long and len(value) >= self._clipboard_min:
            self._paste_value(value)
            return ControlOutcome(ok=True, evidence=f"pasted {len(value)} chars")
        self._keyboard.type_text(value)
        return ControlOutcome(ok=True, evidence=f"typed {len(value)} chars")

    def clear(self, bbox: BBox | None, field_id: str | None = None) -> ControlOutcome:
        self._keyboard.clear_field()
        return ControlOutcome(ok=True, evidence="cleared")

    def select_option(
        self, bbox: BBox | None, value: str, options: list[str] | None = None, field_id: str | None = None
    ) -> ControlOutcome:
        value_str = str(value or "").strip()
        if options:
            idx = self._find_option_index(options, value_str)
            if idx is not None:
                # Let the dropdown animate open before arrow-navigating.
                time.sleep(self._typing.dropdown_wait)
                self._keyboard.press("down", idx + 1)
                self._keyboard.enter()
                return ControlOutcome(ok=True, evidence=f"arrow-selected #{idx}")
        self._keyboard.type_text(value_str)
        time.sleep(self._typing.dropdown_wait)
        self._keyboard.enter()
        return ControlOutcome(ok=True, evidence=f"typed option {value_str!r}")

    def toggle(self, bbox: BBox | None, value: str, field_id: str | None = None) -> ControlOutcome:
        if bbox is None:
            return ControlOutcome(ok=True, evidence="toggle skipped (no bbox)")
        x, y = bbox.center
        self._mouse.click(x, y)
        return ControlOutcome(ok=True, evidence=f"toggled {value!r} at ({x},{y})")

    def choose_date(
        self, bbox: BBox | None, value: str, date_format: str | None = None, field_id: str | None = None
    ) -> ControlOutcome:
        date_str = self._normalize_date(str(value or ""), date_format)
        self._keyboard.clear_field()
        time.sleep(0.1)
        self._keyboard.type_text(date_str)
        time.sleep(self._typing.dropdown_wait)
        self._keyboard.enter()
        return ControlOutcome(ok=True, evidence=f"typed date {date_str!r}")

    def press_tab(self) -> ControlOutcome:
        self._keyboard.tab()
        return ControlOutcome(ok=True, evidence="tab")

    def press_enter(self) -> ControlOutcome:
        self._keyboard.enter()
        return ControlOutcome(ok=True, evidence="enter")

    def press_escape(self) -> ControlOutcome:
        self._keyboard.escape()
        return ControlOutcome(ok=True, evidence="escape")

    def scroll(self, direction: str, amount: int = 3) -> ControlOutcome:
        self._mouse.scroll(direction, amount)
        return ControlOutcome(ok=True, evidence=f"scrolled {direction} {amount}")

    def scroll_by_keys(self, direction: str, amount: int = 3) -> ControlOutcome:
        """Scroll a focused container with PageUp/PageDown keys.

        Useful when the mouse wheel is over a region that does not capture the
        wheel (nested scroll panes, web iframes). Pressing PageUp/PageDown
        scrolls whatever control currently has focus.
        """
        key = "pagedown" if direction == "down" else "pageup"
        presses = max(1, abs(amount))
        self._keyboard.press(key, presses)
        return ControlOutcome(ok=True, evidence=f"key-scrolled {direction} {presses}")

    def scroll_bar(self, direction: str, amount: int = 3) -> ControlOutcome:
        """Scroll by pressing End/Home keys toward the desired edge.

        A true scroll-bar drag requires knowing the bar's geometry; End/Home
        jump to the end of the active scroll region, which covers the same
        goal for the common long-form case.
        """
        key = "end" if direction == "down" else "home"
        self._keyboard.press(key)
        return ControlOutcome(ok=True, evidence=f"scroll-bar jump {direction} ({key})")

    def paste(self, value: str, field_id: str | None = None) -> ControlOutcome:
        self._paste_value(value)
        return ControlOutcome(ok=True, evidence=f"pasted {len(value)} chars")

    def upload_file(self, bbox: BBox | None, path: str, field_id: str | None = None) -> ControlOutcome:
        """Fill a file-upload control.

        Desktop file inputs (and their ``<input type=file>`` equivalents in
        Chromium/Electron) accept a file path once focused. Click the control
        then type the absolute path and confirm. Never raises.
        """
        if not path:
            return ControlOutcome(ok=False, evidence="no file path for upload")
        if bbox is not None:
            x, y = bbox.center
            self._mouse.click(x, y)
            time.sleep(self._typing.dropdown_wait)
        self._keyboard.clear_field()
        time.sleep(0.1)
        self._keyboard.type_text(path)
        time.sleep(self._typing.dropdown_wait)
        self._keyboard.enter()
        return ControlOutcome(ok=True, evidence=f"uploaded file {path!r}")

    # -- internal helpers ----------------------------------------------------

    def _paste_value(self, value: str) -> None:
        from atlas.act.clipboard import ClipboardEngine

        ClipboardEngine(driver=self._keyboard.driver).paste_into_focused(value)
        time.sleep(0.15)

    @staticmethod
    def _find_option_index(options: list[str], value: str) -> int | None:
        target = value.strip().lower()
        normalized = re.sub(r"[^a-z0-9]", "", target)
        for i, option in enumerate(options):
            if option.strip().lower() == target:
                return i
        for i, option in enumerate(options):
            if re.sub(r"[^a-z0-9]", "", option.lower()) == normalized:
                return i
        best_i: int | None = None
        best_score: float = 0.0
        for i, option in enumerate(options):
            o = option.lower()
            if normalized and (normalized in o or o in normalized):
                score = min(len(normalized), len(o)) / max(len(normalized), len(o), 1)
                if score > best_score:
                    best_score, best_i = score, i
        return best_i

    @staticmethod
    def _normalize_date(value: str, date_format: str | None = None) -> str:
        value = value.strip()
        if not value:
            return value
        month_names = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12,
        }
        m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", value)
        if m:
            a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
            if b > 12 and a <= 12:
                # e.g. 03/21/1996 (MM/DD) -> day=21, month=03
                day, month = b, a
            elif a > 12 and b <= 12:
                # e.g. 21/03/1996 (DD/MM) -> day=21, month=03
                day, month = a, b
            else:
                day, month = a, b
            return f"{day:02d}/{month:02d}/{year}"
        m = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", value)
        if m:
            y_str, mo, d = m.groups()
            return f"{int(d):02d}/{int(mo):02d}/{y_str}"
        m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", value)
        if m:
            d_str, month_name, y_str = m.groups()
            month_num = month_names.get(month_name.lower())
            if month_num:
                return f"{int(d_str):02d}/{month_num:02d}/{y_str}"
        if "/" in value and value.count("/") == 2:
            parts = value.split("/")
            if all(p.isdigit() for p in parts):
                return value
        return value


__all__ = ["ControlInterface", "ControlEngine", "ControlOutcome"]
