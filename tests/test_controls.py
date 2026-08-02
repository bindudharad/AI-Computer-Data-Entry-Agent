"""Tests for the desktop control engine.

Covers dropdown/combobox option matching (searchable, virtual, autocomplete,
lazy lists) and date normalization, plus the engine's dispatch to the keyboard.
"""

from __future__ import annotations

from atlas.act.controls import ControlEngine
from atlas.act.mouse import HumanMouse
from atlas.config import TypingConfig
from atlas.vision.models import BBox


class StubDriver:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.pressed: list[str] = []
        self.hotkeys: list[tuple] = []

    def position(self): return (0, 0)
    def move_to(self, x, y, duration=0.0): pass
    def click(self, x, y, clicks=1, interval=0.0): pass
    def type_char(self, char): self.typed.append(char)
    def press(self, key): self.pressed.append(key)
    def hotkey(self, *keys): self.hotkeys.append(keys)
    def scroll(self, dx, dy): pass


class StubMouse:
    def __init__(self) -> None:
        self.scrolls: list[str] = []

    def move_to(self, x, y): pass
    def click(self, x, y): pass
    def scroll(self, direction, amount=3): self.scrolls.append(direction)


def _engine(driver: StubDriver | None = None, **typing_kwargs) -> ControlEngine:
    from atlas.act.keyboard import HumanKeyboard

    keyboard = HumanKeyboard(driver or StubDriver(), TypingConfig(**typing_kwargs))
    return ControlEngine(mouse=StubMouse(), keyboard=keyboard)


def test_find_option_index_exact() -> None:
    assert ControlEngine._find_option_index(["Male", "Female", "Other"], "Female") == 1
    assert ControlEngine._find_option_index(["Male", "Female", "Other"], "male") == 0


def test_find_option_index_normalized() -> None:
    assert ControlEngine._find_option_index(["Mr.", "Ms.", "Mrs."], "mr") == 0
    assert ControlEngine._find_option_index(["A & B", "C & D"], "A&B") == 0


def test_find_option_index_fuzzy() -> None:
    assert ControlEngine._find_option_index(["Single", "Married", "Divorced"], "Single") == 0
    idx = ControlEngine._find_option_index(["Single", "Married", "Divorced"], "married")
    assert idx == 1


def test_find_option_index_missing_returns_none() -> None:
    assert ControlEngine._find_option_index([], "anything") is None
    assert ControlEngine._find_option_index(["Male", "Female"], "Robot") is None


def test_select_option_arrow_when_options_match() -> None:
    driver = StubDriver()
    engine = _engine(driver)
    outcome = engine.select_option(None, "Female", ["Male", "Female"], "f0")
    assert outcome.ok is True
    assert driver.pressed == ["down", "down", "enter"]


def test_select_option_types_when_no_options() -> None:
    driver = StubDriver()
    engine = _engine(driver)
    outcome = engine.select_option(None, "Alpha Corp", None, "f1")
    assert outcome.ok is True
    assert "".join(driver.typed) == "Alpha Corp"
    assert driver.pressed[-1] == "enter"


def test_select_option_types_when_value_unmatched() -> None:
    driver = StubDriver()
    engine = _engine(driver)
    outcome = engine.select_option(None, "Startup", ["Male", "Female"], "f2")
    assert outcome.ok is True
    assert "".join(driver.typed) == "Startup"
    assert driver.pressed[-1] == "enter"


def test_normalize_date_dmy() -> None:
    assert ControlEngine._normalize_date("21/03/1996") == "21/03/1996"


def test_normalize_date_ambiguous_swaps() -> None:
    # 03/21/1996: month=21 is impossible -> day=21, month=03
    assert ControlEngine._normalize_date("03/21/1996") == "21/03/1996"


def test_normalize_date_iso() -> None:
    assert ControlEngine._normalize_date("1996-03-21") == "21/03/1996"


def test_normalize_date_word_month() -> None:
    assert ControlEngine._normalize_date("21 March 1996") == "21/03/1996"


def test_normalize_date_keep_dd_mm_yyyy() -> None:
    assert ControlEngine._normalize_date("05/08/1990") == "05/08/1990"


def test_choose_date_types_normalized_date() -> None:
    driver = StubDriver()
    engine = _engine(driver)
    outcome = engine.choose_date(None, "1996-03-21", None, "f3")
    assert outcome.ok is True
    assert "".join(driver.typed) == "21/03/1996"
    assert driver.pressed[-1] == "enter"


def test_choose_date_empty_value_skips() -> None:
    driver = StubDriver()
    engine = _engine(driver)
    outcome = engine.choose_date(None, "", None, "f4")
    assert outcome.ok is True
    assert driver.typed == []
