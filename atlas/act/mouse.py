"""Human-like mouse control.

Implements Bezier movement, acceleration, small random jitter, hover-before-
click and occasional hesitation so that automation is indistinguishable from a
human operator. All low-level calls go through an injected :class:`InputDriver`
so the behaviour is unit-testable without a live desktop.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod

from atlas.config import MouseConfig
from atlas.core.logging import logger


class InputDriver(ABC):
    """Low-level input backend (mouse + keyboard)."""

    @abstractmethod
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None: ...

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1, interval: float = 0.2) -> None: ...

    @abstractmethod
    def right_click(self, x: int, y: int) -> None: ...

    @abstractmethod
    def position(self) -> tuple[int, int]: ...

    @abstractmethod
    def scroll(self, dx: int, dy: int) -> None: ...

    @abstractmethod
    def type_char(self, char: str) -> None: ...

    @abstractmethod
    def press(self, key: str) -> None: ...

    @abstractmethod
    def hotkey(self, *keys: str) -> None: ...

    @abstractmethod
    def release_all(self) -> None:
        """Release any held mouse buttons / keyboard modifiers.

        Called on emergency stop so no physical key or button stays pressed
        after the agent is interrupted mid-gesture.
        """


class PyAutoGuiDriver(InputDriver):
    """pyautogui-based input driver."""

    def __init__(self) -> None:
        import pyautogui

        # Fail-safe triggers when the mouse reaches a screen corner, which can
        # happen during legitimate automation (e.g. scrolling to the bottom of
        # a form). The ExecutionSandbox already confines all input to the
        # target window, so the fail-safe is unnecessary and harmful here.
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self._pg = pyautogui

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        self._pg.moveTo(x, y, duration=duration)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1, interval: float = 0.2) -> None:
        self._pg.click(x, y, button=button, clicks=clicks, interval=interval)

    def right_click(self, x: int, y: int) -> None:
        self._pg.rightClick(x, y)

    def position(self) -> tuple[int, int]:
        return self._pg.position()

    def scroll(self, dx: int, dy: int) -> None:
        self._pg.scroll(dy)

    def type_char(self, char: str) -> None:
        self._pg.typewrite(char)

    def press(self, key: str) -> None:
        self._pg.press(key)

    def hotkey(self, *keys: str) -> None:
        self._pg.hotkey(*keys)

    def release_all(self) -> None:
        """Release held buttons/modifiers via pyautogui (best-effort)."""
        try:
            self._pg.mouseUp()
        except Exception:
            pass
        for key in ("ctrl", "alt", "shift"):
            try:
                self._pg.keyUp(key)
            except Exception:
                pass


class HumanMouse:
    """Human-like mouse behaviour on top of an :class:`InputDriver`."""

    def __init__(self, driver: InputDriver, config: MouseConfig | None = None) -> None:
        self._driver = driver
        self._cfg = config or MouseConfig()

    @property
    def driver(self) -> InputDriver:
        return self._driver

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to (x, y) along a human-like Bezier path."""
        start = self._driver.position()
        points = self._bezier_path(start, (x, y), steps=self._cfg.bezier_steps)
        total_duration = self._cfg.speed
        for point in points:
            px, py = self._jitter(point)
            try:
                self._driver.move_to(px, py, duration=0.0)
            except Exception as exc:
                # Never log-spam for skipped intermediate points.
                logger.debug("move_to skipped intermediate point: {}", exc)
            time.sleep(random.uniform(self._cfg.min_delay, self._cfg.max_delay) * 0.15)
        # settle on the exact target - always deliver the final position.
        try:
            self._driver.move_to(x, y, duration=total_duration * 0.3)
        except Exception as exc:
            logger.debug("move_to final settle failed: {}", exc)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.max_delay) * 0.4)

    def click(self, x: int, y: int) -> None:
        """Hover, hesitate, then click."""
        self.move_to(x, y)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_before_click))
        if random.random() < 0.15:  # occasional hesitation
            time.sleep(random.uniform(0.05, 0.2))
        self._driver.click(x, y)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_after_click))

    def double_click(self, x: int, y: int) -> None:
        self.move_to(x, y)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_before_click))
        self._driver.click(x, y, clicks=2, interval=self._cfg.double_click_interval)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_after_click))

    def right_click(self, x: int, y: int) -> None:
        self.move_to(x, y)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_before_click))
        self._driver.right_click(x, y)
        time.sleep(random.uniform(self._cfg.min_delay, self._cfg.pause_after_click))

    def hover(self, x: int, y: int) -> None:
        self.move_to(x, y)

    def release(self) -> None:
        """Release any held buttons on emergency stop."""
        try:
            self._driver.release_all()
        except Exception as exc:
            logger.debug("mouse release failed: {}", exc)

    def scroll(self, direction: str, amount: int = 3) -> None:
        if direction not in {"up", "down"}:
            logger.warning("unknown scroll direction {}", direction)
            return
        sign = 1 if direction == "down" else -1
        for _ in range(max(1, abs(amount))):
            self._driver.scroll(0, sign)
            time.sleep(random.uniform(0.05, 0.12))

    # -- internal helpers ----------------------------------------------------

    def _jitter(self, point: tuple[int, int]) -> tuple[int, int]:
        r = self._cfg.jitter_px
        return (point[0] + random.randint(-r, r), point[1] + random.randint(-r, r))

    def _bezier_path(
        self, start: tuple[int, int], end: tuple[int, int], steps: int
    ) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dist = math.hypot(x1 - x0, y1 - y0)
        offset = max(20.0, dist * random.uniform(0.1, 0.35))
        angle = random.uniform(0, math.pi)
        # control points perpendicular to the travel direction
        px = (y1 - y0) / (dist + 1e-9)
        py = (x1 - x0) / (dist + 1e-9)
        cx1 = x0 + (x1 - x0) * 0.33 + px * offset * math.cos(angle)
        cy1 = y0 + (y1 - y0) * 0.33 + py * offset * math.sin(angle)
        cx2 = x0 + (x1 - x0) * 0.66 + px * offset * math.cos(angle + 0.7)
        cy2 = y0 + (y1 - y0) * 0.66 + py * offset * math.sin(angle + 0.7)
        points: list[tuple[int, int]] = []
        for i in range(steps):
            t = i / steps
            t = self._ease_in_out(t)
            inv = 1.0 - t
            x = inv * inv * inv * x0 + 3 * inv * inv * t * cx1 + 3 * inv * t * t * cx2 + t * t * t * x1
            y = inv * inv * inv * y0 + 3 * inv * inv * t * cy1 + 3 * inv * t * t * cy2 + t * t * t * y1
            points.append((int(x), int(y)))
        points.append((x1, y1))
        return points

    @staticmethod
    def _ease_in_out(t: float) -> float:
        return t * t * (3.0 - 2.0 * t) if t <= 0.5 else 1.0 - (1.0 - t) ** 2 * (3.0 - 2.0 * (1.0 - t))


__all__ = ["HumanMouse", "InputDriver", "PyAutoGuiDriver"]
