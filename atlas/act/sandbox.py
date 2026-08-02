"""Application sandbox for safe desktop automation.

Confinement layer that ensures the agent never interacts with applications
other than the target (MPF). Every mouse click, keyboard input, and window
focus operation is validated against the target process before execution.

Uses a finite-state machine to avoid infinite loops and bounded retries
for recovery.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import win32con
import win32gui
import win32process

from atlas.core.logging import focus_logger, logger


class SandboxState(Enum):
    """Sandbox lifecycle states."""
    ATTACHING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    REATTACHING = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class SandboxConfig:
    """Sandbox behavior flags."""
    safe_mode: bool = True
    check_focus: bool = True
    check_mouse: bool = True
    check_keyboard: bool = True
    watchdog_interval: float = 0.25
    auto_refocus: bool = True
    block_other_apps: bool = True
    max_recovery_attempts: int = 5
    recovery_interval: float = 1.0


@dataclass
class TargetInfo:
    """Immutable target application descriptor."""
    handle: int
    pid: int
    tid: int
    class_name: str
    title: str
    exe_name: str = ""
    client_rect: tuple[int, int, int, int] = (0, 0, 0, 0)


class ExecutionSandbox:
    """Validates and confines all execution to the target application.
    
    Finite-state machine with bounded recovery. Never enters an infinite loop.
    States: ATTACHING -> READY -> RUNNING <-> PAUSED -> REATTACHING -> STOPPED/FAILED
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._target: TargetInfo | None = None
        self._state = SandboxState.STOPPED
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._focus_lost_count = 0
        self._blocked_actions = 0
        self._recovery_attempts = 0
        self._last_warning: str | None = None
        self._last_refocus_time: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    def attach(self, target: TargetInfo) -> None:
        """Attach sandbox to target application.
        
        Accepts targets with pid=0 (Electron/Chrome wrappers) as long as the
        handle is valid. PID is optional for sandboxing.
        """
        with self._lock:
            if target.handle is None or target.handle == 0:
                raise ValueError(
                    f"invalid target: handle={target.handle}. "
                    "Cannot attach sandbox without a valid window handle."
                )
            self._target = target
            self._focus_lost_count = 0
            self._blocked_actions = 0
            self._recovery_attempts = 0
            self._last_warning = None
            self._state = SandboxState.READY
        self._start_watchdog()
        logger.info(
            "sandbox attached to pid={} hwnd={} class={} exe={}",
            target.pid, target.handle, target.class_name, target.exe_name,
        )

    def detach(self) -> None:
        """Stop watchdog and clear target."""
        self._state = SandboxState.STOPPED
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)
            self._watchdog = None
        with self._lock:
            self._target = None
        self._stop.clear()
        self._pause.clear()

    # -- state management ---------------------------------------------------

    @property
    def state(self) -> SandboxState:
        with self._lock:
            return self._state

    def set_running(self) -> None:
        with self._lock:
            if self._state in {SandboxState.READY, SandboxState.PAUSED}:
                self._state = SandboxState.RUNNING
                self._focus_lost_count = 0

    def set_paused(self, reason: str = "") -> None:
        with self._lock:
            if self._state == SandboxState.RUNNING:
                self._state = SandboxState.PAUSED
        if reason and reason != self._last_warning:
            logger.warning("sandbox PAUSED: {}", reason)
            focus_logger.warning("sandbox PAUSED: {}", reason)
            self._last_warning = reason

    def set_reattaching(self) -> None:
        with self._lock:
            if self._state != SandboxState.STOPPED:
                self._state = SandboxState.REATTACHING
                self._recovery_attempts = 0

    def set_failed(self) -> None:
        with self._lock:
            self._state = SandboxState.FAILED
        logger.error("Could not recover target window. Automation stopped.")

    # -- pause/resume --------------------------------------------------------

    def pause(self, reason: str = "") -> None:
        self.set_paused(reason)

    def resume(self) -> None:
        with self._lock:
            if self._state == SandboxState.PAUSED:
                self._state = SandboxState.RUNNING
                self._focus_lost_count = 0
                self._last_warning = None
                logger.info("sandbox resumed")
                focus_logger.info("sandbox resumed")

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._state == SandboxState.PAUSED

    def wait_until_resumed(self) -> None:
        """Block until sandbox is resumed (used by executor)."""
        while True:
            with self._lock:
                state = self._state
                if state != SandboxState.PAUSED:
                    return
                if state == SandboxState.STOPPED:
                    return
            time.sleep(0.1)

    # -- validation ----------------------------------------------------------

    def validate_target(self) -> TargetInfo | None:
        """Return current target, or None if not attached."""
        with self._lock:
            return self._target

    def assert_target_alive(self) -> bool:
        """Verify target window still exists and is visible."""
        target = self.validate_target()
        if target is None:
            return False
        try:
            if not win32gui.IsWindow(target.handle):
                return False
            if not win32gui.IsWindowVisible(target.handle):
                return False
            return True
        except Exception:
            return False

    def _point_in_client_rect(self, x: int, y: int, target: TargetInfo) -> bool:
        """True if (x, y) falls inside the target's absolute client rect."""
        if not target.client_rect:
            return False
        left, top, right, bottom = target.client_rect
        if right <= left or bottom <= top:
            return False
        return left <= x <= right and top <= y <= bottom

    def validate_click(self, x: int, y: int) -> tuple[bool, str]:
        """Validate that a click at (x,y) is inside the target application.

        (x, y) are absolute screen coordinates. For Electron/Chrome apps
        (pid=0) the window hierarchy cannot be trusted, so clicks are confined
        to the target's client rect. For other apps the window hierarchy is
        checked first, with the client rect as a final bound. Clicks outside
        the target are always rejected.
        """
        if not self._config.check_mouse:
            return True, "mouse check disabled"
        target = self.validate_target()
        if target is None:
            return False, "no target attached"

        if target.pid == 0:
            # Electron/Chrome wrapper: window hierarchy is unreliable, so bound
            # clicks to the recorded client rect (absolute screen coordinates).
            if target.client_rect and self._point_in_client_rect(x, y, target):
                return True, "ok (inside client rect)"
            return False, "click outside target client rect"

        try:
            hwnd = win32gui.WindowFromPoint((x, y))
            if hwnd:
                clicked_root = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
                if clicked_root == target.handle:
                    return True, "ok (same root window)"

                parent = win32gui.GetAncestor(hwnd, win32con.GA_PARENT)
                if parent == target.handle:
                    return True, "ok (child of target)"

                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == target.pid and pid > 0:
                    return True, "ok (pid match)"

                # A window resolved at this point that is not part of the target
                # hierarchy (e.g. a foreign overlay/popup) would steal the click.
                return False, "click on foreign window"
        except Exception:
            pass

        # No window resolved at the point: fall back to the client-rect bound
        # so clicks never escape the recorded target area.
        if target.client_rect and self._point_in_client_rect(x, y, target):
            return True, "ok (inside client rect)"
        return False, "click outside target window"

    def validate_keyboard(self) -> tuple[bool, str]:
        """Validate that keyboard input will go to the target application."""
        if not self._config.check_keyboard:
            return True, "keyboard check disabled"
        target = self.validate_target()
        if target is None:
            return True, "no desktop target attached - allowing"
        
        try:
            fg = win32gui.GetForegroundWindow()
            if not self._window_belongs_to_target(fg, target):
                if self._config.auto_refocus:
                    self._refocus_target(target)
                    time.sleep(0.2)
                    fg = win32gui.GetForegroundWindow()
                    if not self._window_belongs_to_target(fg, target):
                        return False, f"foreground != target"
                else:
                    return False, f"focus lost"
        except Exception as exc:
            logger.debug("validate_keyboard error: {}", exc)
        
        return True, "ok"

    # -- internals -----------------------------------------------------------

    def _start_watchdog(self) -> None:
        self._stop.clear()
        self._pause.clear()
        with self._lock:
            self._state = SandboxState.READY
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._watchdog_tick()
            except Exception as exc:
                logger.debug("watchdog error: {}", exc)
            
            # Sleep based on current state
            with self._lock:
                state = self._state
            if state == SandboxState.PAUSED:
                time.sleep(1.0)  # slow polling while paused
            elif state == SandboxState.REATTACHING:
                time.sleep(self._config.recovery_interval)
            elif state == SandboxState.RUNNING:
                time.sleep(self._config.watchdog_interval)
            else:
                time.sleep(0.5)

    def _watchdog_tick(self) -> None:
        with self._lock:
            state = self._state
            target = self._target
        
        if state in {SandboxState.STOPPED, SandboxState.FAILED}:
            return
        
        if target is None:
            return
        
        # Check target alive
        if not self.assert_target_alive():
            with self._lock:
                if self._state == SandboxState.REATTACHING:
                    self._recovery_attempts += 1
                    if self._recovery_attempts >= self._config.max_recovery_attempts:
                        logger.error("Could not recover target window. Automation stopped.")
                        self.set_failed()
                        return
                    logger.warning(
                        "Target window lost. Recovery attempt {}/{}",
                        self._recovery_attempts,
                        self._config.max_recovery_attempts,
                    )
                else:
                    logger.warning("Target window lost. Attempting recovery...")
                    self._recovery_attempts = 1
                    self._state = SandboxState.REATTACHING
            return
        
        # Target is alive - reset recovery if we were reattaching
        with self._lock:
            if self._state == SandboxState.REATTACHING:
                logger.info("Recovery successful.")
                self._recovery_attempts = 0
                self._last_warning = None
        
        # Focus check
        if not self._config.check_focus:
            with self._lock:
                if self._state == SandboxState.READY:
                    self._state = SandboxState.RUNNING
            return

        try:
            fg = win32gui.GetForegroundWindow()
            with self._lock:
                current_state = self._state

            # Check if foreground window belongs to our target hierarchy.
            # For Electron apps the foreground HWND may differ from the wrapper.
            has_focus = self._window_belongs_to_target(fg, target)
            focus_logger.debug(
                "watchdog tick: fg={} target={} has_focus={} state={} lost={}",
                fg, target.handle, has_focus, current_state.value, self.focus_lost_count,
            )
            
            if not has_focus:
                with self._lock:
                    self._focus_lost_count += 1
                
                if current_state == SandboxState.REATTACHING:
                    return
                
                # Only refocus after 500ms of focus loss AND no recent refocus (5s cooldown)
                now = time.time()
                recent_refocus = (now - self._last_refocus_time) < 5.0
                if self._config.auto_refocus and self._focus_lost_count >= 3 and not recent_refocus:
                    self._last_refocus_time = now
                    self._refocus_target(target)
                    time.sleep(0.2)
                    fg = win32gui.GetForegroundWindow()
                    if self._window_belongs_to_target(fg, target):
                        with self._lock:
                            self._focus_lost_count = 0
                            self._last_warning = None
                    else:
                        self.pause("Focus lost. Waiting for MPF.")
                elif self._focus_lost_count >= 10:
                    self.pause("Focus lost. Waiting for MPF.")
            else:
                with self._lock:
                    if self._focus_lost_count > 0:
                        pass  # Focus restored, but don't log every time
                    self._focus_lost_count = 0
                    self._last_warning = None
                    if current_state == SandboxState.PAUSED:
                        self._state = SandboxState.RUNNING
        except Exception:
            pass

    @staticmethod
    def _refocus_target(target: TargetInfo) -> None:
        try:
            if win32gui.IsIconic(target.handle):
                win32gui.ShowWindow(target.handle, 9)
            win32gui.SetForegroundWindow(target.handle)
        except Exception as exc:
            logger.debug("refocus failed: {}", exc)

    @staticmethod
    def _window_belongs_to_target(hwnd: int, target: TargetInfo) -> bool:
        """Check whether a window belongs to the target application hierarchy."""
        if hwnd is None or hwnd == 0:
            return False
        if hwnd == target.handle:
            return True
        try:
            root = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
            if root == target.handle:
                return True
            # Electron apps: foreground may be a child window of the wrapper
            owner = win32gui.GetAncestor(hwnd, win32con.GA_ROOTOWNER)
            if owner == target.handle:
                return True
        except Exception:
            pass
        return False

    # -- stats ---------------------------------------------------------------

    @property
    def focus_lost_count(self) -> int:
        with self._lock:
            return self._focus_lost_count

    @property
    def blocked_actions(self) -> int:
        return self._blocked_actions


__all__ = ["ExecutionSandbox", "SandboxConfig", "TargetInfo", "SandboxState"]
