"""Tests for the ExecutionSandbox click/keyboard confinement.

The real win32 API needs a desktop session, so win32gui/win32process are
mocked to simulate a target window hierarchy and foreground state.
"""

from __future__ import annotations

import pytest

from atlas.act.sandbox import ExecutionSandbox, SandboxConfig, SandboxState, TargetInfo

TARGET_HWND = 1000
TARGET_PID = 42
TARGET_RECT = (100, 200, 500, 600)  # left, top, right, bottom (absolute screen)


class _FakeWin32:
    def __init__(self) -> None:
        self.foreground = TARGET_HWND
        self._window_roots: dict[int, int] = {}
        self._window_parents: dict[int, int] = {}
        self._window_pids: dict[int, int] = {}
        self._window_from_point: dict[tuple[int, int], int] = {}
        self.alive = True
        self.visible = True
        self.iconic = False

    def WindowFromPoint(self, pt) -> int:
        return self._window_from_point.get(tuple(pt), 0)

    def GetAncestor(self, hwnd, flag) -> int:
        if flag == 2:  # GA_ROOT
            return self._window_roots.get(hwnd, hwnd)
        if flag == 1:  # GA_PARENT
            return self._window_parents.get(hwnd, hwnd)
        if flag == 3:  # GA_ROOTOWNER
            return self._window_roots.get(hwnd, hwnd)
        return hwnd

    def GetWindowThreadProcessId(self, hwnd) -> tuple[int, int]:
        return hwnd, self._window_pids.get(hwnd, TARGET_PID)

    def GetForegroundWindow(self) -> int:
        return self.foreground

    def IsWindow(self, hwnd) -> int:
        return 1 if self.alive else 0

    def IsWindowVisible(self, hwnd) -> int:
        return 1 if self.visible else 0

    def IsIconic(self, hwnd) -> int:
        return 1 if self.iconic else 0

    def ShowWindow(self, hwnd, cmd) -> int:
        return 1

    def SetForegroundWindow(self, hwnd) -> int:
        self.foreground = hwnd
        return 1


@pytest.fixture
def fake_win32(monkeypatch):
    fake = _FakeWin32()
    import win32con
    import win32gui
    import win32process

    monkeypatch.setattr(win32gui, "WindowFromPoint", fake.WindowFromPoint)
    monkeypatch.setattr(win32gui, "GetAncestor", fake.GetAncestor)
    monkeypatch.setattr(win32gui, "GetForegroundWindow", fake.GetForegroundWindow)
    monkeypatch.setattr(win32gui, "IsWindow", fake.IsWindow)
    monkeypatch.setattr(win32gui, "IsWindowVisible", fake.IsWindowVisible)
    monkeypatch.setattr(win32gui, "IsIconic", fake.IsIconic)
    monkeypatch.setattr(win32gui, "ShowWindow", fake.ShowWindow)
    monkeypatch.setattr(win32gui, "SetForegroundWindow", fake.SetForegroundWindow)
    monkeypatch.setattr(win32process, "GetWindowThreadProcessId", fake.GetWindowThreadProcessId)
    return fake


def _target(pid: int = TARGET_PID, rect: tuple = TARGET_RECT) -> TargetInfo:
    return TargetInfo(
        handle=TARGET_HWND,
        pid=pid,
        tid=1,
        class_name="MPF",
        title="MPF (Download and Upload Form)",
        exe_name="mpf.exe",
        client_rect=rect,
    )


def _sandbox(target: TargetInfo, **config) -> ExecutionSandbox:
    sb = ExecutionSandbox(SandboxConfig(**config))
    sb.attach(target)
    return sb


def test_pid_zero_click_inside_client_rect_allowed(fake_win32) -> None:
    sb = _sandbox(_target(pid=0))
    ok, reason = sb.validate_click(300, 400)
    assert ok is True
    assert "client rect" in reason
    sb.detach()


def test_pid_zero_click_outside_client_rect_blocked(fake_win32) -> None:
    sb = _sandbox(_target(pid=0))
    ok, reason = sb.validate_click(900, 900)
    assert ok is False
    assert "outside target client rect" in reason
    sb.detach()


def test_click_on_foreign_window_blocked(fake_win32) -> None:
    # Click resolves to a window owned by another root window and process.
    fake_win32._window_from_point[(300, 400)] = 9999
    fake_win32._window_roots[9999] = 7777  # foreign root
    fake_win32._window_pids[9999] = 777  # foreign pid
    sb = _sandbox(_target())
    ok, reason = sb.validate_click(300, 400)
    assert ok is False
    assert "foreign window" in reason
    sb.detach()


def test_click_on_target_root_allowed(fake_win32) -> None:
    fake_win32._window_from_point[(300, 400)] = 1234
    fake_win32._window_roots[1234] = TARGET_HWND
    sb = _sandbox(_target())
    ok, reason = sb.validate_click(300, 400)
    assert ok is True
    assert "root window" in reason
    sb.detach()


def test_click_within_client_rect_allowed_when_hierarchy_unclear(fake_win32) -> None:
    # WindowFromPoint returns the target root itself.
    fake_win32._window_from_point[(300, 400)] = TARGET_HWND
    sb = _sandbox(_target())
    ok, reason = sb.validate_click(300, 400)
    assert ok is True
    sb.detach()


def test_mouse_check_disabled_always_allows(fake_win32) -> None:
    sb = _sandbox(_target(pid=0), check_mouse=False)
    ok, _ = sb.validate_click(900, 900)
    assert ok is True
    sb.detach()


def test_validate_click_without_target_blocked(fake_win32) -> None:
    sb = ExecutionSandbox(SandboxConfig())
    ok, reason = sb.validate_click(300, 400)
    assert ok is False
    assert "no target attached" in reason


def test_empty_client_rect_blocks_outside_click(fake_win32) -> None:
    sb = _sandbox(_target(pid=0, rect=(0, 0, 0, 0)))
    ok, reason = sb.validate_click(300, 400)
    assert ok is False
    sb.detach()


def test_keyboard_allowed_when_foreground_is_target(fake_win32) -> None:
    fake_win32.foreground = TARGET_HWND
    sb = _sandbox(_target())
    ok, reason = sb.validate_keyboard()
    assert ok is True
    assert reason == "ok"
    sb.detach()


def test_keyboard_blocked_when_foreground_is_foreign(fake_win32) -> None:
    fake_win32.foreground = 7777  # another app has focus
    sb = _sandbox(_target(), auto_refocus=False)
    ok, reason = sb.validate_keyboard()
    assert ok is False
    assert "focus lost" in reason
    sb.detach()


def test_keyboard_refocuses_foreign_foreground(fake_win32) -> None:
    fake_win32.foreground = 7777  # start with another app focused
    sb = _sandbox(_target(), auto_refocus=True)
    ok, _ = sb.validate_keyboard()
    assert ok is True
    assert fake_win32.foreground == TARGET_HWND  # refocused to target
    sb.detach()


def test_keyboard_blocked_when_refocus_fails(fake_win32) -> None:
    # Foreground reports the target handle but SetForegroundWindow cannot win.
    def stubborn_foreground() -> int:
        return 7777

    fake_win32.foreground = 7777
    import win32gui

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(win32gui, "SetForegroundWindow", lambda hwnd: 0)
    fake_win32.GetForegroundWindow = stubborn_foreground
    sb = _sandbox(_target(), auto_refocus=True)
    ok, reason = sb.validate_keyboard()
    assert ok is False
    assert "foreground != target" in reason
    sb.detach()
    monkeypatch.undo()


def test_watchdog_pauses_when_focus_lost(fake_win32) -> None:
    fake_win32.foreground = 7777  # another app grabs focus
    sb = _sandbox(_target(), auto_refocus=False)
    sb.set_running()
    # Simulate repeated focus losses (10 ticks -> pause).
    for _ in range(11):
        sb._watchdog_tick()
    assert sb.is_paused is True
    sb.detach()


def test_watchdog_resumes_when_focus_restored(fake_win32) -> None:
    sb = _sandbox(_target(), auto_refocus=False)
    sb.set_running()
    fake_win32.foreground = 7777
    for _ in range(11):
        sb._watchdog_tick()
    assert sb.is_paused is True
    # User brings MPF back to the foreground.
    fake_win32.foreground = TARGET_HWND
    sb._watchdog_tick()
    assert sb.is_paused is False
    assert sb.state == SandboxState.RUNNING
    sb.detach()
