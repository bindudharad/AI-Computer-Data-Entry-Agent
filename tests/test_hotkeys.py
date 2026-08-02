"""Tests for the global hotkey manager.

The real win32 API needs a desktop session and a message pump, so these tests
drive the message loop with a fake ``PeekMessage`` that replays queued
``WM_HOTKEY`` messages (exactly what the loop reads in production).
"""

from __future__ import annotations

import threading
import time

import pytest

from atlas.act.hotkeys import HotkeyManager


class _FakeWin32:
    def __init__(self, messages: list[tuple[int, int, int, int]] | None = None) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []
        self._messages = list(messages or [])
        self._pump = threading.Event()

    def RegisterHotKey(self, hwnd, hotkey_id, modifiers, vk) -> None:
        self.registered.append((hwnd, hotkey_id, modifiers, vk))

    def UnregisterHotKey(self, hwnd, hotkey_id) -> None:
        self.unregistered.append(hotkey_id)

    def PeekMessage(self, hwnd, min, max, flags):
        if self._messages:
            bgot, hwnd, message, wparam, lparam, t, pt = self._messages.pop(0)
            return (1, hwnd, message, wparam, lparam, t, pt)
        self._pump.wait(timeout=0.1)
        return None

    def feed(self, hotkey_id: int) -> None:
        self._messages.append((1, None, 0x0312, hotkey_id, 0, 0, (0, 0)))
        self._pump.set()


@pytest.fixture
def fake_win32(monkeypatch):
    fake = _FakeWin32()
    import win32con
    import win32gui

    monkeypatch.setattr(win32gui, "RegisterHotKey", fake.RegisterHotKey)
    monkeypatch.setattr(win32gui, "UnregisterHotKey", fake.UnregisterHotKey)
    monkeypatch.setattr(win32gui, "PeekMessage", fake.PeekMessage)
    monkeypatch.setattr(win32con, "WM_HOTKEY", 0x0312)
    return fake


def test_start_registers_four_hotkeys(fake_win32) -> None:
    mgr = HotkeyManager()
    mgr.register("stop", lambda: None)
    mgr.start()
    time.sleep(0.2)
    assert len(fake_win32.registered) == 4
    mgr.stop()


def test_hotkey_dispatch_invokes_callback(fake_win32) -> None:
    mgr = HotkeyManager()
    calls: list[str] = []
    mgr.register("stop", lambda: calls.append("stop"))
    mgr.register("pause", lambda: calls.append("pause"))
    mgr.start()
    try:
        time.sleep(0.2)
        fake_win32.feed(1)  # ESC -> stop
        fake_win32.feed(2)  # Ctrl+Shift+S -> pause
        time.sleep(0.4)
    finally:
        mgr.stop()
    assert calls == ["stop", "pause"]


def test_stop_unregisters_all_hotkeys(fake_win32) -> None:
    mgr = HotkeyManager()
    mgr.register("quit", lambda: None)
    mgr.start()
    time.sleep(0.2)
    assert len(fake_win32.registered) == 4
    mgr.stop()
    assert sorted(fake_win32.unregistered) == sorted(h for _, h, _, _ in fake_win32.registered)
