"""Low-level mouse click detection.

Installs a ``WH_MOUSE_LL`` hook on a background thread so the agent can wait
for the user to click the first editable field of the target window (the
``StartControl`` anchor). The hook thread only records screen points; UIA
resolution of the clicked control happens later on the main thread, so no COM
objects cross thread boundaries.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from ctypes import wintypes

from atlas.core.logging import logger

WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202

_MSLLHOOKSTRUCT = None


def _hook_struct() -> type:
    global _MSLLHOOKSTRUCT

    if _MSLLHOOKSTRUCT is None:

        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("pt", wintypes.POINT),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        _MSLLHOOKSTRUCT = MSLLHOOKSTRUCT
    return _MSLLHOOKSTRUCT


class MouseClickListener:
    """Background low-level mouse hook exposing click points."""

    def __init__(self) -> None:
        self._clicks: queue.Queue[tuple[int, int]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._keep = threading.Event()
        self._hook_ready = threading.Event()
        self._hook_error: Exception | None = None
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._hook: int | None = None
        self._callback = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._keep.set()
        self._callback = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )(self._on_event)
        self._thread = threading.Thread(
            target=self._run,
            name="atlas-click-hook",
            daemon=True,
        )
        self._thread.start()
        self._hook_ready.wait(timeout=3.0)
        if self._hook_error is not None:
            raise self._hook_error

    def stop(self) -> None:
        self._keep.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- waiting -------------------------------------------------------------

    def drain(self) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        while True:
            try:
                points.append(self._clicks.get_nowait())
            except queue.Empty:
                return points

    def wait_for_click(self, timeout: float, poll: float = 0.05) -> tuple[int, int] | None:
        """Block up to ``timeout`` seconds for the next click point."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                return self._clicks.get_nowait()
            except queue.Empty:
                pass
            if self._hook_error is not None:
                raise self._hook_error
            time.sleep(poll)
        return None

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        self._configure_prototypes()
        try:
            module = self._kernel32.GetModuleHandleW(None)
            self._hook = self._user32.SetWindowsHookExW(
                WH_MOUSE_LL,
                self._callback,
                module,
                0,
            )
            if not self._hook:
                raise ctypes.WinError(ctypes.get_last_error() or 126)
        except Exception as exc:  # noqa: BLE001
            self._hook_error = exc
            self._hook_ready.set()
            logger.error("mouse hook install failed: {}", exc)
            return
        self._hook_ready.set()
        logger.debug("mouse click listener active")

        message = wintypes.MSG()
        try:
            while self._keep.is_set():
                result = self._user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                self._user32.TranslateMessage(ctypes.byref(message))
                self._user32.DispatchMessageW(ctypes.byref(message))
        finally:
            if self._hook:
                try:
                    self._user32.UnhookWindowsHookEx(self._hook)
                except Exception:
                    pass
                self._hook = None

    @staticmethod
    def _configure_prototypes() -> None:
        """Explicit signatures so 64-bit HANDLEs are not truncated to 32 bits."""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.HMODULE,
            wintypes.DWORD,
        ]
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        user32.CallNextHookEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.GetMessageW.restype = ctypes.c_int
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    def _on_event(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == 0 and w_param in (WM_LBUTTONDOWN, WM_LBUTTONUP):
            try:
                struct = _hook_struct()
                info = ctypes.cast(l_param, ctypes.POINTER(struct)).contents
                self._clicks.put((int(info.pt.x), int(info.pt.y)))
            except Exception:
                pass
        if self._hook:
            return self._user32.CallNextHookEx(self._hook, n_code, w_param, l_param)
        return self._user32.CallNextHookEx(None, n_code, w_param, l_param)


__all__ = ["MouseClickListener"]
