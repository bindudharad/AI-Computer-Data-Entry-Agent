"""Window attachment.

The agent attaches to a single target window (selected by the user clicking its
first editable field, or by title/handle). From then on it observes ONLY that
window's client area - ignoring taskbar, desktop, notifications, other monitors
and any overlay it draws on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import win32api
import win32con
import win32gui

from atlas.core.logging import logger
from atlas.vision.capture import WindowCapture


class AttachError(RuntimeError):
    """Raised when a window cannot be attached/focused."""


@dataclass
class WindowTarget:
    """A resolved target window."""

    handle: int
    title: str
    process_id: int
    executable: str = ""
    exe_path: str = ""
    class_name: str = ""
    thread_id: int = 0

    def to_dict(self) -> dict:
        return {
            "handle": self.handle,
            "title": self.title,
            "process_id": self.process_id,
            "executable": self.executable,
            "exe_path": self.exe_path,
            "class_name": self.class_name,
            "thread_id": self.thread_id,
        }


class WindowAttacher:
    """Resolves and brings a target window to the foreground.

    ``attach`` is called after the user clicks the first editable field (the
    foreground window at that moment becomes the target). The foreground window
    is the application the user just interacted with, which is exactly the
    window we should confine observation to.
    """

    def __init__(self, capture: WindowCapture) -> None:
        self._capture = capture

    def attach_foreground(self) -> WindowTarget:
        """Attach to the currently foreground (active) top-level window."""
        handle = win32gui.GetForegroundWindow()
        if not handle:
            raise AttachError("no foreground window available")
        target = self._resolve(handle)
        self._verify_target_tree(target)
        self._capture.attach(target.handle, target.title)
        return target

    def attach_by_title(self, title: str) -> WindowTarget:
        """Attach to the best visible top-level window whose title matches.

        Uses a scored, validated lookup (exact/prefix/substring, PID > 0, real
        executable, non-desktop class) instead of the first substring match, so
        an IDE window that merely contains the query can never be attached.
        """
        from atlas.vision.capture import WindowGeometry

        candidate = WindowGeometry.best_window_by_title(title)
        if candidate is None:
            raise AttachError(self._no_match_detail(title))
        target = self._resolve(candidate["handle"])
        self._verify_target_tree(target)
        self._capture.attach(target.handle, target.title)
        logger.info(
            "attached to window {!r} (pid={}, exe={})",
            target.title,
            target.process_id,
            target.exe_path or target.executable,
        )
        return target

    def attach_by_handle(self, handle: int) -> WindowTarget:
        if not win32gui.IsWindow(handle):
            raise AttachError(f"invalid window handle {handle}")
        target = self._resolve(handle)
        self._capture.attach(target.handle, target.title)
        return target

    def bring_to_front(self, target: WindowTarget) -> None:
        """Restore and foreground the target window (best-effort)."""
        try:
            if win32gui.IsIconic(target.handle):
                win32gui.ShowWindow(target.handle, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(target.handle)
            time.sleep(0.15)
        except Exception as exc:
            logger.warning("could not bring window to front: {}", exc)

    def verify_focused(self, target: WindowTarget) -> bool:
        """Check that the target window is currently the foreground window."""
        try:
            return win32gui.GetForegroundWindow() == target.handle
        except Exception:
            return False

    def focus_and_verify(self, target: WindowTarget, retries: int = 3) -> bool:
        for _ in range(retries):
            if self.verify_focused(target):
                return True
            self.bring_to_front(target)
            time.sleep(0.2)
        return self.verify_focused(target)

    def _resolve(self, handle: int) -> WindowTarget:
        title = win32gui.GetWindowText(handle) or ""
        class_name = ""
        try:
            class_name = win32gui.GetClassName(handle) or ""
        except Exception:
            class_name = ""
        thread_id = 0
        pid = 0
        try:
            thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
        except Exception:
            pass
        exe, exe_path = self._executable_for(pid)
        return WindowTarget(
            handle=handle,
            title=title,
            process_id=pid,
            executable=exe,
            exe_path=exe_path,
            class_name=class_name,
            thread_id=thread_id,
        )

    @staticmethod
    def _executable_for(pid: int) -> tuple[str, str]:
        """Return (exe name, full exe path) for a process id."""
        if pid <= 0:
            return "", ""
        try:
            import psutil

            proc = psutil.Process(pid)
            try:
                path = proc.exe() or ""
            except Exception:
                path = ""
            try:
                name = proc.name() or ""
            except Exception:
                name = ""
            return name, path
        except Exception:
            return "", ""

    def _verify_target_tree(self, target: WindowTarget) -> None:
        """Verify the target exposes a non-empty UIA control tree.

        Aborts with a diagnostic instead of silently proceeding to vision-only
        when the window is not really the application we expect (this is the
        fix for attaching to the desktop / an IDE window whose UIA tree is
        empty and then silently falling back).
        """
        if target.process_id <= 0:
            raise AttachError(
                f"window {target.title!r} has process_id=0 (System Idle Process / "
                "desktop shell) - not a real application window"
            )
        try:
            from atlas.observe.uia import UiaBackend

            nodes = UiaBackend.instance().descendants(target.handle)
        except Exception as exc:
            raise AttachError(f"UIA inspection failed for {target.title!r}: {exc}") from exc
        if not nodes:
            raise AttachError(
                f"window {target.title!r} (pid={target.process_id}) exposes no UIA "
                "controls - the target may be a virtual/hidden window, running in "
                "another session, or the real MPF window is not open. "
                "Run 'python main.py diagnose --title <MPF>' to inspect the window tree."
            )

    @staticmethod
    def _no_match_detail(title: str) -> str:
        from atlas.vision.capture import WindowGeometry

        all_windows = WindowGeometry.find_windows_by_title(title)
        if not all_windows:
            return (
                f"no visible window found matching title {title!r}.\n"
                f"Open the MPF (Download and Upload Form) window first, then re-run."
            )
        invalid = []
        for info in all_windows:
            reason = _invalid_reason(info)
            invalid.append(f"  - {info['title']!r} (pid={info['process_id']}, class={info['class_name']!r}) {reason}")
        return (
            f"{len(all_windows)} window(s) matched title {title!r} but none is a valid "
            f"target window:\n" + "\n".join(invalid)
        )


def window_under_cursor() -> WindowTarget | None:
    """Return the top-level window under the current cursor position."""
    try:
        pos = win32api.GetCursorPos()
        handle = win32gui.WindowFromPoint(pos)
        if not handle:
            return None
        # climb to the top-level owner
        while win32gui.GetParent(handle):
            handle = win32gui.GetParent(handle)
        return _target_from_handle(handle)
    except Exception:
        return None


def _target_from_handle(handle: int) -> WindowTarget | None:
    try:
        title = win32gui.GetWindowText(handle) or ""
        class_name = win32gui.GetClassName(handle) or ""
        thread_id = 0
        pid = 0
        try:
            thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
        except Exception:
            pass
        exe, exe_path = WindowAttacher._executable_for(pid)
        return WindowTarget(
            handle=handle,
            title=title,
            process_id=pid,
            executable=exe,
            exe_path=exe_path,
            class_name=class_name,
            thread_id=thread_id,
        )
    except Exception:
        return None


def _invalid_reason(info: dict) -> str:
    """Why a candidate window was rejected by the attach validation."""
    pid = info.get("process_id") or 0
    if pid <= 0:
        return "-> rejected: no valid process (pid=0)"
    exe = (info.get("executable") or "").split("\\")[-1]
    class_name = info.get("class_name") or ""
    if class_name in {"Progman", "WorkerW", "Shell_TrayWnd", "DV2ControlHost"}:
        return f"-> rejected: desktop/taskbar shell (class={class_name!r})"
    if not exe:
        return "-> rejected: no executable"
    return "-> rejected: system process (not an application window)"


__all__ = ["WindowAttacher", "WindowTarget", "AttachError", "window_under_cursor"]
