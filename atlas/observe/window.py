"""Window attachment.

The agent attaches to a single target window (selected by the user clicking its
first editable field, or by title/handle). From then on it observes ONLY that
window's client area - ignoring taskbar, desktop, notifications, other monitors
and any overlay it draws on top.

Attachment discovers the REAL automation root instead of stopping at the first
title match. For Electron/Chrome-based apps (like MPF) the top-level window may
have pid=0, but its descendants contain the actual application with editable
controls. The attacher:

1. Enumerates all top-level windows matching the title
2. For each candidate, recursively discovers the UIA tree
3. Validates based on the presence of editable controls (not just pid>0)
4. Falls back to interactive click-to-attach if ambiguous

Never attaches to the desktop, shell, or a window without editable controls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import win32api
import win32con
import win32gui

from atlas.core.events import EventType, get_event_bus
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
        bus = get_event_bus()
        bus.publish(EventType.STATE_CHANGED, {"state": "attaching"})
        handle = win32gui.GetForegroundWindow()
        if not handle:
            raise AttachError("no foreground window available")
        target = self._resolve(handle)
        self._verify_and_attach(target)
        return target

    def attach_by_title(self, title: str) -> WindowTarget:
        """Attach to the best visible top-level window whose title matches.

        Uses recursive UI root discovery: if the top-level window has pid=0
        (common for Electron/Chrome apps), we inspect its descendants for
        editable controls and attach to the real application root instead of
        rejecting it.
        """
        import win32gui

        bus = get_event_bus()
        bus.publish(EventType.STATE_CHANGED, {"state": "attaching"})
        
        # Enumerate ALL top-level windows matching the title, including pid=0.
        candidates: list[dict] = []
        title_lower = title.lower()

        def _collect(handle: int, _: Any = None) -> None:
            try:
                if not win32gui.IsWindowVisible(handle):
                    return
                window_title = win32gui.GetWindowText(handle) or ""
                if title_lower not in window_title.lower():
                    return
                class_name = win32gui.GetClassName(handle) or ""
                thread_id, pid = 0, 0
                try:
                    thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
                except Exception:
                    pass
                exe, exe_path = self._executable_for(pid)
                candidates.append({
                    "handle": handle,
                    "title": window_title,
                    "class_name": class_name,
                    "process_id": pid,
                    "thread_id": thread_id,
                    "executable": exe,
                    "exe_path": exe_path,
                })
            except Exception:
                pass

        win32gui.EnumWindows(_collect, None)
        
        if not candidates:
            raise AttachError(self._no_match_detail(title))
        
        # Try each candidate with recursive UIA discovery.
        for candidate in candidates:
            target = self._resolve(candidate["handle"])
            try:
                discovered = self._discover_ui_root(target)
                if discovered is not None:
                    self._verify_and_attach(discovered)
                    logger.info(
                        "attached to window {!r} (pid={}, exe={}, discovered_via=uia_tree)",
                        discovered.title,
                        discovered.process_id,
                        discovered.exe_path or discovered.executable,
                    )
                    return discovered
            except Exception as exc:
                logger.debug("candidate {!r} failed: {}", target.title, exc)
                continue
        
        raise AttachError(self._no_match_detail(title))

    def _discover_ui_root(self, target: WindowTarget) -> WindowTarget | None:
        """Recursively discover the real UI automation root for a candidate.

        If the top-level window has pid=0 or no editable controls, inspect its
        descendants to find the first pane/element that contains editable
        controls. This handles Electron/Chrome apps where the wrapper window
        has pid=0 but the actual app is in a child HWND.
        """
        try:
            from atlas.observe.uia import UiaBackend
            backend = UiaBackend.instance()
            nodes = backend.descendants(target.handle)
            editable = [n for n in nodes if n.editable]
            if editable:
                # Top-level window has editable controls - use it directly.
                return target
            # No editable controls at top level. Try to find a child window
            # that has editable controls.
            child_handle = self._find_child_with_controls(target.handle)
            if child_handle and child_handle != target.handle:
                child_target = self._resolve(child_handle)
                child_nodes = backend.descendants(child_handle)
                child_editable = [n for n in child_nodes if n.editable]
                if child_editable:
                    logger.info(
                        "discovered child window with {} editable controls (parent={}, child={})",
                        len(child_editable),
                        target.handle,
                        child_handle,
                    )
                    return child_target
        except Exception as exc:
            logger.debug("_discover_ui_root failed: {}", exc)
        return None

    def _find_child_with_controls(self, parent_handle: int) -> int | None:
        """Find the first child window that might contain editable controls."""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            
            found: list[int] = []
            def _enum(handle: int, _: Any) -> bool:
                try:
                    if win32gui.IsWindowVisible(handle):
                        found.append(handle)
                except Exception:
                    pass
                return True
            
            win32gui.EnumChildWindows(parent_handle, _enum, None)
            
            # Check each child for editable controls.
            from atlas.observe.uia import UiaBackend
            backend = UiaBackend.instance()
            for handle in found:
                try:
                    nodes = backend.descendants(handle)
                    editable = [n for n in nodes if n.editable]
                    if editable:
                        return handle
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("_find_child_with_controls failed: {}", exc)
        return None

    def _verify_and_attach(self, target: WindowTarget) -> None:
        """Verify the target tree, generate UIA diagnostics, then attach capture."""
        bus = get_event_bus()
        self._verify_target_tree(target)
        # Generate UIA diagnostics (Step 2) to debug/uia/.
        try:
            self._dump_uia_diagnostics(target)
        except Exception as exc:
            logger.debug("uia diagnostics dump failed: {}", exc)
        self._capture.attach(target.handle, target.title)
        bus.publish(EventType.STATE_CHANGED, {"state": "inspecting_ui"})

    @staticmethod
    def _dump_uia_diagnostics(target: WindowTarget) -> None:
        """Write the full UIA diagnostic set to ``debug/uia/`` (Step 2)."""
        try:
            from atlas.observe.uia import UiaBackend

            backend = UiaBackend.instance()
            out = Path("debug/uia")
            backend.dump_diagnostics(target.handle, out)
        except Exception as exc:
            logger.debug("uia diagnostics failed: {}", exc)

    def attach_by_handle(self, handle: int) -> WindowTarget:
        if not win32gui.IsWindow(handle):
            raise AttachError(f"invalid window handle {handle}")
        target = self._resolve(handle)
        self._capture.attach(target.handle, target.title)
        return target

    def attach_by_click(self, timeout: float = 300.0) -> WindowTarget:
        """Attach to the window the user clicks (Step 5 / Step 8).

        Waits for a mouse click, then resolves the real top-level application
        window under the cursor using ``WindowFromPoint`` + ``GetAncestor``.
        This is the reliable way to attach to Electron/Chrome-based apps (like
        MPF) where ``EnumWindows`` finds ghost windows with pid=0.

        Loops forever until a valid application window is selected, timeout
        expires, or ESC is pressed. Never attaches to desktop/shell.
        """
        from atlas.observe.click_hook import MouseClickListener

        bus = get_event_bus()
        bus.publish(EventType.STATE_CHANGED, {"state": "attaching"})
        logger.info("ATTACH MODE: click the MPF application window to attach")
        logger.info("  (click anywhere inside the MPF window's client area)")

        listener = MouseClickListener()
        listener.start()
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                click = listener.wait_for_click(deadline - time.time())
                if click is None:
                    raise AttachError(
                        f"no click received within {timeout:.0f}s - "
                        "click the MPF application window to attach"
                    )
                x, y = click
                target = self._resolve_window_at_point(x, y)
                if target is None:
                    logger.warning(
                        "Ignored click at ({}, {}) - not a valid application window. "
                        "Click inside the MPF application window. Waiting...",
                        x, y,
                    )
                    continue
                # Valid target found - verify and attach.
                try:
                    self._verify_and_attach(target)
                    self._print_attach_summary(target)
                    return target
                except AttachError as exc:
                    logger.warning("Attach attempt failed: {}. Waiting...", exc)
                    continue
                except Exception as exc:
                    logger.warning("Unexpected error during attach: {}. Waiting...", exc)
                    continue
            raise AttachError(f"attach timed out after {timeout:.0f}s")
        finally:
            listener.stop()

    def _resolve_window_at_point(self, x: int, y: int) -> WindowTarget | None:
        """Resolve the real top-level application window at a screen point.

        Uses ``WindowFromPoint`` to get the immediate window under the cursor,
        then ``GetAncestor(GA_ROOTOWNER)`` to climb to the top-level owner.
        Rejects windows with pid=0, desktop shells, and invisible windows.
        """
        try:
            handle = win32gui.WindowFromPoint((x, y))
            if not handle:
                return None
            # Climb to the root owner using GetAncestor (GA_ROOTOWNER = 3).
            root = self._get_ancestor(handle, 3)
            if root:
                handle = root
            else:
                # Fallback: climb via GetParent.
                while win32gui.GetParent(handle):
                    handle = win32gui.GetParent(handle)
            target = self._resolve(handle)
            # Reject invalid windows.
            if not self._is_valid_click_target(target):
                logger.warning(
                    "window at ({}, {}) is not a valid target: "
                    "title={!r} pid={} class={!r}",
                    x, y, target.title, target.process_id, target.class_name,
                )
                return None
            return target
        except Exception as exc:
            logger.debug("resolve_window_at_point failed: {}", exc)
            return None

    @staticmethod
    def _get_ancestor(handle: int, flags: int) -> int | None:
        """Call GetAncestor via ctypes (not in win32gui)."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.GetAncestor.restype = ctypes.c_void_p
            user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            result = user32.GetAncestor(handle, flags)
            return int(result) if result else None
        except Exception:
            return None

    @staticmethod
    def _is_valid_click_target(target: WindowTarget) -> bool:
        """A click-resolved window is potentially valid if it looks like an app.

        Does NOT reject pid=0 immediately - Electron/Chrome apps often have
        pid=0 on the wrapper window. The final validation happens in
        _verify_target_tree which checks for editable controls.
        """
        if not target.title.strip():
            return False
        # Reject desktop shells.
        if target.class_name in {"Progman", "WorkerW", "Shell_TrayWnd", "DV2ControlHost"}:
            return False
        # Reject obvious non-app windows.
        if target.class_name in {"Windows.UI.Core.CoreWindow"}:
            # Might be a notification toast - check if it has editable controls.
            pass  # Let _verify_target_tree decide.
        # Reject invisible / zero-size windows.
        try:
            if not win32gui.IsWindowVisible(target.handle):
                return False
            rect = win32gui.GetWindowRect(target.handle)
            if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _print_attach_summary(target: WindowTarget) -> None:
        """Print the full attachment summary (Step 6)."""
        from atlas.observe.uia import UiaBackend

        print("-" * 50)
        print("ATTACHED")
        print(f"  Process:     {target.executable or target.exe_path or '?'}")
        print(f"  PID:         {target.process_id}")
        print(f"  HWND:        {target.handle}")
        print(f"  Class:       {target.class_name}")
        print(f"  Title:       {target.title}")
        try:
            origin = UiaBackend.client_origin(target.handle)
            size = UiaBackend.client_size(target.handle)
            print(f"  Client Size: {size[0]}x{size[1]} at ({origin[0]}, {origin[1]})")
        except Exception:
            pass
        try:
            backend = UiaBackend.instance()
            nodes = backend.descendants(target.handle)
            editable = [n for n in nodes if n.editable]
            buttons = [n for n in nodes if n.control_type in {"Button", "SplitButton"}]
            combos = [n for n in nodes if n.control_type == "ComboBox"]
            edits = [n for n in nodes if n.control_type == "Edit"]
            scrolls = [n for n in nodes if n.control_type == "ScrollBar"]
            print(f"  UIA Root:    {'yes' if nodes else 'no'}")
            print(f"  Controls:    {len(nodes)}")
            print(f"  Editable:    {len(editable)}")
            print(f"  Buttons:     {len(buttons)}")
            print(f"  ComboBoxes:  {len(combos)}")
            print(f"  TextBoxes:   {len(edits)}")
            print(f"  ScrollBars:  {len(scrolls)}")
        except Exception as exc:
            print(f"  UIA:         error: {exc}")
        print("-" * 50)

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

        For Electron/Chrome apps, pid may be 0 on the wrapper window. We accept
        pid=0 if the UIA tree contains editable controls. The final validation
        is based on the presence of editable controls, not just pid>0.
        """
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
        # Accept windows with editable controls even if pid=0 (Electron/Chrome).
        editable = [n for n in nodes if n.editable]
        if not editable:
            raise AttachError(
                f"window {target.title!r} (pid={target.process_id}) has UIA controls "
                "but no editable fields - not a data-entry form. "
                "Run 'python main.py diagnose --title <MPF>' to inspect the window tree."
            )

    @staticmethod
    def _no_match_detail(title: str) -> str:
        import win32gui

        all_windows: list[dict] = []
        title_lower = title.lower()

        def _collect(handle: int, _: Any = None) -> None:
            try:
                if not win32gui.IsWindowVisible(handle):
                    return
                window_title = win32gui.GetWindowText(handle) or ""
                if title_lower not in window_title.lower():
                    return
                class_name = win32gui.GetClassName(handle) or ""
                thread_id, pid = 0, 0
                try:
                    thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
                except Exception:
                    pass
                all_windows.append({
                    "title": window_title,
                    "class_name": class_name,
                    "process_id": pid,
                    "thread_id": thread_id,
                })
            except Exception:
                pass

        win32gui.EnumWindows(_collect, None)
        
        if not all_windows:
            return (
                f"no visible window found matching title {title!r}.\n"
                f"Open the MPF (Download and Upload Form) window first, then re-run."
            )
        
        lines = []
        for info in all_windows:
            pid = info.get("process_id", 0)
            class_name = info.get("class_name", "")
            lines.append(
                f"  - {info['title']!r} (pid={pid}, class={class_name!r})"
            )
        
        return (
            f"{len(all_windows)} window(s) matched title {title!r}.\n"
            "Recursive UIA discovery was attempted but none contained editable controls.\n"
            "Open the MPF window and click inside it, then re-run with --attach.\n"
            + "\n".join(lines)
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
