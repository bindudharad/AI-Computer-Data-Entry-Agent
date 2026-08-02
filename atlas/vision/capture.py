"""Window client-area capture.

Attaches to a single target window and captures ONLY its client area, returning
the image as a numpy array together with the offset of the client area on the
physical screen (so element boxes can be translated to absolute screen
coordinates for mouse actions).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from atlas.core.logging import logger

try:
    import mss
    import mss.tools
except ImportError:  # pragma: no cover - dependency guard
    mss = None  # type: ignore[assignment]


@dataclass
class ClientArea:
    """A captured client area plus the geometry needed to act on it."""

    image: np.ndarray  # RGB (H, W, 3)
    left: int  # screen x of client origin
    top: int  # screen y of client origin
    width: int
    height: int

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Translate capture-relative coordinates to absolute screen coords."""
        return self.left + x, self.top + y

    @property
    def offset(self) -> tuple[int, int]:
        return (self.left, self.top)

    def save(self, path: Any) -> None:
        from PIL import Image

        Image.fromarray(self.image).save(path)


class ScreenGrabber:
    """Low-level screen grabber using mss, with a BitBlt fallback.

    ``gdi32.GetDIBits() failed.`` errors are a known mss failure mode: once the
    mss instance's internal DC goes stale (display changes, GPU/driver reset,
    prolonged capture) every subsequent grab fails and the agent stalls. We
    re-initialise the mss session on failure and, if it keeps failing, switch to
    a PIL ``ImageGrab`` BitBlt path for a cooldown window so the loop never
    deadlocks on a broken grabber.
    """

    #: Consecutive failures that trigger the BitBlt fallback.
    _FALLBACK_AFTER = 2

    #: How long (seconds) to stay on the BitBlt fallback once engaged.
    _FALLBACK_WINDOW = 30.0

    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError("mss is required for screen capture")
        self._mss = mss.mss()
        self._consecutive_failures = 0
        self._fallback_until = 0.0

    def grab_rect(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Grab a rectangle and return an RGB numpy array."""
        if time.time() < self._fallback_until:
            return self._grab_bitblt(left, top, width, height)
        try:
            image = self._grab_mss(left, top, width, height)
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._FALLBACK_AFTER:
                self._fallback_until = time.time() + self._FALLBACK_WINDOW
                logger.warning(
                    "mss grab failed {}x ({}): switching to BitBlt fallback", self._consecutive_failures, exc
                )
                return self._grab_bitblt(left, top, width, height)
            logger.debug("mss grab failed ({}): re-initialising grabber: {}", self._consecutive_failures, exc)
            self._reinit()
            try:
                image = self._grab_mss(left, top, width, height)
            except Exception as retry_exc:
                logger.warning("mss grab failed after reinit ({}): using BitBlt fallback", retry_exc)
                return self._grab_bitblt(left, top, width, height)
        self._consecutive_failures = 0
        return image

    def _grab_mss(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        shot = self._mss.grab({"left": left, "top": top, "width": width, "height": height})
        arr = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

    def _grab_bitblt(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """BitBlt-based capture via PIL (different GDI path than mss)."""
        from PIL import ImageGrab

        if width <= 0 or height <= 0:
            raise ValueError("grab region must be non-empty")
        img = ImageGrab.grab(bbox=(left, top, left + width, top + height))
        return np.asarray(img.convert("RGB"))

    def _reinit(self) -> None:
        try:
            self._mss.close()
        except Exception:
            pass
        self._mss = mss.mss()

    def close(self) -> None:
        try:
            self._mss.close()
        except Exception:
            pass


class WindowGeometry:
    """Windows API helpers for locating windows and their client areas."""

    @staticmethod
    def client_area(handle: int) -> tuple[int, int, int, int]:
        """Return (left, top, width, height) of the client area in screen coords."""
        import win32gui

        rect = win32gui.GetClientRect(handle)
        width, height = rect[2], rect[3]
        origin = win32gui.ClientToScreen(handle, (0, 0))
        return origin[0], origin[1], width, height

    @staticmethod
    def window_title(handle: int) -> str:
        import win32gui

        try:
            return win32gui.GetWindowText(handle) or ""
        except Exception:
            return ""

    @staticmethod
    def find_windows_by_title(title: str) -> list[dict]:
        """Every visible top-level window whose title contains ``title``.

        Returns full Win32 metadata per candidate so callers can validate the
        window before attaching (PID, executable, class, thread). Ordered by
        ``EnumWindows`` Z-order.
        """
        import win32gui

        query = title.lower()
        candidates: list[dict] = []

        def _collect(handle: int, acc: list[dict]) -> None:
            if not win32gui.IsWindowVisible(handle):
                return
            info = WindowGeometry.window_info(handle)
            if query in info["title"].lower():
                acc.append(info)

        win32gui.EnumWindows(_collect, candidates)  # type: ignore[arg-type]
        return candidates

    @staticmethod
    def best_window_by_title(title: str) -> dict | None:
        """Best valid window matching ``title``, or None.

        Scores every title candidate (exact > prefix > substring > word
        overlap) and rejects windows that are not real target windows (PID 0,
        desktop/taskbar shells, system processes). This is what prevents
        attaching to the desktop or to an unrelated IDE window that merely
        contains the query as a substring.
        """
        best: tuple[float, dict] | None = None
        for candidate in WindowGeometry.find_windows_by_title(title):
            if not _is_valid_target_window(candidate):
                continue
            score = _title_score(title, candidate["title"])
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best is not None else None

    @staticmethod
    def find_window_by_title(title: str) -> int | None:
        """Backwards-compatible: best valid handle matching ``title``."""
        best = WindowGeometry.best_window_by_title(title)
        return best["handle"] if best else None

    @staticmethod
    def window_info(handle: int) -> dict:
        """Full Win32 metadata for a window handle."""
        import win32gui

        try:
            title = win32gui.GetWindowText(handle) or ""
        except Exception:
            title = ""
        try:
            class_name = win32gui.GetClassName(handle) or ""
        except Exception:
            class_name = ""
        pid = 0
        thread_id = 0
        try:
            thread_id, pid = win32gui.GetWindowThreadProcessId(handle)
        except Exception:
            pass
        return {
            "handle": handle,
            "title": title,
            "class_name": class_name,
            "process_id": pid,
            "thread_id": thread_id,
            "executable": _executable_for_pid(pid),
        }


class WindowCapture:
    """Captures only the client area of an attached window.

    All scene-analysis coordinates are relative to ``ClientArea.image``; the
    ``screen_offset`` bridges from the capture to the physical screen.
    """

    def __init__(self, grabber: ScreenGrabber | None = None, monitor: int = 0) -> None:
        self._grabber = grabber or ScreenGrabber()
        self._monitor = monitor
        self.handle: int | None = None
        self.title: str = ""
        self.last_capture: ClientArea | None = None

    def attach(self, handle: int, title: str = "") -> None:
        """Attach to a specific window handle."""
        self.handle = handle
        self.title = title or WindowGeometry.window_title(handle)

    def detach(self) -> None:
        self.handle = None
        self.title = ""
        self.last_capture = None

    @property
    def attached(self) -> bool:
        return self.handle is not None

    def client_area(self) -> ClientArea | None:
        """Capture the client area of the attached window (guard rails included).

        Returns None if capture is unavailable (window closed, zero-size, ...).
        """
        if self.handle is None:
            return None
        try:
            left, top, width, height = WindowGeometry.client_area(self.handle)
        except Exception as exc:
            logger.debug("client_area failed for handle {}: {}", self.handle, exc)
            return None
        if width <= 0 or height <= 0 or width > 20000 or height > 20000:
            return None
        try:
            image = self._grabber.grab_rect(left, top, width, height)
        except Exception as exc:
            logger.debug("grab failed: {}", exc)
            return None
        area = ClientArea(image=image, left=left, top=top, width=width, height=height)
        self.last_capture = area
        return area

    def capture_until_nonempty(self, timeout: float = 8.0, poll: float = 0.5) -> ClientArea | None:
        """Capture, retrying briefly while the client area is blank."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            area = self.client_area()
            if area is not None and area.image.shape[0] > 2 and area.image.shape[1] > 2:
                brightness = float(np.mean(area.image))
                if brightness > 1.0:  # not a pure-black capture
                    self.last_capture = area
                    return area
            time.sleep(poll)
        return self.last_capture

    def close(self) -> None:
        self.detach()
        self._grabber.close()


#: Window classes that are desktop shells, never real target windows.
_DESKTOP_SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "DV2ControlHost"}

#: Executable names that identify OS processes, never real target windows.
_SYSTEM_EXECUTABLES = {"System Idle Process", "System", "Registry", "wininit.exe", "winlogon.exe", "csrss.exe", "dwm.exe", "fontdrvhost.exe"}


def _executable_for_pid(pid: int) -> str:
    """Best-effort full executable path (or name) for a process id."""
    if pid <= 0:
        return ""
    try:
        import psutil

        proc = psutil.Process(pid)
        try:
            return proc.exe() or proc.name() or ""
        except Exception:
            return proc.name() or ""
    except Exception:
        return ""


def _title_score(query: str, title: str) -> float:
    """0..100 relevance of a window title for the query.

    Exact match is perfect; a prefix or full-token containment scores highly;
    a bare substring match scores lower; otherwise a word-overlap score.
    """
    q = (query or "").lower()
    t = (title or "").lower()
    if not q or not t:
        return 0.0
    if t == q:
        return 100.0
    if t.startswith(q):
        return 90.0
    if q in t:
        return 70.0
    q_words = {w for w in q.replace("(", " ").replace(")", " ").split() if len(w) >= 2}
    t_words = set(t.split())
    if q_words and t_words:
        overlap = len(q_words & t_words) / len(q_words)
        return overlap * 60.0
    return 0.0


def _is_valid_target_window(info: dict) -> bool:
    """A window is only attachable if it belongs to a real user process."""
    pid = info.get("process_id") or 0
    if pid <= 0:
        return False
    class_name = (info.get("class_name") or "").strip()
    if class_name in _DESKTOP_SHELL_CLASSES:
        return False
    exe = (info.get("executable") or "").strip().split("\\")[-1].lower()
    if not exe or exe in {s.lower() for s in _SYSTEM_EXECUTABLES}:
        return False
    return True
