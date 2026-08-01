"""Desktop window target.

Attaches to a single desktop window (Win32 / WinForms / WPF / Electron /
Java Swing / Qt) and observes only its client area. This is the default target
for native desktop applications.
"""

from __future__ import annotations

from atlas.observe.window import AttachError, WindowAttacher
from atlas.target.base import TargetAdapter, TargetInfo
from atlas.vision.scene import SceneAnalysis, WindowSceneSource


class DesktopTarget(TargetAdapter):
    """Automation target bound to one desktop window's client area."""

    name = "desktop"

    def __init__(self, scene_source: WindowSceneSource, attacher: WindowAttacher) -> None:
        self._source = scene_source
        self._attacher = attacher
        self._info: TargetInfo | None = None

    @property
    def scene_source(self) -> WindowSceneSource:
        return self._source

    @property
    def info(self) -> TargetInfo | None:
        return self._info

    def attach(self, hint: str | None = None) -> TargetInfo:
        if hint:
            target = self._attacher.attach_by_title(hint)
        else:
            target = self._attacher.attach_foreground()
        self._attacher.bring_to_front(target)
        self._info = self._target_info(target)
        return self._info

    def attach_by_click(self, timeout: float = 120.0) -> TargetInfo:
        """Attach by waiting for the user to click the target window (Step 8).

        This is the reliable attachment method for Electron/Chrome-based apps
        where title-based lookup finds ghost windows with pid=0.
        """
        target = self._attacher.attach_by_click(timeout)
        self._attacher.bring_to_front(target)
        self._info = self._target_info(target)
        return self._info

    def _target_info(self, target) -> TargetInfo:
        """Build a TargetInfo from a WindowTarget."""
        return TargetInfo(
            name=self.name,
            title=target.title,
            handle=target.handle,
            process_id=target.process_id,
            executable=target.executable,
            exe_path=target.exe_path,
            class_name=target.class_name,
            thread_id=target.thread_id,
        )

    def detach(self) -> None:
        self._source.capture.detach()
        self._info = None

    def observe(self) -> SceneAnalysis | None:
        if not self._source.attached:
            return None
        return self._source.observe()

    def is_alive(self) -> bool:
        return self._source.attached

    def read_field_value(self, field_id: str) -> str | None:
        # Desktop verification uses clipboard / vision read-back instead.
        return None

    def close(self) -> None:
        self.detach()
        self._source.close()


__all__ = ["DesktopTarget", "AttachError"]
