"""Target adapter interface.

A target is the application the agent is attached to and observes. Desktop
targets observe the window client area; web targets observe the page viewport.
Both expose the same contract so the workflow loop is target-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from atlas.vision.scene import SceneAnalysis


@dataclass
class TargetInfo:
    """Description of the attached target."""

    name: str
    title: str = ""
    url: str | None = None
    handle: int | None = None
    process_id: int = 0
    executable: str = ""
    exe_path: str = ""
    class_name: str = ""
    thread_id: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "url": self.url,
            "handle": self.handle,
            "process_id": self.process_id,
            "executable": self.executable,
            "exe_path": self.exe_path,
            "class_name": self.class_name,
            "thread_id": self.thread_id,
        }


class TargetAdapter(ABC):
    """Interface implemented by every automation target."""

    name = "abstract"

    @abstractmethod
    def attach(self, hint: str | None = None) -> TargetInfo:
        """Attach to the target. ``hint`` is target-specific (URL or window title)."""

    @abstractmethod
    def detach(self) -> None: ...

    @abstractmethod
    def observe(self) -> SceneAnalysis | None:
        """Capture and analyse the target's visible area."""

    @abstractmethod
    def is_alive(self) -> bool: ...

    @abstractmethod
    def read_field_value(self, field_id: str) -> str | None:
        """Best-effort read of a field's current value (None if unsupported)."""

    def close(self) -> None:
        pass
