"""Target adapters: the thing the agent automates."""

from atlas.target.base import TargetAdapter, TargetInfo
from atlas.target.desktop import DesktopTarget
from atlas.target.web import WebTarget

__all__ = ["TargetAdapter", "TargetInfo", "DesktopTarget", "WebTarget"]
