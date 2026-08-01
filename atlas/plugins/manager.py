"""Plugin system.

A minimal hook-based plugin system: plugins are plain modules discovered from
the configured plugin directory (e.g. ``plugins/``), each exposing a
``register_plugin()`` function or a ``Plugin`` subclass. Plugins are notified
of agent events, per-record results and may refine the perceived scene before
the agent plans against it (used by the MPF plugin to tag source/form
sections and the upload button).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from abc import ABC
from pathlib import Path
from typing import Any

from atlas.core.events import Event
from atlas.core.logging import logger
from atlas.vision.models import SceneDescription


class Plugin(ABC):
    """Base class for agent plugins."""

    name = "plugin"

    def on_register(self, assistant: Any) -> None:
        """Called once when the plugin is attached to the assistant."""

    def on_event(self, event: Event) -> None:
        """Called for every event published on the bus."""

    def on_record(self, record: Any) -> None:
        """Called after each record completes."""

    def refine_scene(self, scene: SceneDescription) -> SceneDescription:
        """Optionally annotate/clean the perceived scene before planning.

        Plugins may tag elements with ``section`` (e.g. ``"source"`` /
        ``"form"``), mark the submit button, or drop noise elements. Return
        the (possibly same) scene.
        """
        return scene

    def close(self) -> None:
        """Release plugin resources."""


class PluginManager:
    """Registers plugins and forwards events / records / scenes to them."""

    def __init__(self, assistant: Any = None) -> None:
        self._assistant = assistant
        self._plugins: list[Plugin] = []

    @property
    def assistant(self) -> Any:
        return self._assistant

    @property
    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._plugins]

    def register(self, plugin: Plugin) -> None:
        if any(p.name == plugin.name for p in self._plugins):
            logger.debug("plugin '{}' already registered", plugin.name)
            return
        self._plugins.append(plugin)
        if self._assistant is not None:
            try:
                plugin.on_register(self._assistant)
            except Exception:
                logger.exception("plugin {} failed on_register", plugin.name)
        logger.info("plugin registered: {}", plugin.name)

    def load_from(self, directory: str | Path) -> int:
        """Load plugin modules from ``directory``.

        Discovers both flat plugin files (``<dir>/*.py``) and subpackage
        plugins (``<dir>/*/plugin.py``), matching the layout used by the MPF
        plugin.
        """
        path = Path(directory)
        if not path.is_dir():
            return 0
        found = 0
        candidates: list[Path] = list(path.glob("*.py")) + list(path.glob("*/plugin.py"))
        seen: set[Path] = set()
        for module_path in sorted(candidates):
            if module_path.name.startswith("_"):
                continue
            if module_path in seen:
                continue
            seen.add(module_path)
            if _register_from_file(self, module_path):
                found += 1
        return found

    def event(self, event: Event) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_event(event)
            except Exception:
                logger.exception("plugin {} failed on_event", plugin.name)

    def record(self, record: Any) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_record(record)
            except Exception:
                logger.exception("plugin {} failed on_record", plugin.name)

    def refine_scene(self, scene: SceneDescription) -> SceneDescription:
        """Run every plugin's scene hook in registration order."""
        for plugin in self._plugins:
            try:
                refined = plugin.refine_scene(scene)
                scene = refined if refined is not None else scene
            except Exception:
                logger.exception("plugin {} failed refine_scene", plugin.name)
        return scene

    def close(self) -> None:
        for plugin in self._plugins:
            try:
                plugin.close()
            except Exception:
                pass
        self._plugins.clear()


def _register_from_file(manager: PluginManager, module_path: Path) -> bool:
    """Execute a single plugin file and register the plugin(s) it defines."""
    stem = module_path.stem
    module_name = f"atlas_plugin_{stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            logger.warning("cannot load plugin module {}", module_path)
            return False
        module = importlib.util.module_from_spec(spec)
        parent = str(module_path.parent)
        sys.path.insert(0, parent)
        try:
            spec.loader.exec_module(module)
        finally:
            if parent in sys.path:
                sys.path.remove(parent)
    except Exception as exc:
        logger.warning("failed to import plugin {}: {}", stem, exc)
        return False

    factory = getattr(module, "register_plugin", None)
    plugins: list[Plugin] = []
    if callable(factory):
        try:
            candidate = factory()
        except Exception as exc:
            logger.warning("register_plugin() failed for {}: {}", stem, exc)
            candidate = None
        if isinstance(candidate, Plugin):
            plugins.append(candidate)
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate is Plugin:
            continue
        if isinstance(candidate, type) and issubclass(candidate, Plugin):
            try:
                plugins.append(candidate())
            except Exception as exc:
                logger.warning("cannot instantiate plugin {} from {}: {}", candidate.__name__, stem, exc)
    if not plugins:
        logger.warning("no Plugin subclass or register_plugin() found in {}", stem)
        return False
    for plugin in plugins:
        manager.register(plugin)
    return True


__all__ = ["Plugin", "PluginManager"]
