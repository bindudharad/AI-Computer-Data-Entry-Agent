"""MPF plugin entry point.

Discovered by the PluginManager from the ``plugins/`` directory. When attached
to the assistant it:

1. Seeds the semantic mapper with MPF-specific aliases (from
   ``field_mapping.json``) so the LEFT source panel labels resolve to the
   RIGHT form's labels.
2. Refines every perceived scene: tags source/form sections and the
   "Upload Details" button by relative geometry and label semantics.
3. Logs a per-record summary and tracks upload completion.

No fixed screen coordinates are used anywhere - the app's layout is re-read
every cycle.
"""

from __future__ import annotations

from pathlib import Path

from atlas.core.logging import logger
from atlas.plugins import Plugin
from atlas.vision.models import SceneDescription

from plugins.mpf.mpf_detector import MpfDetector, load_field_mapping
from plugins.mpf.mpf_workflow import MpfWorkflow


class MpfPlugin(Plugin):
    """Data-entry operator behaviour for the MPF (Download and Upload Form)."""

    name = "mpf"

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).with_name("field_mapping.json")
        self._config = load_field_mapping(config_path)
        self._detector = MpfDetector(
            window_keywords=self._config.get("window_keywords", ["mpf"]),
            upload_labels=self._config.get("upload_button_labels", ["upload", "submit", "save"]),
            field_map=self._config.get("fields", {}),
        )
        self._workflow = MpfWorkflow(self._detector)
        self._assistant: object | None = None

    @property
    def detector(self) -> MpfDetector:
        return self._detector

    @property
    def workflow(self) -> MpfWorkflow:
        return self._workflow

    def on_register(self, assistant: object) -> None:
        self._assistant = assistant
        mapper = getattr(assistant, "mapper", None)
        aliases = self._config.get("aliases", {})
        if mapper is not None and aliases:
            for variant, canonical in aliases.items():
                try:
                    mapper.aliases.learn(variant, canonical)
                except Exception as exc:
                    logger.debug("mpf alias skipped {}: {}", variant, exc)
            logger.info("mpf: seeded {} aliases", len(aliases))

    def refine_scene(self, scene: SceneDescription) -> SceneDescription:
        if not self._detector.is_mpf_window(scene.window_title):
            return scene
        return self._detector.refine(scene)

    def on_event(self, event: object) -> None:
        self._workflow.on_event(event)

    def on_record(self, record: object) -> None:
        self._workflow.on_record(record)

    def close(self) -> None:
        self._assistant = None


def register_plugin() -> MpfPlugin:
    """Plugin factory used by the PluginManager."""
    return MpfPlugin()
