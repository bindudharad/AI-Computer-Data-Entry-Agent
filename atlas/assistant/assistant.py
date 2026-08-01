"""Assistant: the wiring layer.

Builds and owns every component of the agent for one target:

    config -> vision provider -> scene analyzer
           -> memory -> mapper (seeded with learned aliases)
           -> planner / recovery / executor (target-specific controls)
           -> AgentLoop

The assistant is target-agnostic: ``attach_desktop`` and ``attach_web`` both
produce an attached :class:`~atlas.target.base.TargetAdapter`, and the same
loop runs against either.
"""

from __future__ import annotations

import json
from pathlib import Path

from atlas.act.clipboard import ClipboardEngine
from atlas.act.controls import ControlEngine, ControlInterface
from atlas.act.executor import ActionExecutor
from atlas.act.keyboard import HumanKeyboard
from atlas.act.mouse import HumanMouse, PyAutoGuiDriver
from atlas.act.verify import ClipboardVerifier, CompositeVerifier, TargetFieldVerifier, VisionVerifier
from atlas.config import AppConfig, load_config
from atlas.core.events import EventType, get_event_bus
from atlas.core.logging import logger
from atlas.core.record_builder import RecordBuilder
from atlas.core.states import AgentState
from atlas.mapping.mapper import SemanticMapper
from atlas.mapping.uia_map import UiaFieldMap, UiaFieldMapBuilder
from atlas.memory.store import MemoryStore
from atlas.observe.click_hook import MouseClickListener
from atlas.observe.uia import UiaBackend, UiaNode
from atlas.observe.window import WindowAttacher
from atlas.plugins.manager import PluginManager
from atlas.reason.planner import ActionPlanner
from atlas.reason.provider import LLMAdvisor, create_llm_provider
from atlas.reason.recovery import RecoveryPlanner
from atlas.target.base import TargetAdapter
from atlas.target.desktop import DesktopTarget
from atlas.target.web import WebTarget
from atlas.understanding.source import SourceReader
from atlas.vision.capture import ScreenGrabber, WindowCapture
from atlas.vision.models import BBox, OcrText
from atlas.vision.ocr import create_ocr_reader
from atlas.vision.providers import create_vision_provider
from atlas.vision.scene import SceneAnalyzer, WindowSceneSource
from atlas.workflow.loop import AgentLoop, WorkflowSummary


class Assistant:
    """Top-level agent facade.

    Example::

        assistant = Assistant()
        assistant.attach_desktop(title="Customer Entry")
        summary = assistant.run()
        assistant.close()
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()
        self._memory = MemoryStore(
            db_path=self._config.memory.db_path,
            alias_learning=self._config.memory.alias_learning,
        )
        self._bus = get_event_bus()
        self._target: TargetAdapter | None = None
        self._loop: AgentLoop | None = None
        self._executor: ActionExecutor | None = None
        self._providers: list[object] = []
        self._plugins = PluginManager(self)
        self._unsubscribe_plugins: Callable[[], None] | None = None

        ocr_reader = create_ocr_reader(self._config.ocr)
        self._ocr_reader = ocr_reader
        vision = create_vision_provider(self._config.vision, ocr_reader)
        self._providers.append(vision)
        self._analyzer = SceneAnalyzer(vision, cache_ttl=2.0)

        self._source_reader = SourceReader()
        aliases = self._memory.all_aliases()
        self._mapper = SemanticMapper(aliases=aliases if aliases else None)

        self._driver = PyAutoGuiDriver()
        self._mouse = HumanMouse(self._driver, self._config.mouse)
        self._keyboard = HumanKeyboard(self._driver, self._config.typing)
        self._grabber = ScreenGrabber()

        advisor = LLMAdvisor(create_llm_provider(self._config.reasoning), self._config.reasoning.confidence_threshold)
        self._providers.append(advisor)
        self._recovery = RecoveryPlanner(advisor=advisor)
        self._planner = ActionPlanner(verify_after_action=self._config.workflow.verify_after_action)

        self._setup_plugins()

    # -- properties ----------------------------------------------------------

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    @property
    def target(self) -> TargetAdapter | None:
        return self._target

    @property
    def mapper(self) -> SemanticMapper:
        return self._mapper

    @property
    def plugins(self) -> PluginManager:
        return self._plugins

    @property
    def state(self) -> str:
        if self._loop is not None:
            return self._loop.state.value
        return "idle"

    # -- attach --------------------------------------------------------------

    def attach_desktop(self, title: str | None = None) -> TargetAdapter:
        """Attach to a desktop window (foreground or matching ``title``)."""
        if self._target is not None:
            self.detach()
        capture = WindowCapture(grabber=self._grabber)
        source = WindowSceneSource(capture, self._analyzer)
        target = DesktopTarget(source, WindowAttacher(capture))
        target.attach(title)
        self._target = target
        self._build_executor(target)
        self._bus.publish(EventType.WINDOW_ATTACHED, target.info.to_dict() if target.info else {})
        logger.info("attached to desktop window '{}'", target.info.title if target.info else "")
        return target

    def attach_web(
        self,
        url: str | None = None,
        browser: str = "chromium",
        headless: bool = False,
    ) -> TargetAdapter:
        """Attach to a web page in a Playwright browser."""
        if self._target is not None:
            self.detach()
        target = WebTarget(
            analyzer=self._analyzer,
            browser_type=browser,
            headless=headless,
            viewport=(1280, 900),
        )
        target.attach(url)
        self._target = target
        self._build_executor(target)
        self._bus.publish(EventType.WINDOW_ATTACHED, target.info.to_dict() if target.info else {})
        logger.info("attached to web target '{}'", target.info.url if target.info else "")
        return target

    def detach(self) -> None:
        if self._target is not None:
            try:
                self._target.detach()
            except Exception as exc:
                logger.warning("detach failed: {}", exc)
            self._bus.publish(EventType.WINDOW_DETACHED)
            self._target = None

    # -- run -----------------------------------------------------------------

    def run(
        self,
        max_records: int = 0,
        out_dir: str | Path = "debug/auto",
    ) -> WorkflowSummary:
        """Run the workflow loop; returns the run summary.

        For a desktop target this runs the full pipeline automatically: it
        builds the UI tree straight from the attached window handle (no start
        click required), wires the record builder and session artifacts, then
        runs the loop. ``out_dir`` receives the field map, debug artifacts and
        per-record session files.
        """
        if self._target is None:
            raise RuntimeError("assistant is not attached - call attach_desktop() or attach_web() first")
        if self._executor is None:
            raise RuntimeError("executor not built - attachment failed")
        records = max_records or self._config.workflow.max_records

        kwargs: dict[str, object] = {}
        if isinstance(self._target, DesktopTarget) and self._target.info is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._publish_state(AgentState.BUILD_UI_TREE)
            handle = self._target.info.handle
            if handle is None:
                raise RuntimeError("attached window has no handle")
            field_map = self._build_field_map(handle, None, out)
            kwargs = {
                "field_map": field_map,
                "ocr_callback": self._read_region_ocr,
                "debug_dir": out,
                "session_dir": out / "session",
                "record_builder": RecordBuilder(
                    declared_fields=self._declared_fields(),
                    aliases=self._mapper.aliases.as_dict(),
                ),
            }

        self._loop = self._build_loop(records, **kwargs)
        return self._loop.run()

    def run_anchored(
        self,
        max_records: int = 0,
        out_dir: str | Path = "debug/mpf",
        start_timeout: float = 300.0,
    ) -> WorkflowSummary:
        """Desktop data entry anchored on the user's first click.

        Flow: attach -> WAITING_FOR_START_FIELD (user clicks the first editable
        field) -> FIELD_MAPPING (UIA field map written to ``field_map.json``)
        -> AgentLoop. The UIA field map gives the loop exact form geometry and
        OCR source pairs, so it never depends on the VLM to find the form.
        """
        if not isinstance(self._target, DesktopTarget) or self._target.info is None:
            raise RuntimeError("run_anchored requires an attached desktop target")
        if self._executor is None:
            raise RuntimeError("executor not built - attachment failed")
        handle = self._target.info.handle
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        listener = MouseClickListener()
        listener.start()
        try:
            start_control = self._capture_start_control(listener, out, start_timeout)
            field_map = self._build_field_map(handle, start_control, out)
        finally:
            listener.stop()

        records = max_records or self._config.workflow.max_records
        self._loop = self._build_loop(
            records,
            field_map=field_map,
            ocr_callback=self._read_region_ocr,
            debug_dir=out,
        )
        return self._loop.run()

    # -- anchored-flow internals ---------------------------------------------

    def _build_loop(self, max_records: int = 0, **kwargs: object) -> AgentLoop:
        return AgentLoop(
            target=self._target,
            source_reader=self._source_reader,
            mapper=self._mapper,
            planner=self._planner,
            executor=self._executor,
            memory=self._memory,
            verify_after_action=self._config.workflow.verify_after_action,
            max_records=max_records,
            next_record_timeout=self._config.workflow.next_record_timeout,
            next_record_poll=self._config.workflow.next_record_poll,
            alias_learning=self._config.memory.alias_learning,
            scene_hook=self._plugins.refine_scene,
            on_record=self._plugins.record,
            **kwargs,  # type: ignore[arg-type]
        )

    def _publish_state(self, state: AgentState) -> None:
        self._bus.publish(EventType.STATE_CHANGED, {"state": state.value})

    def _capture_start_control(
        self,
        listener: MouseClickListener,
        out: Path,
        timeout: float,
    ) -> UiaNode:
        """Wait for the user's click and resolve the StartControl anchor."""
        title = self._target.info.title if self._target.info else ""
        self._publish_state(AgentState.WAITING_FOR_START_FIELD)
        logger.info("waiting for you to click the first editable field in '{}'", title)
        logger.info("click a text/date/dropdown field in the RIGHT form panel to anchor the form")

        click = listener.wait_for_click(timeout)
        if click is None:
            raise RuntimeError(
                f"no click received within {timeout:.0f}s - click the first form field to begin"
            )
        x, y = click
        backend = UiaBackend.instance()
        node = backend.element_at(x, y)
        if node is None or not node.editable:
            node = backend.focused()
        if node is None or not node.editable:
            node = backend.element_at(x, y)
            raise RuntimeError(
                "the clicked control is not an editable field "
                f"(name={getattr(node, 'name', '?')!r}, type={getattr(node, 'control_type', '?')!r}). "
                "Click a text box, dropdown or date field, then re-run."
            )
        start = node.to_dict()
        start["clicked_at"] = {"x": x, "y": y}
        (out / "start_control.json").write_text(
            json.dumps(start, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("start control anchored: {!r} ({}) at {}", node.name, node.control_type, node.center)
        return node

    def _build_field_map(
        self,
        handle: int,
        start_control: UiaNode | None,
        out: Path,
    ) -> UiaFieldMap:
        backend = UiaBackend.instance()
        self._publish_state(AgentState.FIELD_MAPPING)
        declared = self._declared_fields()
        builder = UiaFieldMapBuilder(backend=backend, declared_fields=declared)
        field_map = builder.build(handle, start_control)
        field_map.save(out / "field_map.json")
        (out / "uia_tree.json").write_text(
            json.dumps(backend.dump_tree(handle), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._dump_panels(handle, field_map, out)
        if not field_map.has_form:
            logger.warning("uia field map found no editable form fields - falling back to vision")
        return field_map

    def _declared_fields(self) -> dict:
        path = Path("plugins/mpf/field_mapping.json")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("fields", {})
        except Exception as exc:
            logger.debug("declared fields unreadable: {}", exc)
            return {}

    def _dump_panels(self, handle: int, field_map: UiaFieldMap, out: Path) -> None:
        for name, rect in (("left_panel.png", field_map.left_rect), ("right_panel.png", field_map.right_rect)):
            if rect is None:
                continue
            try:
                image = self._grabber.grab_rect(rect.left, rect.top, rect.width, rect.height)
                from PIL import Image

                Image.fromarray(image).save(out / name)
            except Exception as exc:
                logger.debug("panel dump {} failed: {}", name, exc)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.stop()

    def pause(self) -> None:
        if self._loop is not None:
            self._loop.pause()

    def resume(self) -> None:
        if self._loop is not None:
            self._loop.resume()

    # -- internals -----------------------------------------------------------

    def _setup_plugins(self) -> None:
        if self._config.plugins.enabled:
            loaded = self._plugins.load_from(self._config.plugins.directory)
            if loaded:
                logger.info("loaded {} plugin file(s) from {}", loaded, self._config.plugins.directory)
        self._unsubscribe_plugins = self._bus.subscribe_all(self._plugins.event)

    def _build_executor(self, target: TargetAdapter) -> None:
        controls: ControlInterface = self._desktop_controls()
        verifier = self._desktop_verifier()
        if isinstance(target, WebTarget):
            controls = target.controls
            verifier = CompositeVerifier([
                TargetFieldVerifier(target.read_field_value),
            ])
        self._executor = ActionExecutor(
            mouse=self._mouse,
            keyboard=self._keyboard,
            controls=controls,
            verifier=verifier,
            recovery=self._recovery,
            verify_after_action=self._config.workflow.verify_after_action,
            max_retries=self._config.workflow.max_retries_per_action,
            retry_delay=self._config.workflow.retry_delay,
        )

    def _desktop_controls(self) -> ControlEngine:
        return ControlEngine(
            mouse=self._mouse,
            keyboard=self._keyboard,
            typing_config=self._config.typing,
            clipboard_use_long=self._config.typing.use_clipboard_for_long,
            clipboard_min_length=self._config.typing.clipboard_min_length,
        )

    def _desktop_verifier(self) -> CompositeVerifier:
        clipboard = ClipboardEngine(driver=self._driver)
        vision = VisionVerifier(self._read_region_ocr)

        def _desktop_read(field_id: str) -> str | None:
            return clipboard.read_focused()

        return CompositeVerifier([
            TargetFieldVerifier(_desktop_read),
            ClipboardVerifier(self._keyboard, clipboard),
            vision,
        ])

    def _read_region_ocr(self, bbox: BBox) -> list[OcrText]:
        image = self._grabber.grab_rect(bbox.x, bbox.y, bbox.width, bbox.height)
        return self._ocr_reader.read_image(image)

    def close(self) -> None:
        """Release every resource: target, providers, memory."""
        try:
            self.detach()
        finally:
            if self._unsubscribe_plugins is not None:
                try:
                    self._unsubscribe_plugins()
                except Exception:
                    pass
            try:
                self._plugins.close()
            except Exception:
                pass
            for provider in self._providers:
                try:
                    close = getattr(provider, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
            try:
                self._grabber.close()
            except Exception:
                pass
            try:
                self._memory.close()
            except Exception:
                pass
            self._mapper = self._executor = self._loop = None  # type: ignore[assignment]

    def __enter__(self) -> Assistant:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["Assistant"]
