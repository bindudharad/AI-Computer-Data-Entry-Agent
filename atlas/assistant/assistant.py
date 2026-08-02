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
import time
from collections.abc import Callable
from pathlib import Path

from atlas.act.clipboard import ClipboardEngine
from atlas.act.controls import ControlEngine, ControlInterface
from atlas.act.executor import ActionExecutor
from atlas.act.hotkeys import HotkeyManager
from atlas.act.keyboard import HumanKeyboard
from atlas.act.mouse import HumanMouse, PyAutoGuiDriver
from atlas.act.sandbox import ExecutionSandbox, SandboxConfig, TargetInfo
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
        self._sandbox = ExecutionSandbox(SandboxConfig())
        self._hotkeys = HotkeyManager()
        self._hotkeys.register("stop", self.stop)
        self._hotkeys.register("pause", self.pause)
        self._hotkeys.register("resume", self.resume)
        self._hotkeys.register("quit", self.close)
        self._hotkeys.start()

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
        self._attach_sandbox()
        self._build_executor(target)
        self._bus.publish(EventType.WINDOW_ATTACHED, target.info.to_dict() if target.info else {})
        logger.info("attached to desktop window '{}'", target.info.title if target.info else "")
        return target

    def attach_desktop_by_click(self, timeout: float = 120.0) -> TargetAdapter:
        """Attach to a desktop window by waiting for the user to click it.

        This is the reliable attachment method for Electron/Chrome-based apps
        (like MPF) where title-based lookup finds ghost windows with pid=0.
        The user clicks the real application window and we resolve it via
        ``WindowFromPoint`` + ``GetAncestor``.
        """
        if self._target is not None:
            self.detach()
        capture = WindowCapture(grabber=self._grabber)
        source = WindowSceneSource(capture, self._analyzer)
        target = DesktopTarget(source, WindowAttacher(capture))
        target.attach_by_click(timeout)
        self._target = target
        self._attach_sandbox()
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
        self._sandbox.detach()

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
            capture_callback=self._capture_screenshot,
        )
        return self._loop.run()

    def _capture_screenshot(self, path: str | Path) -> bool:
        """Save the target's current client area to ``path``. Returns True on success."""
        try:
            from atlas.vision.capture import WindowCapture

            info = self._target.info if self._target is not None else None
            if info is None or info.handle is None:
                return False
            capture = WindowCapture(self._grabber)
            capture.attach(info.handle, title=info.title)
            try:
                area = capture.capture_until_nonempty(timeout=3.0, poll=0.2)
                if area is None:
                    return False
                area.save(path)
                return True
            finally:
                capture.close()
        except Exception as exc:
            logger.debug("screenshot capture failed: {}", exc)
            return False

    # -- anchored-flow internals ---------------------------------------------

    def _build_loop(self, max_records: int = 0, **kwargs: object) -> AgentLoop:
        loop = AgentLoop(
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
        if self._executor is not None:
            self._executor.set_reobserve(loop.reobserve_scene)
        return loop

    def _publish_state(self, state: AgentState) -> None:
        self._bus.publish(EventType.STATE_CHANGED, {"state": state.value})

    def _capture_start_control(
        self,
        listener: MouseClickListener,
        out: Path,
        timeout: float,
    ) -> UiaNode:
        """Wait for the user's click and resolve the StartControl anchor.

        Uses both the low-level mouse hook AND foreground-window polling so the
        agent never exits just because the hook thread died silently (common on
        Electron/CEF apps). The first method to deliver a valid editable control wins.
        """
        title = self._target.info.title if self._target.info else ""
        handle = self._target.info.handle if self._target.info else None
        self._publish_state(AgentState.WAITING_FOR_START_FIELD)
        logger.info("waiting for you to click the first editable field in '{}'", title)
        logger.info("click a text/date/dropdown field in the RIGHT form panel to anchor the form")

        backend = UiaBackend.instance()
        deadline = time.time() + timeout
        poll = 0.1

        while time.time() < deadline:
            # Method 1: low-level mouse hook.
            click = listener.wait_for_click(poll)
            if click is not None:
                x, y = click
                try:
                    node = backend.element_at(x, y)
                    if self._is_valid_anchor(node, handle, out):
                        return node
                except Exception:
                    pass

            # Method 2: foreground-window polling. If the target window is now
            # foreground, try to find the focused editable control.
            if handle is not None:
                try:
                    import win32gui

                    fg = win32gui.GetForegroundWindow()
                    if fg == handle:
                        focused = backend.focused()
                        if self._is_valid_anchor(focused, handle, out):
                            logger.info("detected focus in target window while waiting for click")
                            return focused
                except Exception:
                    pass

        raise RuntimeError(
            f"no editable field selected within {timeout:.0f}s - click the first form field to begin"
        )

    def _is_valid_anchor(self, node: UiaNode | None, root_handle: int | None, out: Path) -> bool:
        """Validate that a clicked control is a valid MPF form anchor.
        
        In MPF (a Chromium-hosted app), the LEFT panel is a large List/Grid
        ("Items View") and the RIGHT panel contains the editable form fields.
        Only Edit/ComboBox/Calendar/Spinner controls in the right half of the
        window are valid anchors. List controls are rejected because they
        correspond to the left data panel or menu list.
        """
        if node is None or not node.editable:
            return False
        
        # Only form-field control types are valid anchors.
        # List/ListItem are rejected: in MPF these are the left-side data grid
        # ("Items View") or menu lists, not editable form fields.
        valid_types = {"Edit", "ComboBox", "Calendar", "Spinner"}
        if node.control_type not in valid_types:
            logger.info(
                "rejected anchor: control type {} - expected Edit/ComboBox/Calendar/Spinner",
                node.control_type,
            )
            return False
        
        # Must belong to MPF root window
        if root_handle is not None and node.handle is not None:
            try:
                import win32con
                control_root = win32gui.GetAncestor(node.handle, win32con.GA_ROOT)
                if control_root != root_handle:
                    return False
            except Exception:
                pass
        
        # Reject oversized controls (e.g. a List that fills the whole window).
        # Form fields are small; a control covering the whole client area is a panel/grid.
        if node.rect is not None and root_handle is not None:
            try:
                import win32gui
                client_rect = win32gui.GetClientRect(root_handle)
                client_w = max(1, client_rect[2])
                client_h = max(1, client_rect[3])
                ctrl_w = node.rect.width
                ctrl_h = node.rect.height
                if ctrl_w > client_w * 0.7 or ctrl_h > client_h * 0.7:
                    logger.info(
                        "rejected anchor: control {}x{} too large for form field (client {}x{})",
                        ctrl_w, ctrl_h, client_w, client_h,
                    )
                    return False
            except Exception:
                pass
        
        # Accept: clicked inside a small editable MPF form control
        logger.info(
            "Anchor accepted: {} ({}) | Root: {} | Rect: {}",
            node.name or node.automation_id,
            node.control_type,
            root_handle,
            node.rect,
        )
        self._save_start_control(node, node.center[0] if node.center else 0, node.center[1] if node.center else 0, out)
        return True

    def _save_start_control(self, node: UiaNode, x: int, y: int, out: Path) -> UiaNode:
        """Persist the start control and return it."""
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
        uia_tree = backend.dump_tree(handle)
        (out / "uia_tree.json").write_text(
            json.dumps(uia_tree, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (out / "window_tree.json").write_text(
            json.dumps(uia_tree, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        # Step 2: write the full UIA diagnostic set to debug/uia/.
        try:
            backend.dump_diagnostics(handle, out / "uia")
        except Exception as exc:
            logger.debug("uia diagnostics dump failed: {}", exc)
        self._dump_panels(handle, field_map, out)
        if not field_map.has_form:
            # Step 3: generate diagnostics instead of silently falling back.
            logger.error(
                "uia field map found no editable form fields - UI Automation exhausted. "
                "Diagnostics written to %s/uia/. Check editable_controls.json.",
                out,
            )
            self._write_field_map_failure(handle, field_map, out)
        return field_map

    def _write_field_map_failure(self, handle: int, field_map: UiaFieldMap, out: Path) -> None:
        """Write a diagnostic when the field map has no form fields (Step 3)."""
        backend = UiaBackend.instance()
        payload = {
            "reason": "no editable form controls found via UI Automation",
            "handle": handle,
            "left_labels": len(field_map.left_labels),
            "right_fields": len(field_map.right_fields),
            "upload_button": bool(field_map.upload_button),
            "inspectable_controls": len(backend.inspectable_nodes(handle)),
            "diagnostics_dir": str(out / "uia"),
        }
        try:
            (out / "field_map_failure.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("field_map_failure write failed: {}", exc)

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
        self._release_inputs()
        if self._loop is not None:
            self._loop.stop()

    def pause(self) -> None:
        self._release_inputs()
        if self._loop is not None:
            self._loop.pause()

    def resume(self) -> None:
        if self._loop is not None:
            self._loop.resume()

    def _release_inputs(self) -> None:
        """Release held mouse buttons / keyboard modifiers and clear locks.

        Called on pause and safe stop so a mid-gesture interruption never leaves
        a physical key or button pressed, and so the sandbox is not left paused.
        """
        try:
            self._mouse.release()
        except Exception:
            pass
        try:
            self._keyboard.release()
        except Exception:
            pass
        try:
            self._sandbox.resume()
        except Exception:
            pass

    # -- internals -----------------------------------------------------------

    def _setup_plugins(self) -> None:
        if self._config.plugins.enabled:
            loaded = self._plugins.load_from(self._config.plugins.directory)
            if loaded:
                logger.info("loaded {} plugin file(s) from {}", loaded, self._config.plugins.directory)
        self._unsubscribe_plugins = self._bus.subscribe_all(self._plugins.event)

    def _attach_sandbox(self) -> None:
        """Attach the execution sandbox to the current target.

        Uses the wrapper window handle (which is valid) as the sandbox owner.
        For Electron/Chrome apps, PID may be 0 - this is acceptable.
        """
        if self._target is None or self._target.info is None:
            return
        info = self._target.info
        root_handle = info.handle
        root_pid = info.process_id
        root_tid = info.thread_id
        client_rect = (0, 0, 0, 0)

        # Try to get PID from the window if it's 0.
        if root_pid == 0 and root_handle:
            try:
                import win32process
                _, root_pid = win32process.GetWindowThreadProcessId(root_handle)
            except Exception:
                pass

        # Compute client rect from the wrapper window.
        try:
            import win32gui
            rect = win32gui.GetClientRect(root_handle)
            origin = win32gui.ClientToScreen(root_handle, (0, 0))
            client_rect = (origin[0], origin[1], origin[0] + rect[2], origin[1] + rect[3])
        except Exception:
            pass

        logger.info(
            "sandbox attaching: handle=%s pid=%s class=%s",
            root_handle, root_pid, info.class_name,
        )

        target_info = TargetInfo(
            handle=root_handle,
            pid=root_pid,
            tid=root_tid,
            class_name=info.class_name,
            title=info.title,
            exe_name=info.executable,
            client_rect=client_rect,
        )
        self._sandbox.attach(target_info)
        logger.info("sandbox attached to hwnd={} pid={}", root_handle, root_pid)

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
            sandbox=self._sandbox,
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
            self._hotkeys.stop()
        except Exception:
            pass
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
