"""The agent workflow loop.

Orchestrates the Observe -> Understand -> Reason -> Plan -> Execute -> Verify
loop for a stream of source records, target-agnostic (desktop window or web
page). It drives every stage explicitly, emits events and state transitions,
and refuses to continue past a record whose actions failed verification.

    while records remain:
        observe    -> target.observe()                       (VLM scene)
        understand -> SourceReader  -> SourceRecord
                     discover_fields -> editable fields
        reason     -> SemanticMapper -> MappingResult
        plan       -> ActionPlanner  -> FillPlan
        execute    -> ActionExecutor  -> verified results
        verify     -> every value-producing action verified (executor)
        next       -> poll until the source record changes, or timeout
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.act.executor import ActionExecutor
from atlas.act.models import ActionResult, ActionType
from atlas.core.events import EventType, get_event_bus
from atlas.core.logging import log_screenshot, logger
from atlas.core.metrics import Timer
from atlas.core.record_builder import RecordBuilder, RecordBuildResult
from atlas.core.states import AgentState, StateMachine
from atlas.mapping.mapper import MappingResult, SemanticMapper
from atlas.mapping.uia_map import UiaFieldMap, pair_source_pairs
from atlas.observe.screen_state import build_screen_state
from atlas.reason.planner import ActionPlanner, FillPlan
from atlas.target.base import TargetAdapter
from atlas.understanding.fields import discover_fields
from atlas.understanding.source import SourceReader, SourceRecord
from atlas.vision.models import BBox, ElementType, OcrText, SceneDescription, ScreenElement
from atlas.vision.scene import SceneAnalysis

ACTION_STATE = {
    ActionType.TYPE: AgentState.TYPING,
    ActionType.PASTE: AgentState.TYPING,
    ActionType.CLEAR: AgentState.TYPING,
    ActionType.SELECT: AgentState.TYPING,
    ActionType.TOGGLE: AgentState.TYPING,
    ActionType.CHOOSE_DATE: AgentState.TYPING,
    ActionType.TAB: AgentState.TYPING,
    ActionType.PRESS_ENTER: AgentState.TYPING,
    ActionType.PRESS_ESCAPE: AgentState.TYPING,
    ActionType.CLICK: AgentState.CLICKING,
    ActionType.DOUBLE_CLICK: AgentState.CLICKING,
    ActionType.RIGHT_CLICK: AgentState.CLICKING,
    ActionType.HOVER: AgentState.CLICKING,
    ActionType.MOVE_MOUSE: AgentState.CLICKING,
    ActionType.SUBMIT: AgentState.UPLOADING,
    ActionType.SCROLL: AgentState.SCROLLING,
    ActionType.WAIT: AgentState.WAITING,
    ActionType.VERIFY: AgentState.VERIFYING,
    ActionType.CAPTURE: AgentState.ANALYZING,
    ActionType.ANALYZE: AgentState.ANALYZING,
    ActionType.STOP: AgentState.STOPPED,
}


@dataclass
class RecordResult:
    """Outcome of processing one source record."""

    index: int
    record: SourceRecord
    mapping: MappingResult
    actions: list[ActionResult] = field(default_factory=list)
    skipped_fields: list[str] = field(default_factory=list)
    success: bool = False
    incomplete_fields: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "record": self.record.to_dict(),
            "mapping": self.mapping.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "skipped_fields": list(self.skipped_fields),
            "success": self.success,
            "incomplete_fields": list(self.incomplete_fields),
            "duration_ms": self.duration_ms,
            "message": self.message,
        }


@dataclass
class WorkflowSummary:
    """Aggregate of a whole workflow run."""

    records: list[RecordResult] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    stopped_reason: str = ""

    @property
    def completed(self) -> int:
        return sum(1 for r in self.records if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.records if not r.success)

    @property
    def total_duration(self) -> float:
        return self.finished - self.started if self.finished else 0.0

    @property
    def fields_filled(self) -> int:
        return sum(len(r.actions) for r in self.records)

    def to_dict(self) -> dict:
        return {
            "records": [r.to_dict() for r in self.records],
            "completed": self.completed,
            "failed": self.failed,
            "total_duration": self.total_duration,
            "stopped_reason": self.stopped_reason,
        }


class AgentLoop:
    """Runs the observe -> ... -> verify loop until records run out."""

    def __init__(
        self,
        target: TargetAdapter,
        source_reader: SourceReader,
        mapper: SemanticMapper,
        planner: ActionPlanner,
        executor: ActionExecutor,
        memory: Any | None = None,
        verify_after_action: bool = True,
        max_records: int = 0,
        next_record_timeout: float = 120.0,
        next_record_poll: float = 1.5,
        alias_learning: bool = False,
        scene_hook: Callable[[SceneDescription], SceneDescription] | None = None,
        on_record: Callable[[RecordResult], None] | None = None,
        field_map: UiaFieldMap | None = None,
        ocr_callback: Callable[[BBox], list[OcrText]] | None = None,
        debug_dir: str | Path | None = None,
        session_dir: str | Path | None = None,
        state_budget: float | dict[str, float] | None = None,
        record_builder: RecordBuilder | None = None,
        capture_callback: Callable[[Path], bool] | None = None,
    ) -> None:
        self._target = target
        self._source_reader = source_reader
        self._mapper = mapper
        self._planner = planner
        self._executor = executor
        self._memory = memory
        self._verify_after_action = verify_after_action
        self._max_records = max_records
        self._next_timeout = next_record_timeout
        self._next_poll = next_record_poll
        self._alias_learning = alias_learning
        self._scene_hook = scene_hook
        self._on_record = on_record
        self._field_map = field_map
        self._ocr_callback = ocr_callback
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._session_dir = Path(session_dir) if session_dir else (self._debug_dir / "session" if self._debug_dir else None)
        self._record_builder = record_builder or RecordBuilder()
        self._capture_callback = capture_callback
        self._state_budget = self._normalize_budget(state_budget)
        self._states = StateMachine()
        self._stop = False
        self._pause = False
        self._last_layout = ""
        self._state_entered: dict[AgentState, float] = {}
        self._state_warned: set[AgentState] = set()
        self._bus = get_event_bus()
        self._cached_analysis: SceneAnalysis | None = None
        self._last_signature = ""
        self._force_rebuild = False
        self._last_field: str | None = None
        self._planner_status = ""
        self._last_exception: str | None = None
        self._no_record_last_reason = ""

    # -- lifecycle -----------------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._states.state

    def stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def run(self) -> WorkflowSummary:
        summary = WorkflowSummary()
        self._states.reset()
        self._bus.publish(EventType.AGENT_STARTED)
        try:
            if not self._target.is_alive():
                raise RuntimeError("target is not attached")
            count = 0
            last_key: str | None = None
            while not self._stop:
                self._check_state_budget()
                if self._max_records and count >= self._max_records:
                    summary.stopped_reason = f"max_records reached ({count})"
                    break
                if self._pause:
                    time.sleep(0.2)
                    continue
                awaited = self._await_record(last_key)
                if awaited is None:
                    # Only reached when the loop was stopped; never on records==0.
                    break
                analysis, record = awaited
                result = self._run_record(analysis, record, count + 1)
                summary.records.append(result)
                count += 1
                if self._on_record is not None:
                    try:
                        self._on_record(result)
                    except Exception:
                        logger.exception("on_record callback failed")
                last_key = result.record.record_key
                self._bus.publish(
                    EventType.RECORD_COMPLETED if result.success else EventType.RECORD_FAILED,
                    result.to_dict(),
                )
                # After an upload the left panel changes to the next record:
                # force the screen model to rebuild on the next observation.
                self._force_rebuild = True
        except Exception as exc:
            logger.exception("workflow failed")
            summary.stopped_reason = str(exc)
            self._last_exception = str(exc)
        finally:
            summary.finished = time.time()
            if self._stop:
                summary.stopped_reason = summary.stopped_reason or "stopped by user"
            self._states.transition(AgentState.STOPPED)
            self._bus.publish(EventType.WORKFLOW_COMPLETE, summary.to_dict())
            self._bus.publish(EventType.AGENT_STOPPED, {"reason": summary.stopped_reason})
            self._dump_timeline(summary)
            self._dump_failure(summary)
            self._dump_focus_history()
        return summary

    # -- record processing ----------------------------------------------------

    def _run_record(self, analysis: SceneAnalysis, record: SourceRecord, index: int) -> RecordResult:
        with Timer() as timer:
            scene = analysis.scene
            self._set(AgentState.SCREEN_MODEL)
            self._set(AgentState.RECORD_EXTRACTION)
            self._bus.publish(EventType.SOURCE_READ, record.to_dict())
            self._bus.publish(EventType.RECORD_STARTED, {"index": index, "record": record.to_dict()})

            fields = discover_fields(scene)
            self._bus.publish(
                EventType.FIELD_DISCOVERED, {"count": len(fields), "fields": [f.to_dict() for f in fields]}
            )

            self._set(AgentState.FIELD_MAPPING)
            mapping = self._mapper.map(record, fields)
            self._bus.publish(EventType.MAPPING, mapping.to_dict())

            submit_id = self._find_submit(scene)
            self._set(AgentState.PLANNING)
            plan = self._planner.plan_fill(record, mapping, scene, submit_id)
            self._bus.publish(EventType.PLAN_CREATED, plan.to_dict())
            self._planner_status = f"{len(plan.actions)} actions planned"

        self._set(AgentState.THINKING)
        key = record.record_key or ""
        self._snapshot("before-fill", index, key)
        self._dump_record_debug(plan, [], index, record)
        results = self._execute_plan(plan, submit_id, index=index, record_key=key)
        if not self._all_ok(results):
            self._snapshot("failure", index, key)
        self._snapshot("after-fill", index, key)
        self._dump_record_debug(plan, results, index, record)
        self._bus.publish(
            EventType.SCREEN_STATE,
            build_screen_state(
                scene=scene,
                record=record,
                mapping=mapping,
                results=results,
                window_title=getattr(getattr(self._target, "info", None), "title", "") or "",
                record_index=index,
            ),
        )

        result = RecordResult(
            index=index,
            record=record,
            mapping=mapping,
            actions=results,
            success=self._all_ok(results),
            duration_ms=timer.elapsed * 1000.0,
        )
        self._learn_aliases(record, mapping, results)
        result.incomplete_fields = self._unmapped_required(mapping)
        result.skipped_fields = self._skipped_fields(results)
        if result.skipped_fields or result.incomplete_fields:
            result.message = (
                f"skipped {len(result.skipped_fields)} field(s), "
                f"{len(result.incomplete_fields)} required unmapped"
            )
        logger.info(
            "record {} ({}) -> {} in {:.1f}s",
            index,
            record.record_key or "?",
            "OK" if result.success else "FAILED",
            result.duration_ms / 1000.0,
        )
        return result

    def _execute_plan(
        self, plan: FillPlan, submit_element_id: str | None = None,
        index: int = 0, record_key: str = "",
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in plan.actions:
            if self._stop:
                break
            self._check_state_budget()
            is_upload = action.type == ActionType.SUBMIT or (
                action.type == ActionType.CLICK and action.field_id == submit_element_id
            )
            if is_upload:
                self._bus.publish(EventType.UPLOADING, action.to_dict())
            self._set(ACTION_STATE.get(action.type, AgentState.THINKING))
            self._last_field = action.field_id or action.reason
            self._bus.publish(EventType.ACTION_STARTED, action.to_dict())
            result = self._executor.execute(action)
            results.append(result)
            if result.ok and is_upload:
                self._bus.publish(EventType.UPLOAD_COMPLETED, result.to_dict())
                self._snapshot("after-upload", index, record_key)
            if not result.ok and action.type in {ActionType.SUBMIT, ActionType.CLICK}:
                logger.warning("submit/click failed; stopping record: {}", result.message)
                break
        return results

    # -- debug dumps ----------------------------------------------------------

    def _debug_path(self, name: str) -> Path | None:
        if self._debug_dir is None:
            return None
        path = self._debug_dir / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return path

    def _write_debug(self, name: str, data: Any) -> None:
        path = self._debug_path(name)
        if path is None:
            return
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.debug("failed to write {}: {}", path, exc)

    def _dump_record_debug(
        self, plan: FillPlan, results: list[ActionResult], index: int, record: SourceRecord
    ) -> None:
        if self._debug_dir is None:
            return
        self._write_session("record.json", {
            "index": index,
            "key": record.record_key,
            "title": record.title,
            "pairs": dict(record.pairs),
            "ordered_labels": record.ordered_labels,
        })
        self._write_debug("planner.json", plan.to_dict())
        self._write_debug("execution_plan.json", plan.to_dict())
        self._write_debug("execution.json", {
            "record_index": index,
            "actions": [r.to_dict() for r in results],
        })
        self._write_debug("verification.json", {
            "record_index": index,
            "results": [
                {
                    "field_id": r.action.field_id,
                    "action": r.action.type.value,
                    "expected": r.action.expected or r.action.value,
                    "verified": r.verified,
                    "success": r.success,
                    "evidence": r.verification_evidence or r.message,
                }
                for r in results
            ],
        })

    def _write_session(self, name: str, payload: dict) -> Path | None:
        if self._session_dir is None:
            return None
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_dir / name
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path
        except Exception as exc:
            logger.debug("session write failed: {}", exc)
            return None

    def _snapshot(self, context: str, index: int, key: str) -> Path | None:
        """Capture a screenshot for the given record lifecycle point.

        Writes ``debug/screenshots/{index}-{key}-{context}.png`` via the
        target-agnostic capture callback. Never raises.
        """
        if self._capture_callback is None or self._debug_dir is None:
            return None
        folder = self._debug_dir / "screenshots"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in (key or "record"))
        path = folder / f"{index:04d}-{safe_key}-{context}.png"
        try:
            if self._capture_callback(path):
                log_screenshot(path, context)
                return path
        except Exception as exc:
            logger.debug("screenshot {} failed: {}", context, exc)
        return None

    def _dump_timeline(self, summary: WorkflowSummary) -> None:
        if self._session_dir is None:
            return
        self._write_session("timeline.json", {
            "records": summary.completed,
            "failed": summary.failed,
            "stopped_reason": summary.stopped_reason,
            "duration_ms": summary.total_duration * 1000.0,
            "records_json": [r.to_dict() for r in summary.records],
        })

    def _dump_failure(self, summary: WorkflowSummary) -> None:
        """Write ``failure.json`` when the run did not fully succeed.

        A clean ``max_records`` stop is not a failure; an aborted run (no
        record, stopped early with pending records) is.
        """
        if self._debug_dir is None:
            return
        failed = summary.failed
        clean_stop = not summary.stopped_reason or summary.stopped_reason.startswith("max_records")
        if failed == 0 and clean_stop:
            return
        payload = {
            "failed_records": failed,
            "completed_records": summary.completed,
            "stopped_reason": summary.stopped_reason,
            "duration_ms": summary.total_duration * 1000.0,
            "records": [r.to_dict() for r in summary.records],
            "state": self._states.state.value,
            "current_field": self._last_field,
            "planner_status": self._planner_status,
            "last_exception": self._last_exception,
            "no_record_reason": self._no_record_last_reason,
        }
        self._write_debug("failure.json", payload)

    def _dump_focus_history(self) -> None:
        """Write ``focus_history.json`` from the RECOVERY event stream.

        Every focus-related pause/refocus decision published by the sandbox or
        the workflow is replayed here so focus-loss episodes are auditable
        offline without re-running the automation.
        """
        if self._debug_dir is None:
            return
        try:
            history = [
                e.to_dict()
                for e in self._bus.history(EventType.RECOVERY)
                if "focus" in str(e.data.get("reason", "")).lower()
            ]
            self._write_debug("focus_history.json", {
                "count": len(history),
                "events": history,
            })
        except Exception as exc:
            logger.debug("focus_history write failed: {}", exc)

    # -- helpers --------------------------------------------------------------

    def _await_record(self, previous_key: str | None) -> tuple[SceneAnalysis, SourceRecord] | None:
        """Wait for the next source record.

        Event-driven: the screen model is only rebuilt when the observed scene
        actually changes (app switch, upload click, left-panel change, scroll or
        focused-control change). On ``records == 0`` the loop never terminates:
        it shows "No valid record detected.", keeps retrying and waits for the
        next record. It only returns ``None`` when the loop is stopped.
        """
        self._set(AgentState.WATCHING)
        while not self._stop:
            self._check_state_budget()
            if self._next_timeout is not None and not self._stop:
                deadline = time.time() + self._next_timeout
                while not self._stop and time.time() < deadline:
                    analysis, changed = self._observe()
                    if analysis is not None and changed:
                        record = self._extract_record(analysis.scene)
                        if record is not None and self._accept_record(record, previous_key, analysis.scene):
                            self._bus.publish(EventType.NEXT_RECORD_DETECTED, {"key": record.record_key})
                            return analysis, record
                        if record is None:
                            # A no-record (e.g. loading) screen must not be
                            # cached by signature: force a fresh observation on
                            # the next poll so the following record is detected.
                            self._force_rebuild = True
                            time.sleep(self._next_poll)
                            continue
                        if self._same_record(record, previous_key):
                            self._bus.publish(EventType.NEXT_RECORD_WAITING, {"key": record.record_key})
                    time.sleep(self._next_poll)
                if not self._stop:
                    self._report_no_record()
            else:
                analysis, changed = self._observe()
                if analysis is not None and changed:
                    record = self._extract_record(analysis.scene)
                    if record is not None and self._accept_record(record, previous_key, analysis.scene):
                        self._bus.publish(EventType.NEXT_RECORD_DETECTED, {"key": record.record_key})
                        return analysis, record
                    if record is None:
                        # Never let a no-record screen stay cached (see above).
                        self._force_rebuild = True
                time.sleep(self._next_poll)
        self._set(AgentState.STOPPED)
        return None

    def _accept_record(self, record: SourceRecord, previous_key: str | None, scene: SceneDescription) -> bool:
        key = record.record_key
        if key and key != previous_key:
            return True
        if not record.pairs:
            return False
        if key is None and scene.layout_summary != (self._last_layout or ""):
            self._last_layout = scene.layout_summary
            return True
        return False

    def _same_record(self, record: SourceRecord, previous_key: str | None) -> bool:
        key = record.record_key
        if key and key == previous_key:
            return True
        return bool(key is None and self._last_layout and record.record_key is None)

    def reobserve_scene(self) -> SceneDescription | None:
        """Force a fresh observation and return the (field-map-merged) scene.

        Used by the executor after scrolling so bboxes stay accurate. Never
        raises: returns None on observation failure.
        """
        try:
            self._force_rebuild = True
            analysis, _ = self._observe()
            return analysis.scene if analysis is not None else None
        except Exception as exc:
            logger.debug("reobserve_scene failed: {}", exc)
            return None

    def _observe(self) -> tuple[SceneAnalysis | None, bool]:
        """Observe the target and rebuild the screen model only when changed.

        Returns ``(analysis, changed)``. ``changed`` is True when the screen
        model was rebuilt (first observation, app change, scroll, focus change,
        upload click or forced rebuild).
        """
        self._set(AgentState.OBSERVING)
        signature = self._target.signature() if hasattr(self._target, "signature") else ""
        if self._cached_analysis is None or self._force_rebuild or signature != self._last_signature:
            self._last_signature = signature
            self._force_rebuild = False
            analysis = self._target.observe()
            if analysis is None:
                return None, False
            if self._scene_hook is not None:
                try:
                    analysis.scene = self._scene_hook(analysis.scene)
                except Exception:
                    logger.exception("scene hook failed; using raw scene")
            self._merge_field_map(analysis.scene)
            self._bus.publish(EventType.OBSERVED, analysis.to_dict())
            self._cached_analysis = analysis
            self._write_debug("vision_output.json", {
                "provider": analysis.scene.provider,
                "window_title": analysis.scene.window_title,
                "layout_summary": analysis.scene.layout_summary,
                "screen_offset": list(analysis.scene.screen_offset),
                "sections": [s.to_dict() for s in analysis.scene.sections],
                "elements": [e.to_dict() for e in analysis.scene.elements],
            })
            return analysis, True
        return self._cached_analysis, False

    def _extract_record(self, scene: SceneDescription) -> SourceRecord | None:
        """Run the Record Extraction stage from UIA/OCR source pairs, falling
        back to the VLM scene reader. On failure writes ``debug/no_record.json``
        and ``debug/record_failure.json`` (Step 6).
        """
        self._set(AgentState.RECORD_EXTRACTION)
        pairs = self._collect_source_pairs(scene)
        result = self._record_builder.build(pairs, title=scene.window_title)
        if result.record is None:
            self._report_no_record(scene, result)
            self._write_record_failure(scene, result)
            return None
        self._bus.publish(EventType.SOURCE_READ, result.record.to_dict())
        return result.record

    def _write_record_failure(self, scene: SceneDescription, result: RecordBuildResult) -> None:
        """Write ``debug/record_failure.json`` with full diagnostics (Step 6)."""
        if self._debug_dir is None:
            return
        payload = {
            "reason": result.reason or "record could not be built",
            "detected_labels": list(result.labels),
            "detected_values": list(result.values),
            "missing_required": list(result.missing_required),
            "missing_controls": [],
            "missing_mappings": [],
            "window_title": scene.window_title,
            "layout_summary": scene.layout_summary,
        }
        # Report missing controls/mappings from the field map if available.
        if self._field_map is not None:
            payload["missing_controls"] = [
                n.name for n in (self._field_map.right_fields or [])
                if n.control_type in {"Edit", "ComboBox", "CheckBox", "RadioButton"}
            ]
            payload["missing_mappings"] = [
                m for m in (self._field_map.mappings or [])
            ]
        self._write_debug("record_failure.json", payload)

    @staticmethod
    def _clean_node_name(node: Any) -> str:
        return (node.name or node.automation_id or "").strip()

    def _collect_source_pairs(self, scene: SceneDescription) -> list[tuple[str, str]]:
        """Prefer exact UIA/OCR source pairs over the VLM scene pairs."""
        if self._field_map is not None and self._field_map.has_source:
            if self._ocr_callback is not None and self._field_map.left_rect is not None:
                left = self._field_map.left_rect
                try:
                    lines = self._ocr_callback(BBox(left.left, left.top, left.width, left.height))
                except Exception as exc:
                    logger.debug("source OCR failed: {}", exc)
                    lines = []
                self._write_debug("ocr_output.json", {
                    "region": {"left": left.left, "top": left.top, "width": left.width, "height": left.height},
                    "lines": [line.to_dict() for line in lines],
                })
                pairs = pair_source_pairs(lines, self._field_map.left_labels)
                if pairs:
                    return pairs
            labels = self._field_map.left_labels
            if labels:
                return [(self._clean_node_name(label), "") for label in labels]
        record = self._source_reader.read(scene)
        return [(label, record.pairs.get(label, "")) for label in record.ordered_labels]

    def _report_no_record(self, scene: SceneDescription | None = None, result: RecordBuildResult | None = None) -> None:
        """Surface a no-record condition and write ``debug/no_record.json``."""
        self._set(AgentState.WAITING)
        reason = (result.reason if result is not None else None) or "no valid record detected"
        if reason != self._no_record_last_reason:
            self._no_record_last_reason = reason
            logger.warning("no record: {}", reason)
            self._bus.publish(EventType.RECOVERY, {"reason": reason, "state": "record_extraction"})
        if self._debug_dir is not None:
            if result is not None and scene is not None:
                try:
                    self._record_builder.write_no_record(self._debug_dir / "no_record.json", result, scene=scene)
                except Exception as exc:
                    logger.debug("no_record write failed: {}", exc)
            else:
                self._write_debug("no_record.json", {"reason": reason})
        self._bus.publish(EventType.NO_RECORD, {"reason": reason})

    def _merge_field_map(self, scene: SceneDescription) -> None:
        """Synthesise exact UIA geometry onto the observed scene when a map exists.

        The UIA field map replaces the VLM's fuzzy editable fields with exact
        controls and injects OCR source pairs, so mapping/planning/execution use
        reliable geometry even when the VLM fails to identify the form.
        """
        if self._field_map is None or not self._field_map.has_form:
            return
        origin_x, origin_y = scene.screen_offset
        added: list[ScreenElement] = []
        seen_ids: set[str] = set()

        for node in self._field_map.right_fields:
            if node.rect is None:
                continue
            element_id = f"uia-{node.handle or node.automation_id or len(added)}"
            if element_id in seen_ids:
                continue
            seen_ids.add(element_id)
            box = BBox(
                node.rect.left - origin_x,
                node.rect.top - origin_y,
                node.rect.width,
                node.rect.height,
            )
            label = (node.name or node.automation_id or "").strip()
            added.append(ScreenElement(
                element_id=element_id,
                type=node.element_type,
                label=label,
                name=node.name or "",
                bbox=box,
                confidence=1.0,
                value=None,
                required=None,
                disabled=not node.enabled,
                section="form",
                options=list(node.options),
            ))

        if self._field_map.has_source and self._ocr_callback is not None and self._field_map.left_rect is not None:
            left = self._field_map.left_rect
            try:
                lines = self._ocr_callback(BBox(left.left, left.top, left.width, left.height))
            except Exception as exc:
                logger.debug("source OCR failed: {}", exc)
                lines = []
            for label, value in pair_source_pairs(lines, self._field_map.left_labels):
                element_id = f"uia-src-{len(seen_ids)}"
                seen_ids.add(element_id)
                added.append(ScreenElement(
                    element_id=element_id,
                    type=ElementType.LABEL,
                    label=label,
                    name=label,
                    bbox=None,
                    confidence=0.9,
                    value=value or None,
                    section="source",
                ))

        if self._field_map.upload_button is not None and self._field_map.upload_button.rect is not None:
            btn = self._field_map.upload_button
            box = BBox(btn.rect.left - origin_x, btn.rect.top - origin_y, btn.rect.width, btn.rect.height)
            element_id = f"uia-btn-{btn.handle or 'upload'}"
            if element_id not in seen_ids:
                seen_ids.add(element_id)
                added.append(ScreenElement(
                    element_id=element_id,
                    type=ElementType.BUTTON,
                    label=btn.name or "Upload",
                    name=btn.name or "",
                    bbox=box,
                    confidence=1.0,
                    section="actions",
                ))

        if not added:
            return
        kept = [e for e in scene.elements if not e.editable]
        merged: dict[str, ScreenElement] = {e.element_id: e for e in kept}
        for element in added:
            merged[element.element_id] = element
        scene.elements = list(merged.values())
        scene.layout_summary = scene.layout_summary or "uia-anchored"

    def _find_submit(self, scene: SceneDescription) -> str | None:
        submitish = (
            "upload", "submit", "save", "next", "ok", "apply", "continue",
            "done", "finish", "update", "register", "create", "add", "confirm",
        )
        buttons = [
            e
            for e in scene.elements
            if e.type.value in {"button", "submit"} or e.section == "actions"
        ]
        for label_token in submitish:
            for e in buttons:
                if label_token in (e.label or "").lower() and e.bbox is not None:
                    return e.element_id
        for e in buttons:
            if e.bbox is not None:
                return e.element_id
        for label_token in submitish:
            for e in buttons:
                if label_token in (e.label or "").lower():
                    return e.element_id
        for e in buttons:
            return e.element_id
        return None

    def _all_ok(self, results: list[ActionResult]) -> bool:
        if not results:
            return False
        return all(r.ok for r in results)

    @staticmethod
    def _skipped_fields(results: list[ActionResult]) -> list[str]:
        skipped = []
        for r in results:
            if r.success is False and r.action.field_id:
                skipped.append(r.action.field_id)
        return skipped

    @staticmethod
    def _unmapped_required(mapping: MappingResult) -> list[str]:
        return [f.label for f in mapping.unmatched_fields if f.element.required]

    def _learn_aliases(self, record: SourceRecord, mapping: MappingResult, results: list[ActionResult]) -> None:
        """Conservatively remember fuzzy mappings that verified successfully."""
        if not self._alias_learning or self._memory is None:
            return
        verified_ids = {r.action.field_id for r in results if r.ok}
        for m in mapping.mappings:
            if m.method in {"token", "containment", "fuzzy"} and m.confidence >= 0.9 and m.target_id in verified_ids:
                try:
                    self._memory.learn_alias(m.source_label, m.target_label)
                    self._mapper.aliases.learn(m.source_label, m.target_label)
                except Exception as exc:
                    logger.debug("alias learning skipped: {}", exc)

    def _set(self, state: AgentState) -> None:
        try:
            self._states.transition(state)
        except Exception:
            try:
                self._states.force(state)
            except Exception:
                pass
        self._state_entered[state] = time.time()
        self._state_warned.discard(state)
        self._bus.publish(EventType.STATE_CHANGED, {"state": self._states.state.value})

    # -- watchdog -------------------------------------------------------------

    @staticmethod
    def _normalize_budget(budget: float | dict[str, float] | None) -> dict[str, float]:
        if isinstance(budget, dict):
            return {k: float(v) for k, v in budget.items()}
        default = float(budget) if budget is not None else 10.0
        budgets = {state.value: default for state in AgentState}
        budgets[AgentState.WATCHING.value] = 60.0  # next-record timeout governs this
        budgets[AgentState.OBSERVING.value] = 45.0  # VLM analysis can be slow
        budgets[AgentState.THINKING.value] = 30.0
        budgets[AgentState.WAITING.value] = 60.0
        budgets[AgentState.WAITING_FOR_START_FIELD.value] = 0.0  # user-driven, never times out here
        return budgets

    def _check_state_budget(self) -> None:
        """Log + surface a state that has overrun its budget (never blocks)."""
        state = self._states.state
        budget = self._state_budget.get(state.value, 10.0)
        if budget <= 0:
            return
        entered = self._state_entered.get(state)
        if entered is None:
            return
        elapsed = time.time() - entered
        if elapsed > budget and state not in self._state_warned:
            self._state_warned.add(state)
            reason = f"state '{state.value}' overrun ({elapsed:.1f}s > {budget:.0f}s budget)"
            logger.warning("watchdog: {}", reason)
            self._bus.publish(EventType.RECOVERY, {"reason": reason, "state": state.value})


__all__ = ["AgentLoop", "RecordResult", "WorkflowSummary"]
