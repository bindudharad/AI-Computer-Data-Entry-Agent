"""Action executor.

Executes planned actions through the control engine, verifies every value-
producing action, and coordinates retries with the recovery planner. The
executor never continues blindly: a failed verification triggers corrective
actions (retry / refocus / re-analyse) up to the configured budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from atlas.act.controls import ControlInterface
from atlas.act.keyboard import HumanKeyboard
from atlas.act.models import VERIFYABLE_ACTIONS, Action, ActionResult, ActionType
from atlas.act.mouse import HumanMouse
from atlas.act.sandbox import ExecutionSandbox
from atlas.act.verify import CompositeVerifier
from atlas.core.logging import action_logger, logger, verification_logger
from atlas.core.metrics import Timer
from atlas.reason.recovery import RecoveryDecision, RecoveryPlanner
from atlas.vision.models import SceneDescription

if TYPE_CHECKING:
    from atlas.reason.planner import FillPlan

SceneProvider = Callable[[], SceneDescription | None]


class ActionExecutor:
    """Executes actions with verification and recovery."""

    def __init__(
        self,
        mouse: HumanMouse,
        keyboard: HumanKeyboard,
        controls: ControlInterface,
        verifier: CompositeVerifier,
        recovery: RecoveryPlanner,
        verify_after_action: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.8,
        scene_provider: SceneProvider | None = None,
        reobserve: Callable[[], SceneDescription | None] | None = None,
        sandbox: ExecutionSandbox | None = None,
        max_scroll_attempts: int = 6,
        scroll_amount: int = 3,
    ) -> None:
        self._mouse = mouse
        self._keyboard = keyboard
        self._controls = controls
        self._verifier = verifier
        self._recovery = recovery
        self._verify_after_action = verify_after_action
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._scene_provider = scene_provider
        self._reobserve = reobserve
        self._sandbox = sandbox
        self._max_scroll_attempts = max_scroll_attempts
        self._scroll_amount = scroll_amount

    def set_reobserve(self, reobserve: Callable[[], SceneDescription | None]) -> None:
        self._reobserve = reobserve

    # -- public API ----------------------------------------------------------

    def execute_plan(self, plan: FillPlan) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in plan.actions:
            result = self.execute(action)
            results.append(result)
            if result.action.type in {ActionType.STOP} and not result.success:
                break
        return results

    def execute(self, action: Action) -> ActionResult:
        with Timer() as timer:
            result = self._execute_with_recovery(action)
        result.duration_ms = timer.elapsed * 1000.0
        from atlas.core.events import EventType, get_event_bus

        failed = not (result.ok or result.verified)
        payload = result.to_dict()
        get_event_bus().publish(
            EventType.ACTION_FAILED if failed else EventType.ACTION_COMPLETED,
            payload,
        )
        action_logger.info(
            "action {}{}: {} ({}) in {:.0f}ms | retries={}",
            action.type.value,
            " FAILED" if failed else "",
            action.reason,
            result.message or "ok",
            timer.elapsed * 1000.0,
            result.retries,
        )
        if failed:
            logger.warning(
                "action failed: {} ({}): {}",
                action.type.value,
                action.reason,
                result.message,
            )
        return result

    # -- internals -----------------------------------------------------------

    def _execute_with_recovery(self, action: Action) -> ActionResult:
        # Uploads must never be re-clicked: a retry could double-submit the form.
        # Execute once, then stop the record if it fails so the loop can move on.
        if action.type == ActionType.SUBMIT:
            if self._sandbox is not None and self._sandbox.is_paused:
                self._sandbox.wait_until_resumed()
            if not self._assert_sandbox(action):
                return ActionResult(action=action, success=False, message="sandbox blocked submit")
            self._scroll_into_view(action)
            return self._do(action, 0)

        max_retries = action.max_retries if action.max_retries is not None else self._max_retries
        for attempt in range(max_retries + 1):
            # Check sandbox before each attempt.
            if self._sandbox is not None and self._sandbox.is_paused:
                logger.warning("sandbox paused - waiting for resume")
                self._sandbox.wait_until_resumed()
            if not self._assert_sandbox(action):
                result = ActionResult(action=action, success=False, message="sandbox blocked action")
                return result
            self._scroll_into_view(action)
            result = self._do(action, attempt)
            if action.type not in VERIFYABLE_ACTIONS or not self._verify_after_action:
                result.verified = True
                return result
            ok, evidence = self._verify(action)
            result.verified = ok
            result.verification_evidence = evidence
            self._publish_verification(action, ok, evidence, attempt)
            if ok:
                if action.field_id:
                    self._recovery.on_success(action.field_id)
                return result

            decision = self._recovery.decide(action, result, self._scene_provider() if self._scene_provider else None)
            if decision.skip_field or decision.stop_record:
                result.success = False
                result.message = decision.reason
                return result
            self._apply_correction(decision, action)
            time.sleep(self._retry_delay)

        result.success = False
        result.message = f"action exhausted {max_retries + 1} attempts"
        return result

    def _publish_verification(self, action: Action, ok: bool, evidence: str, attempt: int) -> None:
        from atlas.core.events import EventType, get_event_bus

        observed = self._observed_from_evidence(evidence)
        get_event_bus().publish(EventType.VERIFICATION, {
            "field_id": action.field_id,
            "label": action.reason,
            "expected": action.expected or action.value,
            "observed": observed,
            "ok": ok,
            "attempt": attempt,
            "evidence": evidence,
        })
        verification_logger.info(
            "verify {} [{}] expected={!r} observed={!r} -> {} (attempt {})",
            action.field_id or action.reason,
            action.type.value,
            action.expected or action.value,
            observed,
            "MATCH" if ok else "MISMATCH",
            attempt,
        )

    @staticmethod
    def _observed_from_evidence(evidence: str) -> str:
        """Best-effort extract of the observed value embedded in verifier evidence."""
        import re

        if not evidence:
            return ""
        match = re.search(r"(?:matched|contains|read|observed)[^\n]*?['\"]([^'\"]{1,120})['\"]", evidence, re.I)
        if match:
            return match.group(1)
        match = re.search(r"'([^']{1,120})'", evidence)
        return match.group(1) if match else evidence.strip()[:120]

    def _apply_correction(self, decision: RecoveryDecision, action: Action) -> None:
        logger.info("recovery: {}", decision.reason)
        from atlas.core.events import EventType, get_event_bus

        get_event_bus().publish(EventType.RECOVERY, decision.to_dict())
        if decision.action == ActionType.CLICK and action.bbox is not None:
            self._mouse.click(*action.bbox.center)
        elif decision.action == ActionType.SCROLL:
            self._controls.scroll("down", 3)
        elif decision.action == ActionType.WAIT:
            time.sleep(action.wait_seconds)
        elif decision.action == ActionType.ANALYZE and self._reobserve is not None:
            scene = self._reobserve()
            if scene is not None and action.field_id:
                element = scene.element(action.field_id)
                if element is not None and element.bbox is not None:
                    action.bbox = element.bbox.shifted(*scene.screen_offset)

    def _assert_sandbox(self, action: Action) -> bool:
        """Validate action against sandbox rules. Returns False if blocked."""
        if self._sandbox is None:
            return True
        # Keyboard actions require focus check.
        if action.type in {ActionType.TYPE, ActionType.CLEAR, ActionType.PASTE, ActionType.TAB,
                           ActionType.PRESS_ENTER, ActionType.PRESS_ESCAPE, ActionType.SUBMIT}:
            ok, reason = self._sandbox.validate_keyboard()
            if not ok:
                logger.warning("sandbox blocked keyboard: {}", reason)
                return False
        # Mouse actions require click validation.
        if action.type in {ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK, ActionType.HOVER}:
            if action.bbox is not None:
                x, y = action.bbox.center
                ok, reason = self._sandbox.validate_click(x, y)
                if not ok:
                    logger.warning("sandbox blocked click: {}", reason)
                    return False
        return True

    def _scroll_into_view(self, action: Action) -> None:
        """Bring an off-viewport field into view before acting on it.

        When the action targets a bbox outside the target client rect (e.g. a
        field below the fold), scroll toward it and re-observe to refresh the
        bbox. The scroll strategy escalates when one method makes no progress:

        - mouse wheel first (most natural),
        - keyboard PageUp/PageDown when the wheel scrolls a parent pane but
          not the nested region holding the field,
        - scroll-bar jump (End/Home) as a last resort.

        Bounded and never fatal: if it cannot be brought into view the action
        is left as-is so sandbox validation still blocks it safely.
        """
        if action.bbox is None:
            return
        if self._sandbox is None:
            return
        target = self._sandbox.validate_target()
        if target is None or not target.client_rect:
            return
        left, top, right, bottom = target.client_rect
        strategies: list[str] = ["wheel", "keys", "scrollbar"]
        previous_center: tuple[int, int] | None = None
        for _ in range(self._max_scroll_attempts):
            cx, cy = action.bbox.center
            if left <= cx <= right and top <= cy <= bottom:
                return
            # Escalate when the last attempt moved nothing.
            if previous_center == (cx, cy) and strategies:
                strategies.pop(0)
                if not strategies:
                    return
            previous_center = (cx, cy)

            if cy < top:
                direction = "up"
            elif cy > bottom:
                direction = "down"
            elif cx < left:
                direction = "up"
            else:
                direction = "down"

            strategy = strategies[0]
            if strategy == "keys":
                self._controls.scroll_by_keys(direction, self._scroll_amount)
            elif strategy == "scrollbar":
                self._controls.scroll_bar(direction, self._scroll_amount)
            else:
                self._controls.scroll(direction, self._scroll_amount)

            time.sleep(self._retry_delay)
            if self._reobserve is None:
                continue
            scene = self._reobserve()
            if scene is None or action.field_id is None:
                continue
            element = scene.element(action.field_id)
            if element is not None and element.bbox is not None:
                action.bbox = element.bbox.shifted(*scene.screen_offset)

    def _do(self, action: Action, attempt: int) -> ActionResult:
        try:
            return self._dispatch(action, attempt)
        except Exception as exc:
            logger.debug("action dispatch error: {}", exc)
            return ActionResult(action=action, success=False, message=str(exc), retries=attempt)

    def _dispatch(self, action: Action, attempt: int) -> ActionResult:
        bbox = action.bbox
        value = action.value

        if action.type == ActionType.MOVE_MOUSE and bbox:
            self._mouse.move_to(*bbox.center)
        elif action.type == ActionType.CLICK:
            if bbox is not None or action.field_id:
                outcome = self._controls.click_field(bbox, action.field_id)
                if not outcome.ok:
                    return ActionResult(action=action, success=False, message=outcome.evidence)
            else:
                self._controls.press_enter()  # default: activate focused control
        elif action.type == ActionType.DOUBLE_CLICK and bbox:
            self._mouse.double_click(*bbox.center)
        elif action.type == ActionType.RIGHT_CLICK and bbox:
            self._mouse.right_click(*bbox.center)
        elif action.type == ActionType.HOVER and bbox:
            self._mouse.hover(*bbox.center)
        elif action.type == ActionType.SCROLL:
            self._controls.scroll("down", action.scroll_amount)
        elif action.type == ActionType.TYPE and value is not None:
            self._controls.type_value(bbox, value, action.field_id)
        elif action.type == ActionType.CLEAR:
            self._controls.clear(bbox, action.field_id)
        elif action.type == ActionType.SELECT and value is not None:
            self._controls.select_option(bbox, value, action.options, action.field_id)
        elif action.type == ActionType.TOGGLE and value is not None:
            self._controls.toggle(bbox, value, action.field_id)
        elif action.type == ActionType.CHOOSE_DATE and value is not None:
            self._controls.choose_date(bbox, value, None, action.field_id)
        elif action.type == ActionType.TAB:
            self._controls.press_tab()
        elif action.type == ActionType.PRESS_ENTER:
            self._controls.press_enter()
        elif action.type == ActionType.PRESS_ESCAPE:
            self._controls.press_escape()
        elif action.type == ActionType.PASTE:
            self._controls.paste(value or "", action.field_id)
        elif action.type == ActionType.UPLOAD_FILE:
            if value is None:
                return ActionResult(action=action, success=False, message="no file path for upload")
            outcome = self._controls.upload_file(bbox, value, action.field_id)
            if not outcome.ok:
                return ActionResult(action=action, success=False, message=outcome.evidence)
        elif action.type == ActionType.WAIT:
            time.sleep(action.wait_seconds)
        elif action.type == ActionType.SUBMIT:
            if bbox is not None or action.field_id:
                outcome = self._controls.click_field(bbox, action.field_id)
                if not outcome.ok:
                    return ActionResult(action=action, success=False, message=outcome.evidence)
            else:
                self._controls.press_enter()
        elif action.type in {ActionType.CAPTURE, ActionType.ANALYZE}:
            return ActionResult(action=action, success=True, verified=True, message="handled by loop")
        elif action.type == ActionType.VERIFY:
            if action.value is not None:
                ok, evidence = self._verify(action)
                return ActionResult(
                    action=action, success=ok, verified=ok,
                    message=evidence, verification_evidence=evidence,
                )
            return ActionResult(action=action, success=True, verified=True, message="nothing to verify")
        elif action.type == ActionType.STOP:
            return ActionResult(action=action, success=False, verified=False, message="stop requested")
        else:
            return ActionResult(action=action, success=False, message=f"unsupported action {action.type.value}")

        return ActionResult(action=action, success=True, retries=attempt)

    def _verify(self, action: Action) -> tuple[bool, str]:
        if action.value is None:
            return True, "nothing to verify"
        return self._verifier.verify(action.bbox, action.value, action.field_id)


__all__ = ["ActionExecutor", "SceneProvider"]
