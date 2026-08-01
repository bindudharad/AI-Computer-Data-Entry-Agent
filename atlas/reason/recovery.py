"""Recovery decisioning.

When an action fails or verification disagrees, the recovery planner decides the
corrective step. It escalates deterministically:

  1st failure  -> retry the same action
  2nd failure  -> refocus (click) the field, then retry
  persistent   -> re-analyse the screen (the layout may have changed)
  still stuck  -> skip the field (configurable) or stop the record

The LLM advisor is consulted at each escalation point and may override the
escalation with a better judgement (e.g. "a popup is covering the field").
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.act.models import Action, ActionResult, ActionType
from atlas.core.logging import logger
from atlas.reason.provider import LLMAdvisor
from atlas.vision.models import SceneDescription


@dataclass
class RecoveryDecision:
    """What the agent should do next after a failure."""

    action: ActionType
    reason: str
    field_id: str | None = None
    value: str | None = None
    retry: bool = False
    skip_field: bool = False
    stop_record: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "field_id": self.field_id,
            "value": self.value,
            "retry": self.retry,
            "skip_field": self.skip_field,
            "stop_record": self.stop_record,
        }


class RecoveryPlanner:
    """Decides corrective actions after failures."""

    def __init__(
        self,
        advisor: LLMAdvisor | None = None,
        max_retries: int = 3,
        max_refocus: int = 2,
        max_analyze: int = 2,
        skip_after_exhaust: bool = True,
    ) -> None:
        self._advisor = advisor
        self._max_retries = max_retries
        self._max_refocus = max_refocus
        self._max_analyze = max_analyze
        self._skip_after_exhaust = skip_after_exhaust
        self._failures: dict[str, int] = {}
        self._refocuses: dict[str, int] = {}
        self._analyzes = 0

    def reset(self) -> None:
        self._failures.clear()
        self._refocuses.clear()
        self._analyzes = 0

    def on_success(self, field_id: str) -> None:
        self._failures.pop(field_id, None)
        self._refocuses.pop(field_id, None)

    def decide(
        self,
        failed_action: Action,
        result: ActionResult,
        scene: SceneDescription | None = None,
    ) -> RecoveryDecision:
        field_id = failed_action.field_id or "?"
        failures = self._failures.get(field_id, 0) + 1
        self._failures[field_id] = failures

        # Consult the LLM advisor for a judgement on the failure.
        if self._advisor is not None and self._advisor.available:
            advice = self._advisor.consult({
                "situation": "action failed",
                "failed_action": failed_action.to_dict(),
                "result": result.to_dict(),
                "field_failure_count": failures,
                "screen": scene.to_dict() if scene else None,
            })
            if advice:
                decision = self._from_advice(advice, field_id, failed_action)
                if decision is not None:
                    logger.info("recovery advised by LLM: {}", decision.reason)
                    return decision

        if failures <= self._max_retries:
            return RecoveryDecision(
                action=ActionType.WAIT,
                reason=f"retry action on field '{field_id}' (failure {failures})",
                field_id=field_id,
                value=failed_action.value,
                retry=True,
            )

        refocuses = self._refocuses.get(field_id, 0)
        if refocuses < self._max_refocus:
            self._refocuses[field_id] = refocuses + 1
            self._failures[field_id] = 0
            return RecoveryDecision(
                action=ActionType.CLICK,
                reason=f"refocus field '{field_id}' and retry",
                field_id=field_id,
                retry=True,
            )

        if self._analyzes < self._max_analyze:
            self._analyzes += 1
            self._failures[field_id] = 0
            return RecoveryDecision(
                action=ActionType.ANALYZE,
                reason=f"re-analyse the screen - layout may have changed (field '{field_id}')",
                field_id=field_id,
                retry=True,
            )

        if self._skip_after_exhaust:
            logger.warning("giving up on field '{}' after {} failures", field_id, failures)
            return RecoveryDecision(
                action=ActionType.STOP,
                reason=f"skip field '{field_id}' after repeated failures",
                field_id=field_id,
                skip_field=True,
            )

        return RecoveryDecision(
            action=ActionType.STOP,
            reason=f"stop record: field '{field_id}' cannot be filled",
            field_id=field_id,
            stop_record=True,
        )

    def _from_advice(
        self, advice: dict, field_id: str, failed_action: Action
    ) -> RecoveryDecision | None:
        next_action = advice.get("next_action") or {}
        action_type = str(next_action.get("type", "")).lower()
        mapping = {
            "click": ActionType.CLICK,
            "retry": ActionType.CLICK,
            "scroll": ActionType.SCROLL,
            "wait": ActionType.WAIT,
            "analyze": ActionType.ANALYZE,
            "verify": ActionType.VERIFY,
        }
        if action_type in mapping:
            return RecoveryDecision(
                action=mapping[action_type],
                reason=str(next_action.get("reason", "LLM-recommended correction")),
                field_id=str(next_action.get("field_id") or field_id),
                value=next_action.get("value"),
                retry=True,
            )
        if advice.get("submit") is True:
            return None  # let the deterministic loop decide
        return None


__all__ = ["RecoveryPlanner", "RecoveryDecision"]
