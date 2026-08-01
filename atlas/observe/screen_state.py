"""Screen-state snapshot for the live debug dashboard.

A compact, human-readable view of what the agent currently believes about the
target window: which source record it is working on, which form fields are
visible, which are filled (and verified), which are still missing, and where
the upload/submit action is headed.
"""

from __future__ import annotations

from typing import Any

from atlas.act.models import ActionResult, ActionType
from atlas.mapping.mapper import MappingResult
from atlas.understanding.source import SourceRecord
from atlas.vision.models import SceneDescription


def build_screen_state(
    scene: SceneDescription,
    record: SourceRecord,
    mapping: MappingResult,
    results: list[ActionResult],
    window_title: str = "",
    record_index: int = 0,
) -> dict[str, Any]:
    """Build a dashboard-ready snapshot from the current record cycle."""
    visible = []
    for e in scene.elements:
        if e.bbox is None:
            continue
        visible.append(
            {
                "label": e.label or "",
                "type": e.type.value,
                "editable": bool(e.editable),
                "section": getattr(e, "section", "") or "",
                "required": bool(e.required),
            }
        )

    completed: dict[str, str] = {}
    for r in results:
        if r.ok and r.action.field_id:
            completed[r.action.field_id] = r.action.value or ""

    # Determine current field being worked on (last non-verify action)
    current_field: str | None = None
    current_expected: str | None = None
    current_observed: str | None = None
    current_confidence: float = 0.0
    for r in reversed(results):
        if r.verified and r.action.field_id and r.action.value:
            current_field = r.action.reason or r.action.field_id
            current_expected = r.action.expected or r.action.value
            current_observed = r.verification_evidence or ""
            current_confidence = r.action.confidence
            break

    # Verification status
    last_verify = None
    for r in reversed(results):
        if r.action.type == ActionType.VERIFY:
            last_verify = r
            break
    verify_status = "pending"
    if last_verify is not None:
        verify_status = "verified" if last_verify.ok else "failing"

    missing = []
    for m in mapping.mappings:
        if m.target_id not in completed:
            missing.append({"field": m.target_label, "required": True})
    for f in mapping.unmatched_fields:
        if f.element.required:
            missing.append({"field": f.label, "required": True, "unmapped": True})

    upload_target: str | None = None
    for r in results:
        if r.action.type in {ActionType.SUBMIT, ActionType.CLICK}:
            upload_target = r.action.reason or r.action.field_id or "upload"
            break

    return {
        "window": window_title or scene.window_title or "",
        "record_index": record_index,
        "record_key": record.record_key or "",
        "record_pairs": [{"label": k, "value": v} for k, v in list(record.pairs.items())[:8]],
        "visible_fields": visible,
        "completed_fields": list(completed),
        "field_values": completed,
        "current_field": current_field,
        "current_expected": current_expected,
        "current_observed": current_observed,
        "current_confidence": current_confidence,
        "verify_status": verify_status,
        "missing_fields": missing,
        "unmapped_required": [f.label for f in mapping.unmatched_fields if f.element.required],
        "upload_target": upload_target,
    }


__all__ = ["build_screen_state"]
