"""Record Extraction stage.

Given raw label/value observations from the LEFT (source) panel, produces
exactly one ``Record`` (a :class:`~atlas.understanding.source.SourceRecord`)
or fails loudly. This stage is what guarantees the pipeline never silently
returns ``records: 0``:

* it matches labels with values,
* normalises field names (aliases + canonical forms),
* validates required fields against the declared field config,
* and, when no record can be produced, writes ``debug/no_record.json`` with
  the current screen, detected labels, detected values and the reason.

The builder is source-agnostic: it can be fed OCR pairs, UIA text-node pairs
or VLM scene pairs. The loop feeds it whichever channel produced data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from atlas.core.logging import logger
from atlas.mapping.mapper import normalize_label
from atlas.understanding.source import SourceRecord


@dataclass
class RecordBuildResult:
    """Outcome of one record-extraction attempt."""

    record: SourceRecord | None
    reason: str = ""
    labels: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.record is not None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "labels": list(self.labels),
            "values": list(self.values),
            "missing_required": list(self.missing_required),
            "record": self.record.to_dict() if self.record else None,
        }


class RecordBuilder:
    """Builds one ``SourceRecord`` from raw source-panel label/value pairs.

    Strategy:
    * labels are cleaned (trailing colons/space removed) and kept in visual
      order;
    * every non-empty value is kept; a label with no value still counts toward
      the record only if at least one value was found;
    * required fields (from ``declared_fields``, resolved through ``aliases``)
      that are absent are reported in ``missing_required``; the record is still
      produced so the planner can flag the incomplete fields.
    """

    def __init__(
        self,
        declared_fields: dict[str, dict] | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._declared = {normalize_label(k): v for k, v in (declared_fields or {}).items()}
        self._aliases = {normalize_label(k): normalize_label(v) for k, v in (aliases or {}).items()}
        self._required = [k for k, v in self._declared.items() if v.get("required")]

    def build(self, source_pairs: list[tuple[str, str]], title: str = "") -> RecordBuildResult:
        """Produce one record from ``(label, value)`` pairs, or fail loudly."""
        pairs: dict[str, str] = {}
        ordered: list[str] = []
        for raw_label, raw_value in source_pairs:
            label = _clean_label(raw_label)
            if not label or label in pairs:
                continue
            value = (raw_value or "").strip()
            pairs[label] = value
            ordered.append(label)

        labels = list(ordered)
        values = [pairs[label] for label in ordered]

        if not ordered:
            return RecordBuildResult(None, reason="no label/value pairs detected", labels=labels, values=values)

        if not any(v for v in values):
            return RecordBuildResult(
                None,
                reason="source panel has no values (all pairs empty)",
                labels=labels,
                values=values,
            )

        missing = self._missing_required(ordered)
        record = SourceRecord(pairs=pairs, ordered_labels=ordered, title=title)
        if missing:
            reason = f"required fields missing: {', '.join(missing)}"
        else:
            reason = "ok"
        return RecordBuildResult(record=record, reason=reason, labels=labels, values=values, missing_required=missing)

    def write_no_record(self, path: str | Path, result: RecordBuildResult | None = None, **extra: object) -> Path:
        """Write ``debug/no_record.json`` for a failed extraction. Never raises."""
        payload: dict = {
            "no_record": True,
            "reason": (result.reason if result else "") or "no valid record detected",
            "labels": list(result.labels if result else []),
            "values": list(result.values if result else []),
            "missing_required": list(result.missing_required if result else []),
        }
        payload.update(extra)
        out = Path(path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("failed to write {}: {}", out, exc)
        return out

    # -- helpers --------------------------------------------------------------

    def _canonical(self, label: str) -> str:
        key = normalize_label(label)
        return self._aliases.get(key, key)

    def _missing_required(self, labels: list[str]) -> list[str]:
        present = {self._canonical(label) for label in labels}
        return [req for req in self._required if req not in present]

    def as_dict(self) -> dict:
        return {
            "declared_fields": dict(self._declared),
            "required": list(self._required),
        }


def _clean_label(label: str) -> str:
    return re.sub(r"[:：\s]+$", "", (label or "")).strip()


__all__ = ["RecordBuilder", "RecordBuildResult"]
