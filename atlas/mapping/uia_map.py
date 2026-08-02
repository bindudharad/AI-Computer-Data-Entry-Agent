"""UI Automation field mapping.

Builds a ``UiaFieldMap`` that bridges the agent's loop to the exact UI
Automation structure of a native form:

* ``start_control`` - the first editable control the user clicked (anchor).
* ``left_labels``   - static text controls in the LEFT (source) panel.
* ``right_fields``  - editable controls in the RIGHT (form) panel.
* ``upload_button`` - the button that submits/upload the form.
* ``mappings``      - LEFT label -> RIGHT field name pairs (UIA relationships).

The map is written to ``debug/mpf/field_map.json`` and used by the loop to
synthesise scene elements so SourceReader / field discovery / mapping / the
planner never depend on the VLM for exact form geometry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.core.logging import logger
from atlas.mapping.mapper import SemanticMapper, normalize_label
from atlas.observe.uia import UiaBackend, UiaNode
from atlas.vision.models import BBox, ElementType

UPLOAD_LABELS = ("upload", "submit", "save", "next", "ok", "apply", "create", "register", "done", "finish", "confirm")

#: Strong upload verbs outrank weak confirmation words like "OK"/"Done".
STRONG_UPLOAD_LABELS = {"upload", "submit", "create", "register", "finish", "confirm", "apply"}
WEAK_UPLOAD_LABELS = {"save", "next", "ok", "done", "add"}


@dataclass
class UiaFieldMap:
    """Snapshot of a native form's UIA structure plus its mapping."""

    start_control: UiaNode | None = None
    left_labels: list[UiaNode] = field(default_factory=list)
    right_fields: list[UiaNode] = field(default_factory=list)
    upload_button: UiaNode | None = None
    left_rect: BBox | None = None
    right_rect: BBox | None = None
    mappings: list[dict[str, str]] = field(default_factory=list)
    client_origin: tuple[int, int] = (0, 0)
    client_size: tuple[int, int] = (0, 0)

    @property
    def has_form(self) -> bool:
        return bool(self.right_fields)

    @property
    def has_source(self) -> bool:
        return bool(self.left_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_control": self.start_control.to_dict() if self.start_control else None,
            "left_labels": [n.to_dict() for n in self.left_labels],
            "right_fields": [n.to_dict() for n in self.right_fields],
            "upload_button": self.upload_button.to_dict() if self.upload_button else None,
            "left_rect": self.left_rect.to_dict() if self.left_rect else None,
            "right_rect": self.right_rect.to_dict() if self.right_rect else None,
            "mappings": list(self.mappings),
            "client_origin": list(self.client_origin),
            "client_size": list(self.client_size),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("field map saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> UiaFieldMap | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("field map {} unreadable: {}", path, exc)
            return None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UiaFieldMap:
        def _node(raw: dict[str, Any] | None) -> UiaNode | None:
            if not raw:
                return None
            return UiaNode(
                name=raw.get("name", ""),
                control_type=raw.get("control_type", ""),
                automation_id=raw.get("automation_id", ""),
                class_name=raw.get("class_name", ""),
                handle=raw.get("handle"),
                rect=_rect(raw.get("rect")),
                value=raw.get("value"),
                enabled=raw.get("enabled", True),
                visible=raw.get("visible", True),
                password=raw.get("password", False),
                options=list(raw.get("options") or []),
                type_override=_parse_element_type(raw.get("type_override")),
            )

        def _rect(raw: dict[str, Any] | list[int] | None) -> BBox | None:
            if isinstance(raw, dict):
                return BBox.from_dict(raw)
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                return BBox(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            return None

        return cls(
            start_control=_node(data.get("start_control")),
            left_labels=[_node(n) for n in data.get("left_labels") or []],
            right_fields=[_node(n) for n in data.get("right_fields") or []],
            upload_button=_node(data.get("upload_button")),
            left_rect=_rect(data.get("left_rect")),
            right_rect=_rect(data.get("right_rect")),
            mappings=list(data.get("mappings") or []),
            client_origin=tuple(data.get("client_origin") or (0, 0)),
            client_size=tuple(data.get("client_size") or (0, 0)),
        )


class UiaFieldMapBuilder:
    """Builds a :class:`UiaFieldMap` from a window handle + start control."""

    def __init__(
        self,
        backend: UiaBackend | None = None,
        declared_fields: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._backend = backend or UiaBackend.instance()
        self._declared_fields = declared_fields or {}

    def build(self, hwnd: int, start_control: UiaNode | None = None) -> UiaFieldMap:
        origin = self._backend.client_origin(hwnd)
        size = self._backend.client_size(hwnd)
        mid_x = origin[0] + size[0] // 2

        editable = self._backend.editable_fields(hwnd)
        # editable_fields already does the recursive fallback internally.
        right_fields = [n for n in editable if n.rect is not None and n.rect.center[0] >= mid_x]
        if not right_fields:
            right_fields = editable
        right_fields = [self._attach_declared(n) for n in right_fields]

        for field in right_fields:
            if field.rect is not None and size[1] and field.rect.bottom > origin[1] + size[1]:
                field = self._backend.scroll_into_view(field)

        text_nodes = self._backend.text_nodes(hwnd)
        left_labels = [
            n for n in text_nodes
            if n.rect is not None
            and n.rect.center[0] < mid_x
            and _is_meaningful_label(n.name)
        ]

        upload_button = self._pick_upload_button(self._backend.buttons(hwnd))

        left_rect = _union_rect([n.rect for n in left_labels if n.rect is not None])
        right_rect = _union_rect([n.rect for n in right_fields if n.rect is not None])

        mappings = build_hybrid_mappings(left_labels, right_fields, client_origin=origin)

        field_map = UiaFieldMap(
            start_control=start_control,
            left_labels=left_labels,
            right_fields=right_fields,
            upload_button=upload_button,
            left_rect=left_rect,
            right_rect=right_rect,
            mappings=mappings,
            client_origin=origin,
            client_size=size,
        )
        logger.info(
            "uia map built: {} left labels, {} right fields, upload={}",
            len(left_labels),
            len(right_fields),
            bool(upload_button),
        )
        return field_map

    def _attach_declared(self, node: UiaNode) -> UiaNode:
        """Attach declared widget options/type by normalized label, if any."""
        if not self._declared_fields:
            return node
        declared = self._declared_fields.get(normalize_label(node.name))
        if not declared:
            return node
        options = list(declared.get("options") or [])
        if options:
            node.options = options
        element_type = _parse_element_type(declared.get("type", ""))
        if element_type is not None:
            node.type_override = element_type
        return node

    @staticmethod
    def _pick_upload_button(buttons: list[UiaNode]) -> UiaNode | None:
        candidates: list[tuple[bool, float, UiaNode]] = []
        for button in buttons:
            label = normalize_label(button.name)
            if not label:
                continue
            strong = any(re.search(rf"\b{re.escape(t)}\b", label) for t in STRONG_UPLOAD_LABELS)
            score = 0.0
            for token in UPLOAD_LABELS:
                if re.search(rf"\b{re.escape(token)}\b", label):
                    score = max(score, len(token) / max(len(label), 1))
            if strong or score > 0:
                candidates.append((strong, score, button))
        if not candidates:
            return None
        # Strong verbs outrank weak ones; ties go to the bottom-most button
        # (form submits usually sit under the fields), then the right-most.
        candidates.sort(
            key=lambda pair: (
                -int(pair[0]),
                -pair[1],
                -(pair[2].rect.top if pair[2].rect else 10**9),
                -(pair[2].rect.left if pair[2].rect else 10**9),
            )
        )
        return candidates[0][2]


def pair_source_pairs(
    ocr_lines: list[Any],
    uia_labels: list[UiaNode] | None = None,
) -> list[tuple[str, str]]:
    """Pair OCR text lines from the source panel into (label, value) pairs.

    Strategy, in priority order:

    1. ``Label: value`` lines straight from OCR.
    2. Geometric pairing of UIA text nodes: the source panel renders each
       field as a label text node with its value as a sibling text node on the
       same row, so a row of two nodes is paired left -> right. This is what
       recovers real values even when OCR finds no colon lines.
    3. UIA static labels fill in any labels still missing (value falls back to
       the OCR remainder of the matching line, or an empty string).
    """
    pairs: dict[str, str] = {}
    ordered: list[str] = []
    ocr_texts = [getattr(line, "text", "") for line in ocr_lines if getattr(line, "text", "")]

    for text in ocr_texts:
        text = (text or "").strip()
        if not text:
            continue
        parts = re.split(r"[:：]\s*", text, maxsplit=1)
        if len(parts) == 2 and parts[0].strip():
            label = _clean_label(parts[0])
            if label and label not in pairs:
                pairs[label] = parts[1].strip()
                ordered.append(label)

    consumed: set[int] = set()
    if uia_labels:
        rows = _group_same_row([n for n in uia_labels if n.rect is not None])
        for row in rows:
            row.sort(key=lambda n: (n.rect.left if n.rect else 0, n.rect.top if n.rect else 0))
            i = 0
            while i < len(row) - 1:
                left_node, right_node = row[i], row[i + 1]
                gap = (right_node.rect.left - left_node.rect.right) if left_node.rect and right_node.rect else 0
                label = _clean_label(left_node.name)
                value = _clean_label(right_node.name)
                # Wide nodes are section headers (e.g. "Member Basic
                # Information"), not field labels; never pair them with a value.
                left_wide = left_node.rect is not None and left_node.rect.width > 120
                if label and value and label != value and label not in pairs and gap <= 170 and not left_wide:
                    pairs[label] = value
                    ordered.append(label)
                    consumed.add(id(right_node))
                    i += 2
                    continue
                i += 1

    for node in uia_labels or []:
        if id(node) in consumed:
            continue
        label = _clean_label(node.name)
        if not label or label in pairs:
            continue
        remainder = ""
        for text in ocr_texts:
            if text.lower().startswith(label.lower()):
                remainder = re.sub(r"^[^\w]*", "", text[len(label):]).lstrip(":： \t")
                break
        pairs[label] = remainder
        ordered.append(label)

    return [(label, pairs[label]) for label in ordered]


def _group_same_row(nodes: list[UiaNode], y_tolerance: int = 8) -> list[list[UiaNode]]:
    """Group nodes whose vertical centres fall within ``y_tolerance`` px."""
    rows: list[list[UiaNode]] = []
    for node in sorted(nodes, key=lambda n: (n.rect.center[1], n.rect.center[0])):
        if node.rect is None:
            continue
        placed = False
        for row in rows:
            row_y = sum(n.rect.center[1] for n in row) / len(row)
            if abs(node.rect.center[1] - row_y) <= y_tolerance:
                row.append(node)
                placed = True
                break
        if not placed:
            rows.append([node])
    return rows


def _build_name_mappings(left_labels: list[UiaNode], right_fields: list[UiaNode]) -> list[dict[str, str]]:
    """Map LEFT source labels onto RIGHT form fields by semantic similarity."""
    mapper = SemanticMapper()
    left_map: dict[str, str] = {}
    for node in left_labels:
        label = _clean_label(node.name)
        if label and label not in left_map:
            left_map[label] = ""
    right_names = []
    for node in right_fields:
        name = _clean_label(node.name)
        if name and name not in right_names:
            right_names.append(name)

    mappings: list[dict[str, str]] = []
    used: set[str] = set()
    for label in left_map:
        best: tuple[float, str] | None = None
        for name in right_names:
            if name in used:
                continue
            canonical_source = mapper.aliases.resolve(label)
            canonical_target = mapper.aliases.resolve(name)
            if canonical_source and canonical_source == canonical_target:
                best = (0.99, name)
                break
            score = _fuzzy(label, name)
            if best is None or score > best[0]:
                best = (score, name)
        if best and best[0] >= 0.55:
            mappings.append({"source": label, "target": best[1], "confidence": round(best[0], 3)})
            used.add(best[1])
    return mappings


def build_hybrid_mappings(
    left_labels: list[UiaNode],
    right_fields: list[UiaNode],
    client_origin: tuple[int, int] = (0, 0),
) -> list[dict[str, str]]:
    """Hybrid (semantic-first, geometry-fallback) LEFT->RIGHT mapping.

    The RIGHT form fields carry authoritative UIA names (e.g. "Full Name",
    "Gender", "State"). The LEFT labels are OCR text that may be noisy
    fragments. Therefore we map semantic-first: match each left label to the
    right field whose UIA name is semantically closest (using aliases and
    fuzzy matching). Geometry (same-row + nearest-right) is used only as a
    tiebreaker when semantic scores are close, and as a fallback for labels
    that have no semantic match.

    Never uses absolute screen coordinates - everything is relative to the
    client area origin, so the mapping holds regardless of where the window
    sits on screen.
    """
    ox, oy = client_origin
    mappings: list[dict[str, str]] = []
    used: set[int] = set()
    semantic_inputs: list[UiaNode] = []
    semantic_pool: list[UiaNode] = []

    # Build a semantic index of right fields: name + automation_id + aliases.
    mapper = SemanticMapper()
    right_index: list[tuple[str, UiaNode]] = []
    for field in right_fields:
        name = _clean_label(field.name)
        auto_id = _clean_label(field.automation_id)
        candidates = {n for n in (name, auto_id) if n}
        for c in candidates:
            right_index.append((c, field))

    for label in left_labels:
        source = _clean_label(label.name)
        if not source:
            continue
        # 1) Semantic match against right field names/automation ids.
        best_sem: tuple[float, UiaNode] | None = None
        for candidate, field in right_index:
            if id(field) in used:
                continue
            canonical_source = mapper.aliases.resolve(source)
            canonical_target = mapper.aliases.resolve(candidate)
            if canonical_source and canonical_target and canonical_source == canonical_target:
                best_sem = (0.99, field)
                break
            score = _fuzzy(source, candidate)
            if best_sem is None or score > best_sem[0]:
                best_sem = (score, field)

        # 2) Geometry match (same-row + nearest-right) as a tiebreaker/fallback.
        best_geo: tuple[float, UiaNode] | None = None
        if label.rect is not None and label.rect.width > 0 and label.rect.height > 0:
            lx, ly = label.rect.left - ox, label.rect.top - oy
            label_center_y = ly + label.rect.height / 2
            for field in right_fields:
                if field.rect is None or field.rect.width <= 0 or field.rect.height <= 0:
                    continue
                if id(field) in used:
                    continue
                fx, fy = field.rect.left - ox, field.rect.top - oy
                if fx <= lx:
                    continue  # must be to the RIGHT of the label
                field_center_y = fy + field.rect.height / 2
                row_gap = abs(field_center_y - label_center_y)
                dist = fx - lx
                row_penalty = 0.0 if row_gap <= max(12, field.rect.height // 2) else row_gap / 100.0
                score = dist + row_penalty
                if best_geo is None or score < best_geo[0]:
                    best_geo = (score, field)

        # Decide: prefer semantic if it's strong; otherwise use geometry.
        chosen: tuple[float, UiaNode, str] | None = None
        if best_sem is not None and best_sem[0] >= 0.55:
            chosen = (best_sem[0], best_sem[1], "semantic")
        elif best_geo is not None:
            chosen = (0.82, best_geo[1], "geometry")

        if chosen is None:
            semantic_inputs.append(label)
            continue
        conf, field, method = chosen
        used.add(id(field))
        target = _clean_label(field.name) or field.automation_id or ""
        if source and target:
            mappings.append({
                "source": source,
                "target": target,
                "confidence": round(conf, 3),
                "method": method,
            })
        else:
            semantic_inputs.append(label)

    # Remaining labels map by semantic similarity against the unused fields.
    semantic_pool = [f for f in right_fields if id(f) not in used]
    remaining = _build_name_mappings(semantic_inputs, semantic_pool)
    for entry in remaining:
        entry.setdefault("method", "semantic")
        mappings.append(entry)

    return mappings


def _fuzzy(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        token = fuzz.token_sort_ratio(a, b) / 100.0
        ratio = fuzz.ratio(a, b) / 100.0
        return max(token, ratio)
    except Exception:
        return 0.0


def _parse_element_type(name: Any) -> ElementType | None:
    try:
        return ElementType(str(name).strip().lower())
    except (ValueError, AttributeError):
        return None


def _clean_label(text: str) -> str:
    text = re.sub(r"[:：\s]+$", "", (text or "")).strip()
    return text


def _is_meaningful_label(text: str) -> bool:
    """Filter out OCR noise fragments (single letters, short fragments, etc.)."""
    label = _clean_label(text)
    if not label:
        return False
    # Reject very short fragments (1-2 chars) that are OCR noise.
    if len(label) < 3:
        return False
    # Reject fragments that are just a single letter repeated or punctuation.
    if re.fullmatch(r"[^a-zA-Z0-9]+", label):
        return False
    # Reject fragments that are just a single character repeated (e.g. "aaaa").
    if len(set(label.lower())) == 1:
        return False
    # Reject fragments that are clearly OCR noise like "ile", "ecord", "ools".
    # These are substrings of real words - require at least 2 words or a
    # recognizable word pattern.
    if re.fullmatch(r"[a-z]{2,4}", label.lower()) and not re.search(r"\s", label):
        # Short lowercase-only fragments are likely OCR noise unless they're
        # known field names.
        known_short = {
            "dob", "pan", "mbi", "rai", "mp", "f", "r", "t", "h",
            "app", "age", "sex", "city", "state", "name", "bank",
        }
        if label.lower() not in known_short:
            return False
    return True


def _union_rect(boxes: list[BBox]) -> BBox | None:
    boxes = [b for b in boxes if b is not None and b.width > 0 and b.height > 0]
    if not boxes:
        return None
    left = min(b.left for b in boxes)
    top = min(b.top for b in boxes)
    right = max(b.right for b in boxes)
    bottom = max(b.bottom for b in boxes)
    return BBox(left, top, max(0, right - left), max(0, bottom - top))


__all__ = ["UiaFieldMap", "UiaFieldMapBuilder", "pair_source_pairs", "build_hybrid_mappings", "UPLOAD_LABELS"]
