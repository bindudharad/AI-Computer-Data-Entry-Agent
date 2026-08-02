"""Semantic field mapping.

Maps source record pairs (labels + values from the data panel) onto the
destination form's editable fields using semantic similarity - never hardcoded
label rules. Aliases are resolved through a data-driven store that learns new
aliases over time (see ``atlas.memory``), seeded with common data-entry
synonyms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from atlas.core.logging import logger
from atlas.understanding.fields import EditableField
from atlas.understanding.source import SourceRecord
from atlas.vision.models import ElementType

#: Seed aliases (learned aliases augment this at runtime via memory).
DEFAULT_ALIASES: dict[str, str] = {
    # application identity
    "app no": "application number",
    "app number": "application number",
    "appl no": "application number",
    "application no": "application number",
    "application id": "application number",
    "application number": "application number",
    "app no.": "application number",
    # name
    "applicant name": "name",
    "full name": "name",
    "member name": "name",
    "applicant": "name",
    "first name": "name",
    # date of birth
    "dob": "date of birth",
    "birth date": "date of birth",
    "birthday": "date of birth",
    "date of birth": "date of birth",
    "d.o.b": "date of birth",
    "d.o.b.": "date of birth",
    # gender
    "sex": "gender",
    "applicant gender": "gender",
    "gender": "gender",
    # contact
    "mobile": "mobile number",
    "mobile no": "mobile number",
    "mobile number": "mobile number",
    "phone": "phone number",
    "phone no": "phone number",
    "phone number": "phone number",
    "telephone": "phone number",
    "cell": "mobile number",
    "cell phone": "mobile number",
    "contact": "contact number",
    "contact number": "contact number",
    "email": "email address",
    "email id": "email address",
    "email address": "email address",
    "e mail": "email address",
    "e-mail": "email address",
    # address
    "address": "address",
    "present address": "address",
    "current address": "address",
    "permanent address": "address",
    "house no": "house number",
    "house number": "house number",
    "street": "street",
    "street name": "street",
    "city": "city",
    "town": "city",
    "district": "district",
    "taluk": "taluk",
    "state": "state",
    "province": "state",
    "country": "country",
    "pincode": "pincode",
    "pin code": "pincode",
    "zip": "pincode",
    "zip code": "pincode",
    "postal code": "pincode",
    # identity documents
    "aadhaar": "aadhaar number",
    "aadhar": "aadhaar number",
    "aadhar number": "aadhaar number",
    "aadhaar number": "aadhaar number",
    "uid": "aadhaar number",
    "pan": "pan number",
    "pan no": "pan number",
    "pan number": "pan number",
    "passport": "passport number",
    "passport number": "passport number",
    # demographics
    "religion": "religion",
    "caste": "caste",
    "category": "category",
    "marital status": "marital status",
    "blood group": "blood group",
    "nationality": "nationality",
    "mother tongue": "mother tongue",
    "mothertoungue": "mother tongue",
    "occupation": "occupation",
    "profession": "occupation",
    "education": "education",
    "qualification": "education",
    "income": "income",
    "annual income": "income",
    "family income": "income",
    # relations
    "father": "father name",
    "father name": "father name",
    "father's name": "father name",
    "mother": "mother name",
    "mother name": "mother name",
    "mother's name": "mother name",
    "spouse": "spouse name",
    "spouse name": "spouse name",
    "husband": "husband name",
    "husband name": "husband name",
    "wife": "wife name",
    "wife name": "wife name",
    "guardian": "guardian name",
    "guardian name": "guardian name",
    # financial
    "bank name": "bank name",
    "account no": "account number",
    "account number": "account number",
    "ifsc": "ifsc code",
    "ifsc code": "ifsc code",
    # other common fields
    "remarks": "remarks",
    "notes": "remarks",
    "declaration": "declaration",
    "i agree": "declaration",
    "terms": "declaration",
    "agree": "declaration",
}

#: Field-type groups that must be compatible for a mapping to be accepted.
#: Source values are always text pairs, so every text-receiving target includes
#: ``UNKNOWN``; only boolean labels are constrained to boolean targets.
TYPE_COMPATIBILITY: dict[ElementType, tuple[ElementType, ...]] = {
    ElementType.TEXTBOX: (ElementType.TEXTBOX, ElementType.SEARCH_BOX, ElementType.UNKNOWN),
    ElementType.PASSWORD: (ElementType.PASSWORD, ElementType.TEXTBOX, ElementType.UNKNOWN),
    ElementType.TEXTAREA: (ElementType.TEXTAREA, ElementType.TEXTBOX, ElementType.UNKNOWN),
    ElementType.COMBOBOX: (ElementType.COMBOBOX, ElementType.LISTBOX, ElementType.UNKNOWN),
    ElementType.LISTBOX: (ElementType.LISTBOX, ElementType.COMBOBOX, ElementType.UNKNOWN),
    ElementType.CHECKBOX: (ElementType.CHECKBOX,),
    ElementType.RADIO: (ElementType.RADIO,),
    ElementType.FILE_UPLOAD: (ElementType.FILE_UPLOAD, ElementType.TEXTBOX, ElementType.UNKNOWN),
    ElementType.DATE_PICKER: (ElementType.DATE_PICKER, ElementType.CALENDAR, ElementType.TEXTBOX, ElementType.UNKNOWN),
    ElementType.CALENDAR: (ElementType.CALENDAR, ElementType.DATE_PICKER, ElementType.TEXTBOX, ElementType.UNKNOWN),
}

#: Labels that map to checkboxes/radios regardless of similarity score.
BOOLEAN_LABELS = ("declaration", "i agree", "terms and conditions", "agree", "consent")


@dataclass
class FieldMapping:
    """One source pair mapped to one destination field."""

    source_label: str
    source_value: str
    target: EditableField
    confidence: float
    method: str

    @property
    def target_id(self) -> str:
        return self.target.element_id

    @property
    def target_label(self) -> str:
        return self.target.label

    def to_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "source_value": self.source_value,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass
class MappingResult:
    """Outcome of a mapping pass."""

    mappings: list[FieldMapping] = field(default_factory=list)
    unmapped_source: list[str] = field(default_factory=list)
    unmatched_fields: list[EditableField] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.mappings:
            return 0.0
        return sum(m.confidence for m in self.mappings) / len(self.mappings)

    @property
    def coverage(self) -> float:
        total = len(self.mappings) + len(self.unmapped_source)
        return len(self.mappings) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "mappings": [m.to_dict() for m in self.mappings],
            "unmapped_source": list(self.unmapped_source),
            "unmatched_fields": [f.to_dict() for f in self.unmatched_fields],
            "score": self.score,
            "coverage": self.coverage,
        }


class AliasResolver:
    """Resolves a label to its canonical form via learned + default aliases."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases: dict[str, str] = {}
        self._seed(aliases)

    def _seed(self, extra: dict[str, str] | None) -> None:
        for key, value in DEFAULT_ALIASES.items():
            self._aliases[normalize_label(key)] = value
        if extra:
            for key, value in extra.items():
                self._aliases[normalize_label(key)] = normalize_label(value)

    def learn(self, variant: str, canonical: str) -> None:
        """Learn a new alias (runtime memory)."""
        key = normalize_label(variant)
        if key:
            self._aliases[key] = normalize_label(canonical)

    def resolve(self, label: str) -> str:
        normalized = normalize_label(label)
        if not normalized:
            return normalized
        seen: set[str] = set()
        current = normalized
        while current in self._aliases and current not in seen:
            seen.add(current)
            current = self._aliases[current]
        return current

    def as_dict(self) -> dict[str, str]:
        return dict(self._aliases)

    def known(self) -> set[str]:
        """Set of every known field concept (alias keys and canonical values)."""
        known: set[str] = set()
        for key, value in self._aliases.items():
            known.add(key)
            known.add(value)
        return known


class SemanticMapper:
    """Maps source pairs to destination fields by semantic similarity.

    Strategy (per source pair, greedy, one target per source):
      1. exact canonical match
      2. alias resolution match
      3. token fuzzy match (rapidfuzz) with length guard
      4. substring / containment match
    Confidence is scored from the match quality; spatial ordering breaks ties.
    """

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        threshold: float = 0.55,
        type_strict: bool = True,
    ) -> None:
        self._aliases = AliasResolver(aliases)
        self._threshold = threshold
        self._type_strict = type_strict

    @property
    def aliases(self) -> AliasResolver:
        return self._aliases

    def map(self, record: SourceRecord, fields: list[EditableField]) -> MappingResult:
        result = MappingResult()
        used: set[str] = set()
        by_canonical: dict[str, list[EditableField]] = {}
        for target_field in fields:
            by_canonical.setdefault(normalize_label(target_field.label), []).append(target_field)

        # Pass 1: exact / alias matches. Targets are claimed immediately so two
        # sources resolving to the same canonical never fight over one field.
        pending: list[tuple[str, str, str]] = []  # (source_label, canonical, value)
        for source_label in record.ordered_labels:
            value = record.pairs.get(source_label, "")
            source_canonical = self._aliases.resolve(source_label)
            claimed = False
            for target in by_canonical.get(source_canonical, []):
                if target.element_id in used:
                    continue
                if not self._type_ok(target, source_canonical):
                    continue
                confidence = self._exact_confidence(target, source_canonical)
                result.mappings.append(FieldMapping(
                    source_label, str(value or ""), target, confidence, "exact"
                ))
                used.add(target.element_id)
                claimed = True
                break
            if not claimed:
                pending.append((source_label, source_canonical, str(value or "")))

        # Pass 2: fuzzy match the remaining sources against the remaining fields.
        for source_label, source_canonical, value in pending:
            best: tuple[float, EditableField, str] | None = None
            for target in fields:
                if target.element_id in used:
                    continue
                if self._type_strict and not self._type_ok(target, source_canonical):
                    continue
                score, method = self._similarity(source_canonical, target.label)
                if best is None or score > best[0]:
                    best = (score, target, method)
            if best and best[0] >= self._threshold:
                score, target, method = best
                result.mappings.append(FieldMapping(
                    source_label, value, target, score, method
                ))
                used.add(target.element_id)

        mapped_source = {m.source_label for m in result.mappings}
        result.unmapped_source = [
            label for label in record.ordered_labels if label not in mapped_source
        ]
        result.unmatched_fields = [f for f in fields if f.element_id not in used]

        if result.mappings:
            logger.info(
                "mapped {} source fields (coverage={:.0%})",
                len(result.mappings),
                result.coverage,
            )
        return result

    # -- scoring helpers -----------------------------------------------------

    def _exact_confidence(self, target: EditableField, source_canonical: str) -> float:
        target_canonical = normalize_label(target.label)
        if target_canonical == source_canonical:
            return 0.98
        return 0.93

    def _type_ok(self, target: EditableField, source_canonical: str) -> bool:
        if self._is_boolean_label(source_canonical):
            return target.type in {ElementType.CHECKBOX, ElementType.RADIO, ElementType.UNKNOWN}
        if not self._type_strict:
            return True
        compatible = TYPE_COMPATIBILITY.get(target.type, ())
        if not compatible:
            return True
        return ElementType.UNKNOWN in compatible

    def _similarity(self, source_canonical: str, target_label: str) -> tuple[float, str]:
        target_canonical = self._aliases.resolve(target_label)
        if source_canonical == target_canonical:
            return 1.0, "exact"

        # Safety: never fuzzy-match two *distinct known* field concepts
        # (e.g. "application number" vs "pan number" both have "number").
        known = self._aliases.known()
        if source_canonical in known and target_canonical in known:
            return 0.0, "incompatible"

        # token sort ratio handles word reordering
        token = fuzz.token_sort_ratio(source_canonical, target_canonical) / 100.0
        if token >= 0.98:
            return token, "token"

        ratio = fuzz.ratio(source_canonical, target_canonical) / 100.0
        # containment: one label is part of the other
        if (
            len(source_canonical) >= 4
            and len(target_canonical) >= 4
            and (source_canonical in target_canonical or target_canonical in source_canonical)
        ):
            length_ratio = min(len(source_canonical), len(target_canonical)) / max(
                len(source_canonical), len(target_canonical)
            )
            contained = 0.5 + 0.4 * length_ratio
            if contained >= self._threshold:
                return contained, "containment"

        partial = fuzz.partial_ratio(source_canonical, target_canonical) / 100.0
        score = max(token * 0.6 + ratio * 0.4, partial * 0.85)
        if score < 0.75 and len(source_canonical) < 5 and len(target_canonical) < 5:
            # penalize very short labels (weak evidence)
            score *= 0.8
        method = "fuzzy"
        return score, method

    @staticmethod
    def _is_boolean_label(canonical: str) -> bool:
        return canonical in BOOLEAN_LABELS


def normalize_label(label: str) -> str:
    """Lowercase, strip punctuation/whitespace and collapse spacing."""
    import re

    text = str(label).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


__all__ = ["SemanticMapper", "FieldMapping", "MappingResult", "normalize_label", "DEFAULT_ALIASES", "AliasResolver"]
