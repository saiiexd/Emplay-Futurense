"""Deterministic candidate resolution and addendum precedence.

The model never decides which candidate wins. This layer takes grounded
candidates, attaches document provenance from the registry, applies fixed
precedence rules, and selects exactly one value (or not_found) per field.

Precedence, highest first:

    addendum that amends the field   (higher addendum number wins)
    main solicitation document
    specification document
    portal / listing page
    affidavit or attachment form

An addendum only reaches the top tier when its evidence carries an explicit
amendment cue. Question-and-answer text that merely clarifies an existing
requirement is recorded as a clarification and leaves the value unchanged, which
is why a question in an addendum cannot silently rewrite the solicitation.

No value from the supplied corpus is encoded here. The cue patterns below are
generic procurement amendment language.
"""

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple

from src.schemas.candidates import GroundedCandidate
from src.schemas.enums import ConflictKind, DocType, FieldStatus
from src.schemas.field_catalog import FIELD_CATALOG
from src.schemas.models import (
    Clarification,
    Conflict,
    Contact,
    Evidence,
    ExtractedField,
    BidFields,
    BidRecord,
    DocumentRef,
    ExtractionSummary,
    Supersession,
)
from src.validation.normalize import normalize_value

# --- precedence tiers --------------------------------------------------------

TIER_ADDENDUM_AMENDMENT = 40
TIER_MAIN = 30
TIER_ADDENDUM_BASELINE = 25
TIER_SPECIFICATION = 20
TIER_PORTAL = 10
TIER_ATTACHMENT = 5

BASE_TIERS = {
    DocType.rfp_main: TIER_MAIN,
    DocType.specification: TIER_SPECIFICATION,
    DocType.portal_listing: TIER_PORTAL,
    DocType.affidavit: TIER_ATTACHMENT,
    # An addendum that does not amend the field ranks just below the main
    # solicitation. It still counts as evidence, but it cannot displace the
    # document it was issued against, so restating a requirement changes nothing.
    DocType.addendum: TIER_ADDENDUM_BASELINE,
}

#: Generic amendment language. A match means the document is changing something,
#: not merely discussing it.
AMENDMENT_CUES: Tuple[str, ...] = (
    r"\bnew\s+(due\s+date|closing\s+date|date|deadline|time)\b",
    r"\bis\s+hereby\s+(changed|revised|amended|replaced)\b",
    r"\bis\s+(changed|revised|amended|extended)\s+to\b",
    r"\bshall\s+be\s+replaced\s+(by|with)\b",
    r"\bamended\s+to\s+read\b",
    r"\bthe\s+following\s+replaces\b",
    r"\bextend(s|ed|ing)?\s+the\s+(due\s+date|closing\s+date|deadline)\b",
    r"\brevised\s+(due\s+date|closing\s+date|deadline|schedule)\b",
    r"\bdelete\b.{0,80}\binsert\b",
)

#: How far around the quote we look for an amendment cue.
CUE_WINDOW_CHARS = 300

_QA_ANSWER_RE = re.compile(r"(?im)^\s*answer\s*:")
_CUE_RES = tuple(re.compile(p, re.IGNORECASE) for p in AMENDMENT_CUES)

ROLE_AMENDMENT = "amendment"
ROLE_CLARIFICATION = "clarification"
ROLE_BASELINE = "baseline"


# --- data carried through resolution ----------------------------------------

@dataclass
class CandidateProvenance:
    doc_id: str
    doc_label: str
    filename: str
    document_type: DocType
    addendum_number: Optional[int]
    chunk_ids: List[str]
    quotes: List[str]


@dataclass
class ScoredCandidate:
    """A grounded candidate plus everything the precedence rules need."""

    grounded: GroundedCandidate
    provenance: CandidateProvenance
    tier: int
    role: str
    cue: Optional[str] = None

    @property
    def value(self) -> Any:
        return self.grounded.raw_value

    @property
    def sort_key(self) -> Tuple:
        # Deterministic ordering: strongest tier, then latest addendum, then
        # stable identifiers so repeated runs never reorder.
        return (
            -self.tier,
            -(self.provenance.addendum_number or 0),
            self.provenance.doc_label,
            self.grounded.candidate_id,
        )


@dataclass
class FieldResolution:
    """Audit record for one field."""

    field_name: str
    status: str
    value: Any = None
    rule: str = "no_candidates"
    chosen: Optional[ScoredCandidate] = None
    contributing: List[ScoredCandidate] = dc_field(default_factory=list)
    superseded: List[ScoredCandidate] = dc_field(default_factory=list)
    clarifications: List[ScoredCandidate] = dc_field(default_factory=list)
    conflicts: List[ScoredCandidate] = dc_field(default_factory=list)
    considered: List[ScoredCandidate] = dc_field(default_factory=list)


# --- helpers -----------------------------------------------------------------

def _comparable(value: Any) -> str:
    """Whitespace/case/punctuation-insensitive form used only for comparing
    candidates. The value itself is never rewritten."""
    if isinstance(value, dict):
        parts = [str(value.get(k) or "") for k in ("name", "email", "phone")]
        text = " ".join(parts)
    elif isinstance(value, list):
        text = " ".join(str(v) for v in value)
    else:
        text = str(value)
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _materially_different(left: Any, right: Any) -> bool:
    """Two values differ materially unless one is contained in the other."""
    a, b = _comparable(left), _comparable(right)
    if not a or not b:
        return a != b
    if a == b:
        return False
    return a not in b and b not in a


def classify_addendum_candidate(chunk_text: str, quote: str) -> Tuple[str, Optional[str]]:
    """Decide whether addendum evidence amends a field or merely clarifies it.

    Returns (role, matched_cue). An amendment cue near the quote means the
    addendum is changing the requirement. Otherwise, text that sits inside a
    question-and-answer passage is treated as a clarification.
    """
    position = chunk_text.find(quote)
    if position < 0:
        window = chunk_text
    else:
        start = max(0, position - CUE_WINDOW_CHARS)
        window = chunk_text[start: position + len(quote) + CUE_WINDOW_CHARS]

    for pattern in _CUE_RES:
        found = pattern.search(window)
        if found:
            return ROLE_AMENDMENT, found.group(0)

    if _QA_ANSWER_RE.search(window):
        return ROLE_CLARIFICATION, None

    # An addendum restating a requirement without a cue does not supersede.
    return ROLE_BASELINE, None


class CandidateResolver:
    """Applies the precedence rules to grounded candidates."""

    def __init__(self, registry, doc_label_for=None):
        self.registry = registry
        # ContextBuilder owns the D-label mapping; reuse it when available so
        # diagnostics speak the same language as the prompts.
        self._doc_label_for = doc_label_for

    # --- provenance ---------------------------------------------------------

    def _provenance(self, grounded: GroundedCandidate) -> Optional[CandidateProvenance]:
        if not grounded.chunk_ids:
            return None
        chunk = self.registry.chunk_metadata.get(grounded.chunk_ids[0])
        if not chunk:
            return None
        doc_id = chunk.get("doc_id")
        doc_ref = self.registry.get_document_ref(doc_id)
        if not doc_ref:
            return None
        label = self._doc_label_for(doc_id) if self._doc_label_for else doc_id
        return CandidateProvenance(
            doc_id=doc_id,
            doc_label=label,
            filename=doc_ref.filename,
            document_type=doc_ref.document_type,
            addendum_number=doc_ref.addendum_number,
            chunk_ids=list(grounded.chunk_ids),
            quotes=list(grounded.excerpts),
        )

    def score(self, grounded: GroundedCandidate) -> Optional[ScoredCandidate]:
        """Attach provenance and a precedence tier to one grounded candidate."""
        provenance = self._provenance(grounded)
        if provenance is None:
            return None

        role = ROLE_BASELINE
        cue = None
        tier = BASE_TIERS.get(provenance.document_type, TIER_ATTACHMENT)

        if provenance.document_type == DocType.addendum:
            chunk = self.registry.chunk_metadata.get(provenance.chunk_ids[0], {})
            quote = provenance.quotes[0] if provenance.quotes else ""
            role, cue = classify_addendum_candidate(chunk.get("text", ""), quote)
            if role == ROLE_AMENDMENT:
                tier = TIER_ADDENDUM_AMENDMENT

        return ScoredCandidate(grounded=grounded, provenance=provenance, tier=tier, role=role, cue=cue)

    # --- per-field resolution ------------------------------------------------

    def resolve_field(self, field_name: str, grounded_candidates: List[GroundedCandidate]) -> FieldResolution:
        spec = FIELD_CATALOG[field_name]

        scored: List[ScoredCandidate] = []
        for grounded in grounded_candidates:
            if not grounded.is_valid:
                continue
            item = self.score(grounded)
            if item is None:
                continue
            # Grounding already enforced eligibility; re-check so a resolver
            # bug can never promote an ineligible document.
            if item.provenance.document_type not in spec.eligible_document_types:
                continue
            scored.append(item)

        resolution = FieldResolution(field_name=field_name, status=FieldStatus.not_found.value)
        if not scored:
            return resolution

        scored.sort(key=lambda c: c.sort_key)
        resolution.considered = list(scored)

        clarifications = [c for c in scored if c.role == ROLE_CLARIFICATION]
        deciding = [c for c in scored if c.role != ROLE_CLARIFICATION]
        resolution.clarifications = clarifications

        if not deciding:
            # Only clarification text exists: nothing states a value outright.
            resolution.rule = "clarification_only"
            return resolution

        top_tier = deciding[0].tier
        winners = [c for c in deciding if c.tier == top_tier]

        # Fields the catalog aggregates collect evidence rather than choose it.
        # A required-documents list, for example, is legitimately spread across
        # the solicitation and the forms themselves, so every eligible document
        # contributes. The exception is an amending addendum: when one restates
        # the field, its list replaces the earlier one instead of extending it.
        if spec.aggregation.value == "concat":
            if top_tier == TIER_ADDENDUM_AMENDMENT:
                contributors = winners
                resolution.superseded = [c for c in deciding if c.tier < top_tier]
                resolution.rule = "addendum_amendment_aggregated"
            else:
                contributors = deciding
                resolution.rule = "aggregated_all_tiers"

            resolution.status = FieldStatus.found.value
            resolution.chosen = contributors[0]
            resolution.contributing = contributors
            resolution.value = self._aggregate([c.value for c in contributors])
            return resolution

        chosen = winners[0]
        resolution.chosen = chosen
        resolution.contributing = [chosen]
        resolution.value = chosen.value
        resolution.status = FieldStatus.found.value

        losers = [c for c in deciding if c is not chosen]
        resolution.superseded = [c for c in losers if _materially_different(c.value, chosen.value)]

        if chosen.tier == TIER_ADDENDUM_AMENDMENT:
            resolution.rule = "addendum_amendment"
        elif chosen.provenance.document_type == DocType.portal_listing:
            resolution.rule = "portal_only"
        else:
            resolution.rule = "highest_authority"

        # Same-tier disagreement is reported, not silently picked over.
        rivals = [c for c in winners[1:] if _materially_different(c.value, chosen.value)]
        if rivals:
            resolution.status = FieldStatus.conflicting.value
            resolution.conflicts = rivals
            resolution.rule += "_with_conflict"

        return resolution

    @staticmethod
    def _aggregate(values: List[Any]) -> Any:
        """Merge aggregated fields without inventing content."""
        items: List[Any] = []
        for value in values:
            if isinstance(value, list):
                items.extend(value)
            else:
                items.append(value)

        merged: List[Any] = []
        seen = set()
        for item in items:
            key = _comparable(item)
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    # --- whole bid -----------------------------------------------------------

    def resolve_bid(self, grounded_by_field: Dict[str, List[GroundedCandidate]]) -> Dict[str, FieldResolution]:
        """Resolve all 20 catalog fields, in catalog (assignment) order."""
        return {
            name: self.resolve_field(name, grounded_by_field.get(name, []))
            for name in FIELD_CATALOG
        }


# --- final assembly ----------------------------------------------------------

def _as_contacts(value: Any) -> List[Contact]:
    raw = value if isinstance(value, list) else [value]
    contacts: List[Contact] = []
    for entry in raw:
        if isinstance(entry, dict):
            contacts.append(Contact(
                name=entry.get("name"), email=entry.get("email"), phone=entry.get("phone")
            ))
    return contacts


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_string(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _evidence_for(resolution: FieldResolution, registry) -> Evidence:
    chunk_ids: List[str] = []
    excerpts: List[str] = []
    refs: List[DocumentRef] = []
    for candidate in resolution.contributing:
        chunk_ids.extend(candidate.provenance.chunk_ids)
        excerpts.extend(candidate.provenance.quotes)
        ref = registry.get_document_ref(candidate.provenance.doc_id)
        if ref:
            refs.append(ref)
    return Evidence(chunk_ids=chunk_ids, excerpts=excerpts, document_refs=refs)


def build_bid_record(bid_id: str, resolutions: Dict[str, FieldResolution], registry) -> BidRecord:
    """Turn resolutions into the internal 20-field record with provenance."""
    fields = BidFields()
    found = conflicting = 0

    for field_name, resolution in resolutions.items():
        spec = FIELD_CATALOG[field_name]

        if resolution.status == FieldStatus.not_found.value:
            # Nothing valid: the field stays at its not_found default. No value
            # is ever invented to fill a gap.
            continue

        if field_name == "contact_info":
            public_value: Any = _as_contacts(resolution.value)
        elif spec.value_kind.value == "list":
            public_value = _as_list(resolution.value)
        else:
            public_value = _as_string(resolution.value)

        extracted = ExtractedField(
            normalized_value=public_value,
            original_value=_original_text(resolution),
            status=FieldStatus(resolution.status),
            evidence=_evidence_for(resolution, registry),
            confidence=0.0,
        )

        if resolution.superseded:
            prior = resolution.superseded[0]
            extracted.is_superseded = True
            extracted.supersession = Supersession(
                superseded_by=resolution.chosen.provenance.doc_label,
                reason=(
                    f"rule={resolution.rule}; previous value from "
                    f"{prior.provenance.doc_label}: {prior.value!r}"
                ),
            )

        extracted.clarifications = [
            Clarification(description=f"{c.provenance.doc_label}: {c.value!r}")
            for c in resolution.clarifications
        ]
        extracted.conflicts = [
            Conflict(
                kind=ConflictKind.value_mismatch,
                description=f"{c.provenance.doc_label} reported {c.value!r}",
                involved_chunks=list(c.provenance.chunk_ids),
            )
            for c in resolution.conflicts
        ]

        setattr(fields, field_name, extracted)
        if resolution.status == FieldStatus.conflicting.value:
            conflicting += 1
        found += 1

    summary = ExtractionSummary(
        total_fields=len(FIELD_CATALOG),
        found_fields=found,
        missing_fields=len(FIELD_CATALOG) - found,
        conflicting_fields=conflicting,
    )
    return BidRecord(bid_id=bid_id, fields=fields, summary=summary)


def _original_text(resolution: FieldResolution) -> Optional[str]:
    if not resolution.contributing:
        return None
    values = [c.value for c in resolution.contributing]
    flat: List[str] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(str(v) for v in value)
        else:
            flat.append(str(value))
    return "; ".join(flat) if flat else None


def to_public_json(record: BidRecord) -> Dict[str, Any]:
    """The assignment-facing document: exactly the 20 labels, in order.

    A field with no valid grounded candidate is null. Values keep their source
    wording; only whitespace-level normalization is applied elsewhere.
    """
    public: Dict[str, Any] = {}
    for field_name, spec in FIELD_CATALOG.items():
        extracted = getattr(record.fields, field_name)
        if extracted.status == FieldStatus.not_found or extracted.normalized_value is None:
            public[spec.assignment_label] = None
            continue

        value = extracted.normalized_value
        if field_name == "contact_info":
            value = [
                {k: v for k, v in
                 {"name": c.name, "email": c.email, "phone": c.phone}.items() if v}
                for c in value
            ]
        public[spec.assignment_label] = value
    return public


def build_diagnostics(bid_id: str, resolutions: Dict[str, FieldResolution],
                      rejected: List[GroundedCandidate]) -> Dict[str, Any]:
    """Audit trail. Safe to keep provenance here: it never reaches a prompt."""
    fields: Dict[str, Any] = {}
    for field_name, resolution in resolutions.items():
        fields[field_name] = {
            "status": resolution.status,
            "rule": resolution.rule,
            "value": resolution.value,
            "chosen": _candidate_dump(resolution.chosen) if resolution.chosen else None,
            "contributing": [_candidate_dump(c) for c in resolution.contributing],
            "superseded": [_candidate_dump(c) for c in resolution.superseded],
            "clarifications": [_candidate_dump(c) for c in resolution.clarifications],
            "conflicts": [_candidate_dump(c) for c in resolution.conflicts],
            "considered": [_candidate_dump(c) for c in resolution.considered],
        }

    return {
        "bid_id": bid_id,
        "fields": fields,
        "rejected_candidates": [
            {
                "field_name": gc.field_name,
                "raw_value": gc.raw_value,
                "rejection_code": gc.rejection_code.value if gc.rejection_code else None,
                "rejection_reason": gc.rejection_reason,
                "chunk_ids": gc.chunk_ids,
            }
            for gc in rejected
        ],
    }


def _candidate_dump(candidate: ScoredCandidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.grounded.candidate_id,
        "value": candidate.value,
        "tier": candidate.tier,
        "role": candidate.role,
        "amendment_cue": candidate.cue,
        "doc_label": candidate.provenance.doc_label,
        "document_type": candidate.provenance.document_type.value,
        "addendum_number": candidate.provenance.addendum_number,
        "chunk_ids": candidate.provenance.chunk_ids,
        "quotes": candidate.provenance.quotes,
        "normalized": _normalized_dump(candidate),
    }


def _normalized_dump(candidate: ScoredCandidate) -> Optional[Any]:
    """Deterministic type normalization, kept beside the raw value.

    The public answer uses the source wording; this is only an internal,
    traceable view (for example a date marked relative rather than absolute).
    """
    spec = FIELD_CATALOG.get(candidate.grounded.field_name)
    if spec is None:
        return None
    kind = "contact" if candidate.grounded.field_name == "contact_info" else spec.value_kind.value
    try:
        return normalize_value(candidate.value, kind).model_dump()
    except Exception:  # normalization must never break resolution
        return None
