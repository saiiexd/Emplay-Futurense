"""Stage 4.2 prompt and response-contract layer.

Pure, deterministic and offline. This module owns exactly four things:

* the fixed extraction system prompt,
* rendering a ``ContextPackage`` into a model-facing user prompt,
* the strict machine-readable JSON response contract, and
* parsing raw model output into structures the Stage 4.1 grounding layer consumes.

It never calls an LLM, never reads the registry for evidence text, never grounds
quotes against chunk text, never normalizes values and never resolves candidates.
Those responsibilities belong to Stage 4.1 (``src/validation``) and Stages 4.3+.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import src.config as config
from src.schemas.candidates import (
    FieldCandidate,
    GroundedCandidate,
    GroupExtractionResponse,
    LLMCandidate,
    LLMEvidenceRef,
    NotFoundReport,
    ParsedGroupResponse,
)
from src.schemas.enums import RejectionCode, ValueKind
from src.schemas.field_catalog import FIELD_CATALOG

logger = logging.getLogger(__name__)

PROMPT_VERSION = config.PROMPT_VERSION


class PromptLeakageError(RuntimeError):
    """Raised when a rendered prompt contains information the model must not see."""


class EmptyPackageError(ValueError):
    """Raised when a package carries no offered fields or no excerpts to render."""


# --- system prompt -----------------------------------------------------------

SYSTEM_PROMPT = """You extract candidate field values from procurement document excerpts. Software checks every value against the excerpt text and handles all later decisions about how values relate. Your job is extraction only: do not interpret, reconcile, compare, or choose between values.

EVIDENCE
1. Use only the text inside <excerpt> elements. Do not use outside knowledge or assumptions about typical procurements.
2. Each excerpt has an id, a document label (doc), a follows attribute, and a fields attribute. Use an excerpt only for the fields named in its fields attribute.
3. Excerpts with the same doc label come from the same document and are shown in reading order. follows names the excerpt that comes immediately before this one in that document, or none. Consecutive excerpts can repeat a few lines of text.

CANDIDATES
4. Create a candidate only when an excerpt directly states the value. Never infer a value from another field, from a related document, or from what a document appears to be for. Never calculate, convert, or complete dates, times, time zones, amounts, units, model numbers, part numbers, names, or contact details.
5. value is copied exactly from its quotes, with the same characters, spelling, capitalization, and punctuation. A line break may be written as one space.
6. For kind "contact": value is null; contact holds the name, email, and phone copied exactly, with null for any part not in the quote.
7. For kind "summary": value is one or two short sentences that use only words appearing in its single quote.
8. For kind "item list": return one candidate per item. For kind "requirement lines": return one candidate per requirement line.
9. Report every supported value. When excerpts state different values for the same field, return each as its own candidate. This includes values described as new, changed, extended, or revised, and the values they refer to. Do not decide which value applies and do not leave any of them out.
10. The same value in different documents needs one candidate per document. The same value repeated in one document needs one candidate.
11. Text of a question asked by a bidder is not evidence. An answer to a question can be evidence.
12. Blank lines, underscores, and bracketed placeholders on forms are not values.

EVIDENCE ITEMS
13. Each candidate has one to three evidence items. Each item gives the id of one excerpt and a quote copied exactly from that excerpt as one continuous passage of at most 600 characters.
14. Every quote must contain the complete value itself. For kind "contact", every quote must contain every non-null contact part.
15. All evidence items of a candidate come from excerpts with the same doc label. Kinds "contact" and "summary" use exactly one evidence item.
16. Never shorten a quote with an ellipsis, never join text from two excerpts, and never add, remove, or reorder words.

MISSING INFORMATION
17. If no excerpt directly states a value for a field, add the field to not_found with reason_code "not_stated" when nothing addresses it, "mentioned_without_value" when the topic is referred to without stating a value or requirement, or "only_excluded_subjects" when the excerpts cover only subjects listed under Exclude. Keep reason under 200 characters.
18. Never use N/A, none, unknown, TBD, or similar placeholders as a value.
19. Every field listed under FIELDS appears in candidates or in not_found."""


# --- leakage control lists ---------------------------------------------------

#: Literals that must never appear anywhere in a rendered prompt.
FORBIDDEN_METADATA_LITERALS: Tuple[str, ...] = (
    "rfp_main",
    "portal_listing",
    "document_type",
    "doc_type",
    "addendum_number",
    "doc_id",
    "chunk_id",
    "page_numbers",
    "printed_page_label",
    "section_identifier",
    "retrieval_metadata",
    "fused_score",
    "overall_rank",
    "lexical_rank",
    "semantic_rank",
    "html_label_hit",
    "lexical_floor_inserted",
)

#: Words that must never appear in instruction text (excerpt bodies are exempt,
#: because they are source content rather than instructions).
BANNED_INSTRUCTION_WORDS: Tuple[str, ...] = (
    "prefer",
    "preferred",
    "latest",
    "newest",
    "most recent",
    "supersede",
    "superseded",
    "supersedes",
    "override",
    "precedence",
    "priority",
    "authoritative",
    "official",
    "primary",
    "rank",
    "score",
    "tier",
    "confidence",
    "addendum",
    "amendment",
    "past",
    "expired",
    "today",
)

_NOT_FOUND_REASON_CODES = ("not_stated", "mentioned_without_value", "only_excluded_subjects")

_PLACEHOLDER_WORDS = frozenset(
    {
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "unknown",
        "not specified",
        "not stated",
        "not applicable",
    }
)

_EXCERPT_BODY_RE = re.compile(r"(<excerpt\b[^>]*>)(.*?)(</excerpt>)", re.DOTALL)
_CONTACT_KEYS = ("name", "email", "phone")


# --- field presentation ------------------------------------------------------

def kind_word(field_name: str) -> str:
    """Model-facing description of how a field's value must be shaped."""
    if field_name == "contact_info":
        return "contact"
    if field_name == "bid_summary":
        return "summary"
    if field_name == "product_specification":
        return "requirement lines"
    spec = FIELD_CATALOG.get(field_name)
    if spec is None:
        return "text"
    if spec.value_kind == ValueKind.list:
        return "item list"
    if spec.value_kind == ValueKind.datetime:
        return "date/time text"
    return "text"


def _require_renderable(package) -> None:
    if not package.fields:
        raise EmptyPackageError(
            f"package {package.group}/{package.batch_index} offers no fields; nothing to render"
        )
    if not package.excerpts:
        raise EmptyPackageError(
            f"package {package.group}/{package.batch_index} has no excerpts; nothing to render"
        )


# --- rendering ---------------------------------------------------------------

def _header_lines(group: str, fields: List[str]) -> List[str]:
    """TASK and FIELDS sections, ending with the EVIDENCE heading."""
    lines: List[str] = [
        "TASK",
        "Extract candidate values for the fields under FIELDS using only the excerpts under EVIDENCE.",
        f"Group: {group}",
        "",
        "FIELDS",
    ]
    for idx, field_name in enumerate(fields):
        spec = FIELD_CATALOG[field_name]
        if idx:
            lines.append("")
        lines.append(f"[{field_name}] kind: {kind_word(field_name)}")
        lines.append(f"Definition: {spec.definition}")
        lines.append(f"Include: {spec.include}")
        lines.append(f"Exclude: {spec.exclude}")
    lines.append("")
    lines.append("EVIDENCE")
    return lines


def _footer_lines() -> List[str]:
    return [
        "",
        "OUTPUT",
        "Return one JSON object matching the response schema. "
        "Every field under FIELDS appears in candidates or in not_found.",
    ]


def _excerpt_text(excerpt) -> str:
    text = excerpt.text
    if "</excerpt>" in text:
        logger.warning(
            "excerpt %s contains a literal </excerpt>; escaped for rendering only",
            excerpt.handle,
        )
        text = text.replace("</excerpt>", "</excerpt_>")
    return text


def _excerpt_block(excerpt) -> str:
    """The three rendered lines for one excerpt, joined as they appear in the prompt."""
    follows = excerpt.follows or "none"
    fields_attr = ",".join(excerpt.eligible_fields)
    open_tag = (
        f'<excerpt id="{excerpt.handle}" doc="{excerpt.doc_label}" '
        f'follows="{follows}" fields="{fields_attr}">'
    )
    return "\n".join([open_tag, _excerpt_text(excerpt), "</excerpt>"])


def static_overhead_chars(group: str, fields: List[str]) -> int:
    """Characters a prompt costs before any evidence is added.

    Counts the system prompt plus the TASK/FIELDS/OUTPUT sections. The
    ContextBuilder subtracts this from the prompt limit so the *rendered*
    prompt, not just the evidence text, stays within budget.
    """
    skeleton = "\n".join(_header_lines(group, list(fields)) + _footer_lines())
    return len(SYSTEM_PROMPT) + len(skeleton)


def excerpt_overhead_chars(excerpt) -> int:
    """Characters one excerpt costs on top of its own text.

    That is the open/close tags plus the newline that joins this block into
    the prompt, so ``static + sum(len(text) + overhead)`` is the exact
    rendered size of system + user messages.
    """
    return len(_excerpt_block(excerpt)) - len(_excerpt_text(excerpt)) + 1


def render_user_prompt(package) -> str:
    """Render the model-facing user message for a single ``ContextPackage``."""
    _require_renderable(package)

    parts = _header_lines(package.group, list(package.fields))
    parts.extend(_excerpt_block(excerpt) for excerpt in package.excerpts)
    parts.extend(_footer_lines())
    return "\n".join(parts)


def render_messages(package) -> List[Dict[str, str]]:
    """Return the ``[system, user]`` message list for a package."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(package)},
    ]


def render_correction(errors: List[str]) -> str:
    """Correction text appended by Stage 4.3 after an unusable response."""
    joined = "; ".join(e for e in errors if e)
    if len(joined) > config.MAX_CORRECTION_CHARS:
        joined = joined[: config.MAX_CORRECTION_CHARS]
    return (
        "CORRECTION\n"
        f"Your previous response could not be used: {joined}. "
        "Return a new JSON object that follows the schema and every rule."
    )


# --- response schema ---------------------------------------------------------

def response_schema(package) -> Dict[str, Any]:
    """Strict JSON schema constrained to this package's fields and handles."""
    _require_renderable(package)

    field_enum = list(package.fields)
    handle_enum = [excerpt.handle for excerpt in package.excerpts]

    contact_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_CONTACT_KEYS),
        "properties": {key: {"type": ["string", "null"]} for key in _CONTACT_KEYS},
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["group", "candidates", "not_found"],
        "properties": {
            "group": {"type": "string", "enum": [package.group]},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "value", "contact", "evidence"],
                    "properties": {
                        "field": {"type": "string", "enum": field_enum},
                        "value": {"type": ["string", "null"]},
                        "contact": {"anyOf": [contact_schema, {"type": "null"}]},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["evidence_id", "quote"],
                                "properties": {
                                    "evidence_id": {"type": "string", "enum": handle_enum},
                                    "quote": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "not_found": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "reason_code", "reason"],
                    "properties": {
                        "field": {"type": "string", "enum": field_enum},
                        "reason_code": {"type": "string", "enum": list(_NOT_FOUND_REASON_CODES)},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def prompt_hash(package) -> str:
    """Deterministic hash over everything that defines this call's prompt."""
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "user": render_user_prompt(package),
            "schema": response_schema(package),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- leakage guard -----------------------------------------------------------

def _instruction_text(messages: List[Dict[str, str]]) -> str:
    """Prompt text with excerpt bodies removed; excerpt tags are kept."""
    parts = []
    for message in messages:
        content = message.get("content", "")
        parts.append(_EXCERPT_BODY_RE.sub(lambda m: m.group(1) + m.group(3), content))
    return "\n".join(parts)


def assert_no_leakage(messages: List[Dict[str, str]], package, registry) -> None:
    """Raise :class:`PromptLeakageError` if hidden provenance reached the prompt."""
    full_text = "\n".join(m.get("content", "") for m in messages)
    violations: List[str] = []

    if registry is not None:
        for doc_id, doc_ref in registry.doc_refs.items():
            if doc_id and doc_id in full_text:
                violations.append(f"document id '{doc_id}'")
            filename = getattr(doc_ref, "filename", "") or ""
            if filename and filename in full_text:
                violations.append(f"filename '{filename}'")
            stem = filename.rsplit(".", 1)[0] if "." in filename else filename
            if stem and len(stem) >= 4 and stem in full_text:
                violations.append(f"filename stem '{stem}'")
        for chunk_id in registry.chunk_metadata:
            if chunk_id and chunk_id in full_text:
                violations.append(f"chunk id '{chunk_id}'")

    for chunk_id in package.handle_map.values():
        if chunk_id and chunk_id in full_text:
            violations.append(f"chunk id '{chunk_id}'")

    for literal in FORBIDDEN_METADATA_LITERALS:
        if literal in full_text:
            violations.append(f"metadata literal '{literal}'")

    instructions = _instruction_text(messages).lower()
    for word in BANNED_INSTRUCTION_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", instructions):
            violations.append(f"banned instruction word '{word}'")

    if violations:
        unique = sorted(set(violations))
        raise PromptLeakageError(
            f"prompt for group '{package.group}' leaks: " + ", ".join(unique)
        )


# --- parsing -----------------------------------------------------------------

def _is_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\W*", stripped):
        return True
    if stripped.casefold() in _PLACEHOLDER_WORDS:
        return True
    if "___" in stripped:
        return True
    if re.fullmatch(r"[\(\[\{].*[\)\]\}]", stripped, re.DOTALL):
        return True
    return False


def _reject(
    field_name: str,
    code: RejectionCode,
    detail: str,
    raw_value: Any,
    chunk_ids: List[str],
) -> GroundedCandidate:
    return GroundedCandidate(
        candidate_id=GroundedCandidate.generate_id(field_name, chunk_ids, raw_value),
        field_name=field_name,
        raw_value=raw_value,
        chunk_ids=chunk_ids,
        excerpts=[],
        is_valid=False,
        rejection_code=code,
        rejection_reason=detail,
    )


def _contact_dict(contact) -> Dict[str, Optional[str]]:
    return {key: getattr(contact, key) for key in _CONTACT_KEYS}


def _raw_value_of(wire) -> Any:
    if wire.contact is not None:
        return _contact_dict(wire.contact)
    return wire.value


def parse_response(raw_text: str, package) -> ParsedGroupResponse:
    """Validate raw model output against this package's contract.

    Structural and contract checks only. Quote grounding, value support,
    document-type eligibility and normalization stay in their own layers.
    """
    catalog_index = {name: i for i, name in enumerate(FIELD_CATALOG)}
    handle_to_doc: Dict[str, str] = {}
    handle_to_pos: Dict[str, int] = {}
    for position, excerpt in enumerate(package.excerpts):
        handle_to_doc[excerpt.handle] = excerpt.doc_label
        handle_to_pos[excerpt.handle] = position

    # Step 1 - envelope
    try:
        payload = json.loads(raw_text)
    except (ValueError, TypeError) as exc:
        return ParsedGroupResponse(status="invalid_json", errors=[f"response is not valid JSON: {exc}"])

    try:
        envelope = GroupExtractionResponse.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        errors = []
        for err in getattr(exc, "errors", lambda: [])()[:5]:
            location = ".".join(str(p) for p in err.get("loc", ()))
            errors.append(f"{location or '<root>'}: {err.get('msg', 'invalid')}")
        return ParsedGroupResponse(status="schema_error", errors=errors or [str(exc)[:200]])

    if envelope.group != package.group:
        return ParsedGroupResponse(
            status="schema_error",
            errors=[f"group: expected '{package.group}', got '{envelope.group}'"],
        )

    warnings: List[str] = []
    rejected: List[GroundedCandidate] = []
    accepted: List[FieldCandidate] = []

    wire_candidates = envelope.candidates
    if len(wire_candidates) > config.MAX_CANDIDATES_PER_RESPONSE:
        warnings.append("candidate_cap")
        wire_candidates = wire_candidates[: config.MAX_CANDIDATES_PER_RESPONSE]

    for wire in wire_candidates:
        field_name = wire.field
        raw_value = _raw_value_of(wire)
        chunk_ids = [
            package.handle_map[e.evidence_id]
            for e in wire.evidence
            if e.evidence_id in package.handle_map
        ]

        # P1 - field must be offered by this package
        if field_name not in package.fields:
            rejected.append(
                _reject(field_name, RejectionCode.unknown_field,
                        f"field '{field_name}' is not offered in this package", raw_value, chunk_ids)
            )
            continue

        # Evidence must exist before a document label can be derived. Grounding
        # would reject this with the same code; doing it here avoids emitting a
        # candidate that cannot be attributed to a document.
        if not wire.evidence:
            rejected.append(
                _reject(field_name, RejectionCode.no_evidence,
                        "candidate has no evidence items", raw_value, chunk_ids)
            )
            continue

        # P2 - every handle must belong to this package
        unknown = [e.evidence_id for e in wire.evidence if e.evidence_id not in package.handle_map]
        if unknown:
            rejected.append(
                _reject(field_name, RejectionCode.unknown_evidence_id,
                        f"unknown evidence id(s): {', '.join(sorted(set(unknown)))}", raw_value, chunk_ids)
            )
            continue

        # P3 - value / contact shape
        if field_name == "contact_info":
            if wire.value is not None or wire.contact is None:
                rejected.append(
                    _reject(field_name, RejectionCode.format_error,
                            "contact_info requires value=null and a contact object", raw_value, chunk_ids)
                )
                continue
        else:
            if wire.contact is not None:
                rejected.append(
                    _reject(field_name, RejectionCode.format_error,
                            f"field '{field_name}' must not carry a contact object", raw_value, chunk_ids)
                )
                continue
            if not isinstance(wire.value, str) or not wire.value.strip():
                rejected.append(
                    _reject(field_name, RejectionCode.format_error,
                            f"field '{field_name}' requires a non-empty string value", raw_value, chunk_ids)
                )
                continue
            if len(wire.value) > config.MAX_VALUE_CHARS:
                rejected.append(
                    _reject(field_name, RejectionCode.format_error,
                            f"value exceeds {config.MAX_VALUE_CHARS} characters", raw_value, chunk_ids)
                )
                continue

        # P4 - a contact must carry at least one part
        if field_name == "contact_info":
            parts = [getattr(wire.contact, key) for key in _CONTACT_KEYS]
            if not any(p and str(p).strip() for p in parts):
                rejected.append(
                    _reject(field_name, RejectionCode.contact_empty,
                            "contact object has no name, email or phone", raw_value, chunk_ids)
                )
                continue

        # P5 - placeholders are never values
        if field_name == "contact_info":
            placeholder_parts = [
                key for key in _CONTACT_KEYS
                if getattr(wire.contact, key) and _is_placeholder(getattr(wire.contact, key))
            ]
            if placeholder_parts:
                rejected.append(
                    _reject(field_name, RejectionCode.format_error,
                            f"placeholder contact part(s): {', '.join(placeholder_parts)}", raw_value, chunk_ids)
                )
                continue
        elif _is_placeholder(wire.value):
            rejected.append(
                _reject(field_name, RejectionCode.format_error,
                        f"value '{wire.value.strip()[:40]}' is a placeholder", raw_value, chunk_ids)
            )
            continue

        # P6 - quote shape
        bad_quote = next(
            (e for e in wire.evidence
             if not e.quote.strip() or len(e.quote) > config.MAX_QUOTE_CHARS),
            None,
        )
        if bad_quote is not None:
            rejected.append(
                _reject(field_name, RejectionCode.invalid_quote,
                        f"quote for {bad_quote.evidence_id} is empty or exceeds "
                        f"{config.MAX_QUOTE_CHARS} characters", raw_value, chunk_ids)
            )
            continue

        # Step 3.1 - drop duplicate evidence items
        seen: set = set()
        evidence = []
        for item in wire.evidence:
            key = (item.evidence_id, item.quote)
            if key in seen:
                warnings.append(f"duplicate_evidence:{field_name}")
                continue
            seen.add(key)
            evidence.append(item)

        # Step 3.2 - evidence cardinality
        limit = 1 if field_name in ("contact_info", "bid_summary") else config.MAX_EVIDENCE_PER_CANDIDATE
        if len(evidence) > limit:
            warnings.append(f"evidence_truncated:{field_name}")
            evidence = evidence[:limit]

        # Step 3.3 - never merge evidence across documents
        by_doc: Dict[str, List[Any]] = {}
        for item in evidence:
            by_doc.setdefault(handle_to_doc[item.evidence_id], []).append(item)
        if len(by_doc) > 1:
            warnings.append(f"cross_document_split:{field_name}")

        # Step 3.4 - adapt to the existing LLMCandidate contract
        for doc_label, items in by_doc.items():
            accepted.append(
                FieldCandidate(
                    field_name=field_name,
                    doc_label=doc_label,
                    candidate=LLMCandidate(
                        value=raw_value,
                        evidence=[
                            LLMEvidenceRef(
                                chunk_id=package.handle_map[item.evidence_id],
                                quote=item.quote,
                                explanation="",
                            )
                            for item in items
                        ],
                        confidence_score=None,
                    ),
                )
            )

    # Step 3.5 - drop exact duplicate candidates
    deduped: List[FieldCandidate] = []
    seen_candidates: set = set()
    for fc in accepted:
        key = (
            fc.field_name,
            json.dumps(fc.candidate.value, sort_keys=True, ensure_ascii=False),
            frozenset(ev.chunk_id for ev in fc.candidate.evidence),
        )
        if key in seen_candidates:
            warnings.append(f"duplicate_candidate:{fc.field_name}")
            continue
        seen_candidates.add(key)
        deduped.append(fc)

    # Step 4 - not_found entries
    not_found: List[NotFoundReport] = []
    for entry in envelope.not_found:
        if entry.field not in package.fields:
            warnings.append("not_found_unknown_field")
            continue
        reason = entry.reason[: config.MAX_REASON_CHARS]
        not_found.append(
            NotFoundReport(field_name=entry.field, reason_code=entry.reason_code, reason=reason)
        )

    # Step 5 - coverage
    with_candidates = {fc.field_name for fc in deduped}
    with_not_found = {nf.field_name for nf in not_found}
    coverage: Dict[str, str] = {}
    for field_name in package.fields:
        in_c = field_name in with_candidates
        in_nf = field_name in with_not_found
        if in_c and in_nf:
            coverage[field_name] = "both"
            warnings.append(f"not_found_contradicted:{field_name}")
        elif in_c:
            coverage[field_name] = "candidates"
        elif in_nf:
            coverage[field_name] = "not_found"
        else:
            coverage[field_name] = "missing"

    # Step 6 - deterministic ordering
    def _doc_number(label: str) -> int:
        return int(label[1:]) if label[1:].isdigit() else 0

    def _first_pos(fc: FieldCandidate) -> int:
        positions = [
            handle_to_pos[h]
            for h in (
                handle
                for handle, cid in package.handle_map.items()
                for ev in fc.candidate.evidence
                if ev.chunk_id == cid
            )
            if h in handle_to_pos
        ]
        return min(positions) if positions else 0

    deduped.sort(
        key=lambda fc: (
            catalog_index.get(fc.field_name, len(catalog_index)),
            _doc_number(fc.doc_label),
            _first_pos(fc),
            json.dumps(fc.candidate.value, sort_keys=True, ensure_ascii=False),
        )
    )
    rejected.sort(key=lambda gc: (gc.field_name, gc.candidate_id))
    not_found.sort(key=lambda nf: catalog_index.get(nf.field_name, len(catalog_index)))

    return ParsedGroupResponse(
        status="ok",
        errors=[],
        for_grounding=deduped,
        rejected=rejected,
        not_found=not_found,
        coverage=coverage,
        warnings=warnings,
    )
