from enum import Enum, auto

class DocType(str, Enum):
    rfp_main = "rfp_main"
    addendum = "addendum"
    portal_listing = "portal_listing"
    specification = "specification"
    affidavit = "affidavit"
    unknown = "unknown"

class GroupName(str, Enum):
    identity = "identity"
    schedule = "schedule"
    terms = "terms"
    products = "products"
    specifications = "specifications"
    contacts_summary = "contacts_summary"

class ValueKind(str, Enum):
    string = "string"
    datetime = "datetime"
    list = "list"

class Aggregation(str, Enum):
    first_found = "first_found"
    latest = "latest"
    concat = "concat"

class FieldStatus(str, Enum):
    found = "found"
    not_found = "not_found"
    conflicting = "conflicting"

class EvidenceRole(str, Enum):
    primary = "primary"
    supporting = "supporting"
    conflicting = "conflicting"

class MatchLevel(str, Enum):
    exact = "exact"
    semantic = "semantic"
    none = "none"

class Scope(str, Enum):
    global_scope = "global"
    local_scope = "local"

class ConflictKind(str, Enum):
    value_mismatch = "value_mismatch"
    date_mismatch = "date_mismatch"
    ambiguous = "ambiguous"

class RejectionCode(str, Enum):
    no_evidence = "no_evidence"
    hallucination = "hallucination"
    out_of_scope = "out_of_scope"
    format_error = "format_error"
    quote_not_in_chunk = "quote_not_in_chunk"
    value_not_in_quote = "value_not_in_quote"
    unknown_evidence_id = "unknown_evidence_id"
    ineligible_doc_type = "ineligible_doc_type"
    invalid_quote = "invalid_quote"
    contact_empty = "contact_empty"
    contact_field_not_in_quote = "contact_field_not_in_quote"
    summary_unsupported_token = "summary_unsupported_token"
    unknown_field = "unknown_field"

class CandidateOrigin(str, Enum):
    llm = "llm"
    rule = "rule"
    fallback = "fallback"

class CallStatus(str, Enum):
    success = "success"
    error = "error"
    timeout = "timeout"
    rate_limited = "rate_limited"

class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
