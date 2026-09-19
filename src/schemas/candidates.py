import hashlib
import json
from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.schemas.enums import MatchLevel, RejectionCode, CandidateOrigin, CallStatus, RunStatus

class BaseForbid(BaseModel):
    model_config = ConfigDict(extra="forbid")

class LLMContact(BaseForbid):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_primary: bool = False

class LLMEvidenceRef(BaseForbid):
    chunk_id: str
    quote: str
    explanation: str

class LLMCandidate(BaseForbid):
    value: Any
    evidence: List[LLMEvidenceRef]
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)
    is_superseding: bool = False
    supersedes_reason: Optional[str] = None

class LLMNotFound(BaseForbid):
    reason: str

class LLMExtractionResponse(BaseForbid):
    found: bool
    candidates: List[LLMCandidate] = Field(default_factory=list)
    not_found: Optional[LLMNotFound] = None

# --- Stage 4.2 wire contract -------------------------------------------------
# These models mirror the strict JSON schema sent to the model. Every key is
# required (values may be null) so that a response missing a key is rejected
# rather than silently defaulted.

class WireContact(BaseForbid):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]

class WireEvidence(BaseForbid):
    evidence_id: str
    quote: str

class WireCandidate(BaseForbid):
    field: str
    value: Optional[str]
    contact: Optional[WireContact]
    evidence: List[WireEvidence]

class WireNotFound(BaseForbid):
    field: str
    reason_code: Literal["not_stated", "mentioned_without_value", "only_excluded_subjects"]
    reason: str

class GroupExtractionResponse(BaseForbid):
    group: str
    candidates: List[WireCandidate]
    not_found: List[WireNotFound]

class NormalizedValue(BaseForbid):
    value: Any
    data_type: str

class GroundedCandidate(BaseModel):
    candidate_id: str
    field_name: str
    raw_value: Any
    normalized_value: Optional[NormalizedValue] = None
    chunk_ids: List[str]
    excerpts: List[str]
    origin: CandidateOrigin = CandidateOrigin.llm
    is_valid: bool = True
    match_level: MatchLevel = MatchLevel.none
    rejection_code: Optional[RejectionCode] = None
    rejection_reason: Optional[str] = None
    
    @classmethod
    def generate_id(cls, field_name: str, chunk_ids: List[str], raw_value: Any) -> str:
        cids = "_".join(sorted(chunk_ids))
        val_str = json.dumps(raw_value, sort_keys=True)
        key = f"{field_name}:{cids}:{val_str}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()[:12]

# --- Stage 4.2 parse output --------------------------------------------------

class FieldCandidate(BaseModel):
    """A structurally valid candidate, ready for the Stage 4.1 grounding layer."""
    field_name: str
    doc_label: str
    candidate: LLMCandidate

class NotFoundReport(BaseModel):
    field_name: str
    reason_code: Literal["not_stated", "mentioned_without_value", "only_excluded_subjects"]
    reason: str

class ParsedGroupResponse(BaseModel):
    status: Literal["ok", "invalid_json", "schema_error"]
    errors: List[str] = Field(default_factory=list)
    for_grounding: List[FieldCandidate] = Field(default_factory=list)
    rejected: List[GroundedCandidate] = Field(default_factory=list)
    not_found: List[NotFoundReport] = Field(default_factory=list)
    coverage: Dict[str, Literal["candidates", "not_found", "both", "missing"]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

class CallRecord(BaseModel):
    call_id: str
    status: CallStatus
    duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    error_message: Optional[str] = None

class CoverageEntry(BaseModel):
    chunk_id: str
    field_name: str
    was_provided: bool
    was_used: bool

class RunInfo(BaseModel):
    run_id: str
    timestamp: str
    status: RunStatus
    calls: List[CallRecord] = Field(default_factory=list)
    coverage: List[CoverageEntry] = Field(default_factory=list)

class CandidateFile(BaseModel):
    candidates: List[GroundedCandidate]
    run_info: RunInfo
