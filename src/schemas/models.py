from typing import Generic, TypeVar, Literal, Any, List, Optional
from pydantic import BaseModel, Field, model_validator
from src.schemas.enums import DocType, FieldStatus, EvidenceRole, Scope, ConflictKind
from src.schemas.candidates import RunInfo

T = TypeVar('T')

class DocumentRef(BaseModel):
    document_id: str
    filename: str
    document_type: DocType
    doc_date: Optional[str] = None
    addendum_number: Optional[int] = None
    page_index: Optional[int] = None
    printed_page_label: Optional[str] = None

class Evidence(BaseModel):
    chunk_ids: List[str] = Field(default_factory=list)
    excerpts: List[str] = Field(default_factory=list)
    document_refs: List[DocumentRef] = Field(default_factory=list)
    role: EvidenceRole = EvidenceRole.primary

class Contact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_primary: bool = False

class Supersession(BaseModel):
    superseded_by: str
    reason: str

class Clarification(BaseModel):
    description: str

class Conflict(BaseModel):
    kind: ConflictKind
    description: str
    involved_chunks: List[str]

class ExtractedField(BaseModel, Generic[T]):
    normalized_value: Optional[T] = None
    original_value: Optional[str] = None
    status: FieldStatus = FieldStatus.not_found
    evidence: Optional[Evidence] = None
    confidence: float = 0.0
    is_superseded: bool = False
    run_info: Optional[RunInfo] = None
    supersession: Optional[Supersession] = None
    clarifications: List[Clarification] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_status(self):
        if self.status in [FieldStatus.found, FieldStatus.conflicting] and not self.evidence:
            raise ValueError("Evidence must be provided if status is 'found' or 'conflicting'")
        if self.status == FieldStatus.not_found and self.normalized_value is not None:
            raise ValueError("Normalized value must be null if status is 'not_found'")
        return self

class BidFields(BaseModel):
    bid_number: ExtractedField[str] = Field(default_factory=ExtractedField)
    title: ExtractedField[str] = Field(default_factory=ExtractedField)
    due_date: ExtractedField[str] = Field(default_factory=ExtractedField)
    bid_submission_type: ExtractedField[str] = Field(default_factory=ExtractedField)
    term_of_bid: ExtractedField[str] = Field(default_factory=ExtractedField)
    pre_bid_meeting: ExtractedField[str] = Field(default_factory=ExtractedField)
    installation: ExtractedField[str] = Field(default_factory=ExtractedField)
    bid_bond_requirement: ExtractedField[str] = Field(default_factory=ExtractedField)
    delivery_date: ExtractedField[str] = Field(default_factory=ExtractedField)
    payment_terms: ExtractedField[str] = Field(default_factory=ExtractedField)
    additional_documentation: ExtractedField[List[str]] = Field(default_factory=ExtractedField)
    mfg_for_registration: ExtractedField[List[str]] = Field(default_factory=ExtractedField)
    contract_or_cooperative: ExtractedField[str] = Field(default_factory=ExtractedField)
    model_no: ExtractedField[List[str]] = Field(default_factory=ExtractedField)
    part_no: ExtractedField[List[str]] = Field(default_factory=ExtractedField)
    product: ExtractedField[List[str]] = Field(default_factory=ExtractedField)
    contact_info: ExtractedField[List[Contact]] = Field(default_factory=ExtractedField)
    company_name: ExtractedField[str] = Field(default_factory=ExtractedField)
    bid_summary: ExtractedField[str] = Field(default_factory=ExtractedField)
    product_specification: ExtractedField[str] = Field(default_factory=ExtractedField)

class ExtractionSummary(BaseModel):
    total_fields: int = 20
    found_fields: int = 0
    missing_fields: int = 20
    conflicting_fields: int = 0

class BidRecord(BaseModel):
    bid_id: str
    fields: BidFields
    summary: ExtractionSummary
