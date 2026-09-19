from src.schemas.candidates import LLMCandidate, GroundedCandidate
from src.schemas.enums import RejectionCode, MatchLevel, CandidateOrigin
from src.schemas.field_catalog import FIELD_CATALOG
from src.retrieval.registry import DocumentRegistry
from src.validation.canonical import find_canonical_match
import re

class GroundingValidator:
    def __init__(self, registry: DocumentRegistry):
        self.registry = registry

    def _validate_summary_tokens(self, candidate_value: str, excerpt: str) -> bool:
        # A simple check: do all long tokens in the value appear in the excerpt?
        tokens = [t for t in re.sub(r'[^\w\s]', ' ', candidate_value.lower()).split() if len(t) > 4]
        exc_lower = excerpt.lower()
        for t in tokens:
            if t not in exc_lower:
                return False
        return True

    def validate(self, field_name: str, candidate: LLMCandidate) -> GroundedCandidate:
        gc = GroundedCandidate(
            candidate_id="",
            field_name=field_name,
            raw_value=candidate.value,
            chunk_ids=[],
            excerpts=[],
            origin=CandidateOrigin.llm,
            is_valid=False
        )

        spec = FIELD_CATALOG.get(field_name)
        if not spec:
            gc.rejection_code = RejectionCode.unknown_field
            gc.rejection_reason = "unknown_field"
            return gc
            
        if not candidate.evidence:
            gc.rejection_code = RejectionCode.no_evidence
            gc.rejection_reason = "No evidence provided."
            return gc
            
        chunk_ids = []
        excerpts = []
        best_match_level = None
        
        for ev in candidate.evidence:
            if "..." in ev.quote:
                gc.rejection_code = RejectionCode.invalid_quote
                gc.rejection_reason = "invalid_quote"
                return gc
                
            chunk = self.registry.chunk_metadata.get(ev.chunk_id)
            if not chunk:
                gc.rejection_code = RejectionCode.unknown_evidence_id
                gc.rejection_reason = "unknown_evidence_id"
                return gc
                
            doc_ref = self.registry.get_document_ref(chunk["doc_id"])
            if not doc_ref or doc_ref.document_type not in spec.eligible_document_types:
                gc.rejection_code = RejectionCode.ineligible_doc_type
                gc.rejection_reason = "ineligible_doc_type"
                return gc
                
            match = find_canonical_match(ev.quote, chunk["text"])
            if not match.matched:
                gc.rejection_code = RejectionCode.quote_not_in_chunk
                gc.rejection_reason = "quote_not_in_chunk"
                return gc
                
            if isinstance(candidate.value, str):
                if field_name == "bid_summary":
                    if not self._validate_summary_tokens(candidate.value, match.original_substring):
                        gc.rejection_code = RejectionCode.summary_unsupported_token
                        gc.rejection_reason = "summary_unsupported_token"
                        return gc
                else:
                    val_match = find_canonical_match(candidate.value, match.original_substring)
                    if not val_match.matched:
                        gc.rejection_code = RejectionCode.value_not_in_quote
                        gc.rejection_reason = "value_not_in_quote"
                        return gc
                        
            elif isinstance(candidate.value, dict):
                has_valid = False
                for k, v in candidate.value.items():
                    if k in ["name", "email", "phone"] and v:
                        has_valid = True
                        v_match = find_canonical_match(v, match.original_substring)
                        if not v_match.matched:
                            gc.rejection_code = RejectionCode.contact_field_not_in_quote
                            gc.rejection_reason = "contact_field_not_in_quote"
                            return gc
                if not has_valid:
                    gc.rejection_code = RejectionCode.contact_empty
                    gc.rejection_reason = "contact_empty"
                    return gc
                    
            elif isinstance(candidate.value, list):
                for item in candidate.value:
                    if isinstance(item, str):
                        i_match = find_canonical_match(item, match.original_substring)
                        if not i_match.matched:
                            gc.rejection_code = RejectionCode.value_not_in_quote
                            gc.rejection_reason = "value_not_in_quote"
                            return gc
                            
            ml = MatchLevel.exact if match.level == "L1" else MatchLevel.semantic
            if best_match_level is None or ml == MatchLevel.exact:
                best_match_level = ml
                
            chunk_ids.append(ev.chunk_id)
            excerpts.append(match.original_substring)

        gc.chunk_ids = chunk_ids
        gc.excerpts = excerpts
        gc.is_valid = True
        gc.match_level = best_match_level or MatchLevel.none
        gc.candidate_id = GroundedCandidate.generate_id(field_name, chunk_ids, gc.raw_value)
        
        return gc
