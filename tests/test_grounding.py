import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.registry import DocumentRegistry
from src.schemas.candidates import LLMCandidate, LLMEvidenceRef
from src.schemas.enums import RejectionCode
from src.validation.grounding import GroundingValidator

class TestGroundingValidator(unittest.TestCase):
    def setUp(self):
        self.registry = DocumentRegistry()
        doc_data = {
            "metadata": {
                "doc_id": "doc1",
                "document_type": "rfp_main"
            },
            "sections": [
                {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"page_number": 1}}
            ]
        }
        chunks = [
            {"chunk_id": "c1", "text": "The quick brown fox jumps over the lazy dog."}
        ]
        self.registry.register_document(doc_data, chunks)
        self.validator = GroundingValidator(self.registry)

    def test_unknown_field(self):
        cand = LLMCandidate(value="123", evidence=[LLMEvidenceRef(chunk_id="c1", quote="quick brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("some_fake_field", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.unknown_field)

    def test_invalid_quote(self):
        cand = LLMCandidate(value="123", evidence=[LLMEvidenceRef(chunk_id="c1", quote="quick...fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_number", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.invalid_quote)

    def test_unknown_evidence_id(self):
        cand = LLMCandidate(value="123", evidence=[LLMEvidenceRef(chunk_id="fake_chunk", quote="quick brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_number", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.unknown_evidence_id)

    def test_quote_not_in_chunk(self):
        cand = LLMCandidate(value="123", evidence=[LLMEvidenceRef(chunk_id="c1", quote="slow brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_number", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.quote_not_in_chunk)

    def test_value_not_in_quote(self):
        cand = LLMCandidate(value="cat", evidence=[LLMEvidenceRef(chunk_id="c1", quote="The quick brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_number", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.value_not_in_quote)

    def test_ineligible_doc_type(self):
        # We need a field that doesn't allow rfp_main, or a doc that isn't allowed.
        # specification allows addendum, rfp_main, specification.
        # portal_listing might not be allowed for bid_bond_requirement.
        doc_data2 = {
            "metadata": {"doc_id": "doc2", "document_type": "portal_listing"},
            "sections": [{"text": "Bond is 5%", "metadata": {"page_number": 1}}]
        }
        chunks2 = [{"chunk_id": "c2", "text": "Bond is 5%"}]
        self.registry.register_document(doc_data2, chunks2)
        
        cand = LLMCandidate(value="5%", evidence=[LLMEvidenceRef(chunk_id="c2", quote="Bond is 5%", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_bond_requirement", cand) # portal_listing not eligible for bid_bond_requirement
        self.assertEqual(gc.rejection_code, RejectionCode.ineligible_doc_type)

    def test_contact_empty(self):
        cand = LLMCandidate(value={"name": "", "email": ""}, evidence=[LLMEvidenceRef(chunk_id="c1", quote="The quick brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("contact_info", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.contact_empty)

    def test_contact_field_not_in_quote(self):
        cand = LLMCandidate(value={"name": "Alice", "email": "alice@ex.com"}, evidence=[LLMEvidenceRef(chunk_id="c1", quote="The quick brown fox jumps", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("contact_info", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.contact_field_not_in_quote)

    def test_summary_unsupported_token(self):
        cand = LLMCandidate(value="An entirely hallucinated summary about cats", evidence=[LLMEvidenceRef(chunk_id="c1", quote="The quick brown fox jumps over the lazy dog.", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("bid_summary", cand)
        self.assertEqual(gc.rejection_code, RejectionCode.summary_unsupported_token)

    def test_successful_grounding(self):
        cand = LLMCandidate(value="fox", evidence=[LLMEvidenceRef(chunk_id="c1", quote="The quick brown fox", explanation="none")], confidence_score=0.9)
        gc = self.validator.validate("product", cand)
        self.assertTrue(gc.is_valid)
        self.assertIsNone(gc.rejection_code)
        self.assertEqual(gc.excerpts[0], "The quick brown fox")

if __name__ == '__main__':
    unittest.main()
