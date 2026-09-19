import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schemas.models import (
    Contact,
    Evidence,
    ExtractedField,
    BidFields,
    BidRecord,
    ExtractionSummary,
    DocumentRef,
    Supersession,
    Clarification,
    Conflict
)
from src.schemas.enums import DocType, FieldStatus, EvidenceRole, ConflictKind

from typing import List

class TestSchemas(unittest.TestCase):
    def test_extracted_field_validation(self):
        doc_ref = DocumentRef(document_id="doc1", filename="test.pdf", document_type=DocType.rfp_main)
        ev = Evidence(chunk_ids=["1"], document_refs=[doc_ref], role=EvidenceRole.primary)
        
        # Valid not found
        field_nf = ExtractedField[str](status=FieldStatus.not_found)
        self.assertIsNone(field_nf.normalized_value)
        
        # Valid found
        field_f = ExtractedField[str](normalized_value="Test", status=FieldStatus.found, evidence=ev)
        self.assertEqual(field_f.normalized_value, "Test")

    def test_bid_record(self):
        doc_ref = DocumentRef(document_id="doc1", filename="test.pdf", document_type=DocType.rfp_main)
        ev = Evidence(chunk_ids=["1"], document_refs=[doc_ref])
        
        fields = BidFields(
            title=ExtractedField[str](normalized_value="RFP Title", status=FieldStatus.found, evidence=ev)
        )
        summary = ExtractionSummary(total_fields=20, found_fields=1, missing_fields=19)
        record = BidRecord(bid_id="BID-123", fields=fields, summary=summary)
        self.assertEqual(record.bid_id, "BID-123")
        self.assertEqual(record.summary.missing_fields, 19)
        
    def test_list_types_preserved(self):
        # Verify list fields can hold lists
        fields = BidFields(
            additional_documentation=ExtractedField[List[str]](
                normalized_value=["Doc A", "Doc B"], status=FieldStatus.found, evidence=Evidence(chunk_ids=["1"])
            )
        )
        self.assertEqual(len(fields.additional_documentation.normalized_value), 2)

if __name__ == '__main__':
    unittest.main()
