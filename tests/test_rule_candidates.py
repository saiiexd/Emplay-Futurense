import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.registry import DocumentRegistry
from src.extraction.rule_candidates import RuleCandidateGenerator

class TestRuleCandidates(unittest.TestCase):
    def setUp(self):
        self.registry = DocumentRegistry()
        self.generator = RuleCandidateGenerator(self.registry)
        
    def test_mercury_and_contract_affidavit(self):
        # Setup an eligible document (affidavit)
        doc_data = {
            "metadata": {
                "doc_id": "doc1",
                "document_type": "affidavit"
            },
            "sections": []
        }
        
        chunks = [
            {"chunk_id": "c1", "text": "Please sign the Mercury Affidavit."},
            {"chunk_id": "c2", "text": "Attached is the Contract Affidavit."},
            {"chunk_id": "c3", "text": "Please sign this affidavit for our records."} # Should not match Contract Affidavit
        ]
        
        self.registry.register_document(doc_data, chunks)
        
        # Test additional_documentation
        candidates = self.generator.generate_candidates("additional_documentation")
        
        # Expect 2 matches
        self.assertEqual(len(candidates), 2)
        
        values = [c.raw_value[0] for c in candidates]
        self.assertIn("Mercury Affidavit", values)
        self.assertIn("Contract Affidavit", values)
        self.assertNotIn("this affidavit", values)
        
    def test_ineligible_doc_type_for_rule(self):
        # Setup an ineligible document (e.g. portal_listing)
        doc_data = {
            "metadata": {
                "doc_id": "doc2",
                "document_type": "portal_listing"
            },
            "sections": []
        }
        
        chunks = [
            {"chunk_id": "c4", "text": "Mercury Affidavit"}
        ]
        
        self.registry.register_document(doc_data, chunks)
        
        # Test additional_documentation
        candidates = self.generator.generate_candidates("additional_documentation")
        
        # Should not extract because portal_listing may not be eligible for additional_documentation
        self.assertEqual(len(candidates), 0)

    def test_non_documentation_fields(self):
        # Rule should only run for additional_documentation
        candidates = self.generator.generate_candidates("company_name")
        self.assertEqual(len(candidates), 0)

    def test_connectors_and_determiners_are_not_form_names(self):
        """Regression: "certificate or Affidavit" is prose, not a form title."""
        doc_data = {"metadata": {"doc_id": "doc3", "document_type": "affidavit"}, "sections": []}
        chunks = [
            {"chunk_id": "c5", "text": "Provide a warranty certificate or Affidavit upon award."},
            {"chunk_id": "c6", "text": "Submit the Proposal Affidavit with the response."},
        ]
        self.registry.register_document(doc_data, chunks)

        values = [c.raw_value[0] for c in self.generator.generate_candidates("additional_documentation")]
        self.assertIn("Proposal Affidavit", values)
        for value in values:
            self.assertNotIn("or", value.split())

    def test_evidence_is_an_exact_substring_of_its_chunk(self):
        """Rule candidates must be verifiable against source text like any other."""
        from src.validation.canonical import find_canonical_match

        doc_data = {"metadata": {"doc_id": "doc4", "document_type": "affidavit"}, "sections": []}
        self.registry.register_document(
            doc_data, [{"chunk_id": "c7", "text": "The Disclosure Affidavit must be notarized."}]
        )
        candidates = self.generator.generate_candidates("additional_documentation")
        target = [c for c in candidates if c.chunk_ids == ["c7"]]
        self.assertEqual(len(target), 1)

        chunk_text = self.registry.chunk_metadata["c7"]["text"]
        self.assertTrue(find_canonical_match(target[0].excerpts[0], chunk_text).matched)

if __name__ == '__main__':
    unittest.main()
