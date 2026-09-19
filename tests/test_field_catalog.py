import os
import sys
import unittest
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.schemas.field_catalog import FIELD_CATALOG
from src.schemas.enums import GroupName

class TestFieldCatalog(unittest.TestCase):
    def test_exact_20_fields_and_order(self):
        expected_order = [
            "bid_number", "title", "due_date", "bid_submission_type", "term_of_bid", 
            "pre_bid_meeting", "installation", "bid_bond_requirement", "delivery_date", 
            "payment_terms", "additional_documentation", "mfg_for_registration", 
            "contract_or_cooperative", "model_no", "part_no", "product", "contact_info", 
            "company_name", "bid_summary", "product_specification"
        ]
        
        actual_keys = list(FIELD_CATALOG.keys())
        self.assertEqual(len(actual_keys), 20)
        self.assertEqual(actual_keys, expected_order)

    def test_assignment_labels(self):
        expected_labels = {
            "bid_number": "Bid Number",
            "title": "Title",
            "due_date": "Due Date",
            "bid_submission_type": "Bid Submission Type",
            "term_of_bid": "Term of Bid",
            "pre_bid_meeting": "Pre Bid Meeting",
            "installation": "Installation",
            "bid_bond_requirement": "Bid Bond Requirement",
            "delivery_date": "Delivery Date",
            "payment_terms": "Payment Terms",
            "additional_documentation": "Any Additional Documentation Required",
            "mfg_for_registration": "MFG for Registration",
            "contract_or_cooperative": "Contract or Cooperative to use",
            "model_no": "Model_no",
            "part_no": "Part_no",
            "product": "Product",
            "contact_info": "contact_info",
            "company_name": "company_name",
            "bid_summary": "Bid Summary",
            "product_specification": "Product Specification"
        }
        for key, expected_label in expected_labels.items():
            self.assertEqual(FIELD_CATALOG[key].assignment_label, expected_label)

    def test_group_membership(self):
        # Just ensure they use the enum values
        for key, spec in FIELD_CATALOG.items():
            self.assertTrue(isinstance(spec.group, GroupName))
            self.assertTrue(spec.group in GroupName)

    def test_document_eligibility(self):
        for key, spec in FIELD_CATALOG.items():
            self.assertTrue(len(spec.eligible_document_types) > 0)
            
    def test_compilable_anchors(self):
        for key, spec in FIELD_CATALOG.items():
            for anchor in spec.anchors:
                # Should compile as regex without crashing
                re.compile(anchor, re.IGNORECASE)

    def test_absence_of_banned_words(self):
        banned_words = [
            "prefer", "latest", "newest", "supersede", "override", 
            "precedence", "priority", "authoritative", "official", 
            "primary", "rank", "score", "tier"
        ]
        
        # Golden literals that shouldn't appear directly in prompts
        golden_literals = ["July 9", "JA-207652", "CC7802", "WD22TB4", "LENGTH OF CONTRACT", "Latitude 5550"]

        for key, spec in FIELD_CATALOG.items():
            prompt_texts = [
                spec.definition.lower(),
                spec.include.lower(),
                spec.exclude.lower(),
                spec.semantic_query.lower()
            ]
            
            for text in prompt_texts:
                for banned in banned_words:
                    self.assertNotIn(f" {banned} ", f" {text} ", f"Found banned word '{banned}' in {key}")
                
                for golden in golden_literals:
                    self.assertNotIn(golden.lower(), text, f"Found golden literal '{golden}' in {key}")

if __name__ == '__main__':
    unittest.main()
