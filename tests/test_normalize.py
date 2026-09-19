import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validation.normalize import (
    normalize_identifier,
    normalize_date,
    normalize_contact,
    normalize_list,
    normalize_value
)

class TestNormalize(unittest.TestCase):
    def test_normalize_identifier(self):
        # Space and punctuation removal
        norm = normalize_identifier("JA-207652")
        self.assertEqual(norm.value, "JA207652")
        norm2 = normalize_identifier("JA 207652")
        self.assertEqual(norm2.value, "JA207652")
        self.assertEqual(norm.value, norm2.value)

    def test_normalize_date_timezone_conversion(self):
        # CST/CDT mapped to America/Chicago
        norm = normalize_date("October 15, 2025 at 2:00 PM CST")
        self.assertIn("America/Chicago", norm.value)
        self.assertNotIn("CST", norm.value)
        
        # EST/EDT mapped to America/New_York
        norm2 = normalize_date("October 15, 2025 at 2:00 PM EDT")
        self.assertIn("America/New_York", norm2.value)

    def test_normalize_date_relative(self):
        norm = normalize_date("Delivery within 45 days of Award")
        self.assertEqual(norm.data_type, "relative_date")
        self.assertEqual(norm.value, "Delivery within 45 days of Award")

    def test_normalize_date_date_only(self):
        # Leaves date-only dates without fabricated times
        norm = normalize_date("2025-10-15")
        self.assertEqual(norm.value, "2025-10-15")
        self.assertNotIn("00:00:00", norm.value) # no fabricated time

    def test_normalize_contact(self):
        contact = {"name": "T Hawkins", "email": "thawkins@treasurer.state.md\n.us", "phone": "555 - 1234"}
        norm = normalize_contact(contact)
        self.assertEqual(norm.value["email"], "thawkins@treasurer.state.md.us")
        self.assertEqual(norm.value["phone"], "555-1234")

    def test_normalize_list(self):
        items = [" Product A ", "Product A", "product b"]
        norm = normalize_list(items)
        self.assertEqual(len(norm.value), 2)
        self.assertIn("Product A", norm.value)
        self.assertIn("product b", norm.value)

    def test_normalize_value_dispatch(self):
        norm = normalize_value("JA-123", "identifier")
        self.assertEqual(norm.data_type, "identifier")
        
        norm_list = normalize_value("Single item", "list")
        self.assertEqual(norm_list.data_type, "list")
        self.assertEqual(norm_list.value, ["Single item"])

if __name__ == '__main__':
    unittest.main()
