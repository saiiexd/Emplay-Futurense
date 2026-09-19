import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validation.canonical import find_canonical_match, standardize_punctuation

class TestCanonicalMatching(unittest.TestCase):
    def test_l1_exact_match(self):
        text = "Contact thawkins@treasurer.state.md.us for details."
        quote = "thawkins@treasurer.state.md.us"
        res = find_canonical_match(quote, text)
        self.assertTrue(res.matched)
        self.assertEqual(res.level, "L1")
        self.assertEqual(res.original_substring, quote)

    def test_l2_canonical_match(self):
        # Different quote styles
        text = "The title is \u201CProject Alpha\u201D"
        quote = 'The title is "Project Alpha"'
        res = find_canonical_match(quote, text)
        self.assertTrue(res.matched)
        self.assertEqual(res.level, "L2")

    def test_l3_case_folded_match(self):
        text = "Please submit the PROPOSAL by Friday."
        quote = "please submit the proposal"
        res = find_canonical_match(quote, text)
        self.assertTrue(res.matched)
        self.assertEqual(res.level, "L3")
        self.assertEqual(res.original_substring, "Please submit the PROPOSAL")

    def test_l4_compact_match_line_broken_email(self):
        text = "Contact thawkins@treasurer.state.md\n.us for details."
        quote = "thawkins@treasurer.state.md.us"
        res = find_canonical_match(quote, text)
        self.assertTrue(res.matched)
        self.assertEqual(res.level, "L4")
        self.assertEqual(res.original_substring, "thawkins@treasurer.state.md\n.us")

    def test_rejection_of_fuzzy_matching(self):
        text = "The deadline is October 15th."
        # Levenshtein distance is small but we shouldn't match it
        quote = "The dead line is October 16th."
        res = find_canonical_match(quote, text)
        self.assertFalse(res.matched)
        self.assertEqual(res.level, "none")

if __name__ == '__main__':
    unittest.main()
