import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.components import highlight_text

class TestUIHelpers(unittest.TestCase):
    def test_highlight_text(self):
        text = "This is a simple test document."
        
        # Test basic match
        highlighted = highlight_text(text, "simple")
        self.assertEqual(highlighted, "This is a **simple** test document.")
        
        # Test case-insensitive match
        highlighted = highlight_text(text, "TEST")
        self.assertEqual(highlighted, "This is a simple **test** document.")
        
        # Test no match
        highlighted = highlight_text(text, "missing")
        self.assertEqual(highlighted, text)
        
        # Test empty query
        highlighted = highlight_text(text, "")
        self.assertEqual(highlighted, text)
        
        # Test special characters in query (regex injection prevention)
        text_with_chars = "Cost is $50.00 total."
        highlighted = highlight_text(text_with_chars, "$50.00")
        self.assertEqual(highlighted, "Cost is **$50.00** total.")

if __name__ == '__main__':
    unittest.main()
