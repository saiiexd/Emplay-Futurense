import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.registry import DocumentRegistry

class TestDocumentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = DocumentRegistry()
        
    def test_addendum_provenance_and_doc_text(self):
        doc_data = {
            "metadata": {
                "doc_id": "doc1",
                "source_filename": "Addendum 2 RFP.pdf",
                "document_type": "addendum",
                "addendum_number": 2,
            },
            "sections": [
                {"text": "First section text on page 1.", "metadata": {"page_number": 1}},
                {"text": "Second section text on page 2.", "metadata": {"page_number": 2}}
            ]
        }
        
        chunks = [
            {"chunk_id": "c1", "text": "First section text on page 1.", "page_numbers": [1]},
            {"chunk_id": "c2", "text": "Second section text on page 2.", "page_numbers": [2]}
        ]
        
        self.registry.register_document(doc_data, chunks)
        
        # Test whole document text
        doc_text = self.registry.doc_text.get("doc1")
        self.assertIn("First section text", doc_text)
        self.assertIn("Second section text", doc_text)
        
        # Test provenance
        prov = self.registry.provenance("c2")
        self.assertIsNotNone(prov)
        self.assertEqual(prov.addendum_number, 2)
        self.assertEqual(prov.page_index, 2)
        self.assertEqual(prov.printed_page_label, "2")
        
    def test_quote_aware_page_resolution(self):
        doc_data = {
            "metadata": {
                "doc_id": "doc2",
            },
            "sections": [
                {"text": "Apple on page one.", "metadata": {"page_number": 1}},
                {"text": "Banana on page two.", "metadata": {"page_number": 2}}
            ]
        }
        
        # Simulate a chunk that spans two pages, containing both texts.
        chunks = [
            {"chunk_id": "c3", "text": "Apple on page one. Banana on page two.", "page_numbers": [1, 2]}
        ]
        
        self.registry.register_document(doc_data, chunks)
        
        # Resolve Apple
        pages_apple = self.registry.resolve_page("c3", "Apple")
        self.assertEqual(pages_apple, [1])
        
        # Resolve Banana
        pages_banana = self.registry.resolve_page("c3", "Banana")
        self.assertEqual(pages_banana, [2])
        
        # Resolve general overlapping quote
        pages_overlap = self.registry.resolve_page("c3", "on page")
        self.assertEqual(pages_overlap, [1, 2]) # Cannot uniquely resolve, fallback to all chunk pages

if __name__ == '__main__':
    unittest.main()
