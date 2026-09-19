import os
import sys
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.chunker import DocumentChunker

class TestDocumentChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(max_chunk_size=100, overlap=20)
        
        self.mock_pdf_data = {
            "metadata": {
                "source_filename": "test.pdf",
                "document_type": "rfp_main",
                "source_format": "pdf",
                "section_count": 2
            },
            "sections": [
                {
                    "section_identifier": "Page 1",
                    "text": "This is a short page."
                },
                {
                    "section_identifier": "Page 2",
                    "text": "This is a much longer page. " * 10  # 280 characters
                }
            ]
        }
        
        self.mock_html_data = {
            "metadata": {
                "source_filename": "test.html",
                "document_type": "addendum",
                "source_format": "html",
                "section_count": 1
            },
            "sections": [
                {
                    "section_identifier": "Main Content",
                    "text": "HTML content " * 15 # 195 characters
                }
            ]
        }

    def test_chunk_small_section(self):
        chunks = self.chunker.chunk_document(self.mock_pdf_data)
        
        # Page 1 and Page 3 are small, so they should be combined if they are sequential.
        # But in mock_pdf_data, they are separated by Page 2 (which is large).
        # Let's check Page 1 is one chunk and Page 3 is one chunk.
        page_1_chunks = [c for c in chunks if "Page 1" in c["section_identifier"]]
        self.assertEqual(len(page_1_chunks), 1)
        self.assertEqual(page_1_chunks[0]["text"], "This is a short page.")
        
    def test_combine_small_sections(self):
        # Create data with consecutive small sections
        data = {
            "metadata": {
                "source_filename": "test.pdf",
                "document_type": "rfp_main",
                "source_format": "pdf",
                "section_count": 3
            },
            "sections": [
                {"section_identifier": "Page 1", "text": "Small page one."},
                {"section_identifier": "Page 2", "text": "Small page two."},
                {"section_identifier": "Page 3", "text": "Small page three."}
            ]
        }
        
        # Max chunk size 100 is large enough to hold all three (around 50 chars total)
        chunks = self.chunker.chunk_document(data)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_identifier"], "Page 1 & Page 2 & Page 3")
        self.assertIn("Small page one.\n\nSmall page two.\n\nSmall page three.", chunks[0]["text"])
        self.assertEqual(chunks[0]["page_numbers"], [1, 2, 3])

    def test_chunk_large_section(self):
        chunks = self.chunker.chunk_document(self.mock_pdf_data)
        
        page_2_chunks = [c for c in chunks if c["section_identifier"] == "Page 2"]
        self.assertGreater(len(page_2_chunks), 1)
        
        # Verify chunks don't exceed max_chunk_size by much (allowing for whitespace adjustment)
        for chunk in page_2_chunks:
            self.assertLessEqual(len(chunk["text"]), self.chunker.max_chunk_size + 50)
            
    def test_metadata_preservation(self):
        chunks = self.chunker.chunk_document(self.mock_pdf_data)
        
        for chunk in chunks:
            self.assertEqual(chunk["source_filename"], "test.pdf")
            self.assertEqual(chunk["document_type"], "rfp_main")
            self.assertEqual(chunk["source_format"], "pdf")
            self.assertIn("chunk_id", chunk)
            self.assertIsNotNone(chunk["chunk_id"])
            
    def test_page_number_extraction(self):
        chunks = self.chunker.chunk_document(self.mock_pdf_data)
        
        page_1_chunks = [c for c in chunks if c["section_identifier"] == "Page 1"]
        self.assertEqual(page_1_chunks[0]["page_numbers"], [1])
        
        page_2_chunks = [c for c in chunks if c["section_identifier"] == "Page 2"]
        self.assertEqual(page_2_chunks[0]["page_numbers"], [2])

    def test_html_chunking(self):
        chunks = self.chunker.chunk_document(self.mock_html_data)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["source_filename"], "test.html")
            self.assertEqual(chunk["source_format"], "html")
            self.assertEqual(chunk["section_identifier"], "Main Content")
            self.assertNotIn("page_numbers", chunk) # HTML shouldn't extract page number

if __name__ == '__main__':
    unittest.main()
