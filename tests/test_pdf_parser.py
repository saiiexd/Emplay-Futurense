import os
import sys
import unittest
import tempfile
import json
import fitz  # PyMuPDF

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsers.pdf_parser import PDFParser

class TestPDFParser(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "output")
        self.input_dir = os.path.join(self.test_dir.name, "input")
        os.makedirs(self.output_dir)
        os.makedirs(self.input_dir)
        
        self.parser = PDFParser(output_dir=self.output_dir)
        
        # Create a dummy PDF for testing
        self.dummy_pdf_path = os.path.join(self.input_dir, "test_rfp_addendum.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "This is a dummy addendum document.")
        doc.save(self.dummy_pdf_path)
        doc.close()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_infer_document_type(self):
        self.assertEqual(self.parser._infer_document_type("RFP_Document.pdf"), "rfp_main")
        self.assertEqual(self.parser._infer_document_type("Update_Addendum_1.pdf"), "addendum")
        self.assertEqual(self.parser._infer_document_type("Signed_Affidavit.pdf"), "affidavit")

    def test_parse_document(self):
        result = self.parser.parse_document(self.dummy_pdf_path)
        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["source_filename"], "test_rfp_addendum.pdf")
        self.assertEqual(result["metadata"]["document_type"], "addendum")
        self.assertEqual(result["metadata"]["source_format"], "pdf")
        self.assertEqual(result["metadata"]["section_count"], 1)
        self.assertEqual(len(result["sections"]), 1)
        self.assertIn("dummy addendum document", result["sections"][0]["text"])
        self.assertEqual(result["sections"][0]["section_identifier"], "Page 1")

    def test_process_directory(self):
        self.parser.process_directory(self.input_dir)
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 1)
        
        with open(os.path.join(self.output_dir, output_files[0]), 'r') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["source_filename"], "test_rfp_addendum.pdf")

if __name__ == '__main__':
    unittest.main()
