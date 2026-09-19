import os
import sys
import unittest
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsers.html_parser import HTMLParser

class TestHTMLParser(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "output")
        self.input_dir = os.path.join(self.test_dir.name, "input")
        os.makedirs(self.output_dir)
        os.makedirs(self.input_dir)
        
        self.parser = HTMLParser(output_dir=self.output_dir)
        
        # Create a dummy HTML for testing
        self.dummy_html_path = os.path.join(self.input_dir, "test_rfp_main.html")
        html_content = """
        <html>
            <head>
                <title>Test Title</title>
                <style> body { color: red; } </style>
                <script> console.log("Hello"); </script>
            </head>
            <body>
                <header>Navigation Here</header>
                <div class="content">
                    <h1>This is a dummy RFP document.</h1>
                    <p>Important details here.</p>
                </div>
                <footer>Footer Here</footer>
            </body>
        </html>
        """
        with open(self.dummy_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_infer_document_type(self):
        self.assertEqual(self.parser._infer_document_type("RFP_Document.html"), "portal_listing")
        self.assertEqual(self.parser._infer_document_type("Update_Addendum_1.html"), "addendum")
        self.assertEqual(self.parser._infer_document_type("Signed_Affidavit.html"), "affidavit")

    def test_parse_document(self):
        result = self.parser.parse_document(self.dummy_html_path)
        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["source_filename"], "test_rfp_main.html")
        self.assertEqual(result["metadata"]["document_type"], "portal_listing")
        self.assertEqual(result["metadata"]["source_format"], "html")
        self.assertEqual(result["metadata"]["section_count"], 1)
        self.assertEqual(len(result["sections"]), 1)
        
        text = result["sections"][0]["text"]
        
        # Check meaningful text is extracted
        self.assertIn("This is a dummy RFP document.", text)
        self.assertIn("Important details here.", text)
        
        # Check irrelevant elements are removed
        self.assertNotIn("console.log", text)
        self.assertNotIn("color: red", text)
        self.assertNotIn("Navigation Here", text)
        self.assertNotIn("Footer Here", text)

    def test_process_directory(self):
        self.parser.process_directory(self.input_dir)
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 1)
        
        with open(os.path.join(self.output_dir, output_files[0]), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(data["metadata"]["source_filename"], "test_rfp_main.html")

if __name__ == '__main__':
    unittest.main()
