import os
import sys
import unittest
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsers.bid_manager import BidManager

class TestBidManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "output")
        self.input_dir = os.path.join(self.test_dir.name, "Bid_Test")
        os.makedirs(self.output_dir)
        os.makedirs(self.input_dir)
        
        self.manager = BidManager(output_dir=self.output_dir)
        
        # Create a dummy HTML for testing
        self.dummy_html_path = os.path.join(self.input_dir, "test.html")
        with open(self.dummy_html_path, 'w', encoding='utf-8') as f:
            f.write("<html><body><p>Test</p></body></html>")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_process_bid_directory(self):
        bid_data = self.manager.process_bid_directory(self.input_dir)
        
        self.assertEqual(bid_data["bid_id"], "Bid_Test")
        self.assertEqual(len(bid_data["documents"]), 1)
        self.assertEqual(bid_data["documents"][0]["metadata"]["source_format"], "html")

if __name__ == '__main__':
    unittest.main()
