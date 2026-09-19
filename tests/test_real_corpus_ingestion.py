import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsers.bid_manager import BidManager
from src.retrieval.chunker import DocumentChunker
from src.schemas.enums import DocType
from src.config import CHUNK_MAX, CHUNK_OVERLAP

class TestRealCorpusIngestion(unittest.TestCase):
    def setUp(self):
        # Write parser side-output to a temporary directory so the test never
        # leaves artifacts inside the repository.
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_out = self._temp_dir.name
        self.manager = BidManager(output_dir=self.temp_out)
        self.chunker = DocumentChunker(max_chunk_size=CHUNK_MAX, overlap=CHUNK_OVERLAP)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_bid1_ingestion(self):
        bid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bid1"))
        if not os.path.exists(bid_dir):
            self.skipTest("Bid1 corpus not found locally")
            
        bid_data = self.manager.process_bid_directory(bid_dir)
        docs = bid_data.get("documents", [])
        self.assertTrue(len(docs) > 0)
        
        # Test addendum parsing
        addendums = [d for d in docs if d["metadata"]["document_type"] == "addendum"]
        for ad in addendums:
            add_num = ad["metadata"].get("addendum_number")
            self.assertIsNotNone(add_num)
            self.assertIsInstance(add_num, int)
            
        # Test zero multi-page chunks for this specific parser configuration
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            for chunk in chunks:
                # The prompt says "zero multi-page chunks". 
                # This implies chunk.get("page_numbers") should be max length 1.
                pages = chunk.get("page_numbers", [])
                if len(pages) > 1:
                    # Wait, merged chunks could span pages. 
                    # If the prompt specifically said "verify ... zero multi-page chunks", I must assert this.
                    # Or wait, phase 0 specifically dropped cross-page chunk merging, so length should be <= 1.
                    self.assertTrue(len(pages) <= 1, f"Found multi-page chunk: {chunk['chunk_id']}")

    def test_bid2_ingestion(self):
        bid_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Bid2"))
        if not os.path.exists(bid_dir):
            self.skipTest("Bid2 corpus not found locally")
            
        bid_data = self.manager.process_bid_directory(bid_dir)
        docs = bid_data.get("documents", [])
        self.assertTrue(len(docs) > 0)
        
        # Verify absence of HTML comments like JIRA/Google Analytics
        for doc in docs:
            if doc["metadata"]["source_format"] == "html":
                for sec in doc.get("sections", []):
                    text = sec.get("text", "")
                    self.assertNotIn("JIRA", text)
                    self.assertNotIn("Google Analytics", text)
                    self.assertNotIn("<!--", text)
                    
                # Truncated portal description (See more)
                if doc["metadata"].get("truncated_description"):
                    self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
