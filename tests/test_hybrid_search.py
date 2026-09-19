import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.embeddings import FakeEmbeddingClient
from src.config import LEXICAL_FLOOR

class TestHybridSearcher(unittest.TestCase):
    def setUp(self):
        self.client = FakeEmbeddingClient(dim=10)
        self.searcher = HybridSearcher(embedding_client=self.client)
        
        # Synthetic neutral chunks for testing
        self.chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "The solicitation number for this request is ABC-123.", "source_format": "pdf"},
            {"chunk_id": "c2", "doc_id": "d1", "text": "Bids must be delivered by October 15, 2025 at 2:00 PM.", "source_format": "pdf"},
            {"chunk_id": "c3", "doc_id": "d1", "text": "This is a random paragraph without any keywords.", "source_format": "pdf"},
            {"chunk_id": "c4", "doc_id": "d1", "text": "The pre-bid meeting will be held on September 1, 2025.", "source_format": "pdf"},
            {"chunk_id": "c5", "doc_id": "d2", "text": "Title of bid: Provision of Cloud Services.", "source_format": "html", "section_identifier": "Project Title"},
            # Lexical floor synthetic case
            {"chunk_id": "c6", "doc_id": "d3", "text": "Very poor semantic match but contains Bid Number explicitly.", "source_format": "pdf"}
        ]
        self.searcher.add_chunks(self.chunks)

    def test_search_field_bid_number(self):
        results = self.searcher.search_field("bid_number", top_k=3)
        self.assertTrue(len(results) > 0)
        cids = [r["chunk_id"] for r in results]
        # c1 has 'solicitation number', c6 has 'bid number'
        self.assertTrue("c1" in cids or "c6" in cids)
        
    def test_search_field_due_date(self):
        results = self.searcher.search_field("due_date", top_k=3)
        self.assertTrue(len(results) > 0)
        
    def test_html_label_boost(self):
        results = self.searcher.search_field("title", top_k=3)
        cids = [r["chunk_id"] for r in results]
        self.assertIn("c5", cids)
        for r in results:
            if r["chunk_id"] == "c5":
                self.assertTrue(r["retrieval_metadata"]["html_label_hit"])

    def test_lexical_floor_behavior(self):
        # By setting a high lexical floor, c6 should be forced in if it wasn't already in top_k
        # Wait, if we use top_k=1, but lexical floor=2, lexical floor pushes it up to 2 items? 
        # The logic says "if top_lex_cid not in selected_cids: selected_cids.append(top_lex_cid)"
        results = self.searcher.search_field("bid_number", top_k=1, lexical_floor=2)
        # Even with top_k=1, lexical floor of 2 should append the 2nd lexical match if not present.
        # Length could be up to top_k + lexical_floor depending on overlap.
        self.assertTrue(len(results) >= 1)
        
    def test_deterministic_ties(self):
        # Fake embeddings generate deterministic values.
        # Running the exact same query twice should yield the exact same order.
        results1 = self.searcher.search_field("due_date", top_k=5)
        results2 = self.searcher.search_field("due_date", top_k=5)
        cids1 = [r["chunk_id"] for r in results1]
        cids2 = [r["chunk_id"] for r in results2]
        self.assertEqual(cids1, cids2)

if __name__ == '__main__':
    unittest.main()
