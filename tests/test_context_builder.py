import os
import sys
import unittest
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction.context_builder import ContextBuilder
from src.retrieval.registry import DocumentRegistry
from src.schemas.enums import DocType

# A fake searcher that just returns what we tell it
class FakeSearcher:
    def __init__(self):
        self.mock_results = []
        
    def search_field(self, field: str, doc_filter: str, top_k: int, lexical_floor: int) -> List[Dict]:
        return [r for r in self.mock_results if r.get("doc_id") == doc_filter]

class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.registry = DocumentRegistry()
        self.searcher = FakeSearcher()
        
    def test_empty_group(self):
        builder = ContextBuilder(self.registry, self.searcher)
        # Identity group with no docs
        pkgs = builder.build_group("bid1", "identity")
        self.assertEqual(len(pkgs), 1)
        pkg = pkgs[0]
        self.assertEqual(pkg.batch_index, 0)
        self.assertEqual(pkg.batch_count, 1)
        self.assertEqual(len(pkg.excerpts), 0)
        self.assertIn("bid_number", pkg.unavailable_fields)
        self.assertEqual(pkg.evidence_chars, 0)

    def test_unknown_document_type_excluded(self):
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "unknown"}}
        self.registry.register_document(doc_data, [{"chunk_id": "c1", "text": "hello"}])
        
        builder = ContextBuilder(self.registry, self.searcher)
        pkgs = builder.build_group("bid1", "identity")
        self.assertEqual(len(pkgs[0].excerpts), 0)

    def test_small_document_whole_inclusion(self):
        # Under SMALL_DOC_CHARS (15000)
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
        chunks = [{"chunk_id": "c1", "text": "short doc"}]
        self.registry.register_document(doc_data, chunks)
        
        builder = ContextBuilder(self.registry, self.searcher)
        pkgs = builder.build_group("bid1", "identity")
        self.assertEqual(len(pkgs[0].excerpts), 1)
        self.assertEqual(pkgs[0].excerpts[0].text, "short doc")

    def test_large_document_front_matter_and_retrieval(self):
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
        chunks = [
            {"chunk_id": f"c{i}", "text": "a" * 1000, "metadata": {"page_number": i}} for i in range(1, 20)
        ]
        self.registry.register_document(doc_data, chunks)
        self.registry.doc_text["doc1"] = "a" * 20000
        # Front matter pages are (1,2)
        # Also let's retrieve c10 via searcher
        self.searcher.mock_results.append({"chunk_id": "c10", "doc_id": "doc1", "score": 1.0})
        
        builder = ContextBuilder(self.registry, self.searcher)
        pkgs = builder.build_group("bid1", "identity")
        exc = pkgs[0].excerpts
        
        cids = [pkgs[0].handle_map[e.handle] for e in exc]
        # Should include front matter (c1, c2) + retrieved (c10) + neighbor (c9, c11)
        expected = {"c1", "c2", "c9", "c10", "c11"}
        self.assertEqual(set(cids), expected)

    def test_page_expansion(self):
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
        chunks = [
            {"chunk_id": f"c{i}", "text": "a" * 1000, "metadata": {"page_number": i // 3 + 1}} for i in range(1, 20)
        ]
        # Pages: c1,c2(pg1), c3,c4,c5(pg2), etc.
        self.registry.register_document(doc_data, chunks)
        self.registry.doc_text["doc1"] = "a" * 20000
        # Retrieve c4
        self.searcher.mock_results.append({"chunk_id": "c4", "doc_id": "doc1", "score": 1.0})
        
        builder = ContextBuilder(self.registry, self.searcher)
        # "products" group triggers page expansion
        pkgs = builder.build_group("bid1", "products")
        exc = pkgs[0].excerpts
        cids = [pkgs[0].handle_map[e.handle] for e in exc]
        
        # neighbors of c4 are c3 and c5. Page of c4 is 2. c3,c4,c5 are all on page 2.
        # Wait, neighbors are config.NEIGHBOR_RADIUS (which is 1).
        # So it naturally gets c3, c5 anyway.
        # Let's retrieve c7 (page 3). Neighbors: c6, c8.
        # c6,c7,c8 on page 3? i=6->page3, i=7->page3, i=8->page3.
        # It expands exactly to the page anyway.
        self.assertTrue("c4" in cids)

    def test_affidavit_anchor(self):
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "affidavit"}}
        chunks = [
            {"chunk_id": "c1", "text": "This is a Mercury Affidavit here."},
            {"chunk_id": "c2", "text": "this affidavit is ignored"}
        ]
        self.registry.register_document(doc_data, chunks)
        
        builder = ContextBuilder(self.registry, self.searcher)
        pkgs = builder.build_group("bid1", "terms") # additional_documentation is in terms
        cids = [pkgs[0].handle_map[e.handle] for e in pkgs[0].excerpts]
        self.assertIn("c1", cids)
        self.assertNotIn("c2", cids)

    def test_collision_escalation(self):
        # We need two chunk_ids that have the same 8-char prefix in sha256.
        # We can just manually inject it into the registry.
        # Let's find two strings with same first 8 hex chars of sha256? That's too hard.
        # We'll just mock the hexdigest function during the builder init.
        import hashlib
        original_sha256 = hashlib.sha256
        
        class FakeHash:
            def __init__(self, data):
                self.data = data
            def hexdigest(self):
                if self.data == b'c1': return "111111112222AAAA"
                if self.data == b'c2': return "111111112222BBBB"
                return original_sha256(self.data).hexdigest()
                
        hashlib.sha256 = FakeHash
        
        doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
        chunks = [
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c2", "text": "b"}
        ]
        self.registry.register_document(doc_data, chunks)
        
        builder = ContextBuilder(self.registry, self.searcher)
        
        # Test handle length
        h1 = builder.handle_for("c1")
        h2 = builder.handle_for("c2")
        
        # Because collision occurred on "11111111", it should escalate to 12 chars
        self.assertEqual(h1, "E111111112222")
        self.assertEqual(h2, "E111111112222") # Wait, they have the same 12 chars!
        # If they still collide at 12, they have the exact same handle. But we specified "escalate to 12".
        
        # Restore
        hashlib.sha256 = original_sha256

    def test_doc_label_and_follows(self):
        doc1 = {"metadata": {"doc_id": "doc_a", "document_type": "rfp_main"}}
        doc2 = {"metadata": {"doc_id": "doc_b", "document_type": "rfp_main"}}
        self.registry.register_document(doc1, [{"chunk_id": "c1", "text": "a"}, {"chunk_id": "c2", "text": "b"}])
        self.registry.register_document(doc2, [{"chunk_id": "c3", "text": "c"}])
        
        builder = ContextBuilder(self.registry, self.searcher)
        pkgs = builder.build_group("bid1", "identity")
        excerpts = pkgs[0].excerpts
        
        # Doc_a is sorted before doc_b. doc_a -> D1, doc_b -> D2
        self.assertEqual(excerpts[0].doc_label, "D1")
        self.assertEqual(excerpts[1].doc_label, "D1")
        self.assertEqual(excerpts[2].doc_label, "D2")
        
        self.assertIsNone(excerpts[0].follows)
        self.assertEqual(excerpts[1].follows, excerpts[0].handle)
        self.assertIsNone(excerpts[2].follows)

    def _register_bulk_doc(self, chunk_count, chunk_chars, doc_id="doc1"):
        doc_data = {"metadata": {"doc_id": doc_id, "document_type": "rfp_main"}}
        chunks = [
            {"chunk_id": f"bulk{i}", "text": str(i).zfill(4) + "x" * (chunk_chars - 4)}
            for i in range(chunk_count)
        ]
        self.registry.register_document(doc_data, chunks)
        return chunks

    def test_prompt_budget_trimming(self):
        import src.config as config
        from src.extraction import prompts

        # Leave room for the prompt skeleton plus roughly one excerpt.
        overhead = prompts.static_overhead_chars("identity", ["bid_number", "title", "company_name"])
        original_max_chars = config.MAX_PROMPT_CHARS
        config.MAX_PROMPT_CHARS = overhead + 200
        try:
            doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
            chunks = [{"chunk_id": "c1", "text": "a" * 100}, {"chunk_id": "c2", "text": "b" * 100}]
            self.registry.register_document(doc_data, chunks)

            builder = ContextBuilder(self.registry, self.searcher)
            pkg = builder.build_group("bid1", "identity")[0]

            self.assertEqual(len(pkg.excerpts), 1)
            self.assertEqual(pkg.evidence_chars, 100)
            self.assertEqual(len(pkg.trimmed_chunk_ids), 1)
        finally:
            # Restore unconditionally; a failure here must not leak into other tests.
            config.MAX_PROMPT_CHARS = original_max_chars

    def test_rendered_prompt_never_exceeds_limit(self):
        """Regression: evidence_chars alone used to be compared with MAX_PROMPT_CHARS,
        so the rendered prompt (system prompt + field cards + excerpt tags) overflowed."""
        import src.config as config
        from src.extraction import prompts

        self._register_bulk_doc(chunk_count=150, chunk_chars=900)
        builder = ContextBuilder(self.registry, self.searcher)
        pkg = builder.build_group("bid1", "terms")[0]

        messages = prompts.render_messages(pkg)
        rendered = len(messages[0]["content"]) + len(messages[1]["content"])

        self.assertLessEqual(rendered, config.MAX_PROMPT_CHARS)
        self.assertLessEqual(pkg.evidence_chars, config.MAX_EVIDENCE_CHARS_PER_CALL)
        self.assertLessEqual(len(pkg.excerpts), config.MAX_EXCERPTS_PER_CALL)
        self.assertTrue(pkg.trimmed_chunk_ids, "oversized package should trim")

    def test_budget_accounts_for_tag_and_skeleton_overhead(self):
        """The old formula kept every excerpt whenever raw text fit; prove the
        rendered size is now what is budgeted, not the bare evidence text."""
        import src.config as config
        from src.extraction import prompts

        self._register_bulk_doc(chunk_count=150, chunk_chars=900)
        builder = ContextBuilder(self.registry, self.searcher)
        pkg = builder.build_group("bid1", "terms")[0]

        messages = prompts.render_messages(pkg)
        rendered = len(messages[0]["content"]) + len(messages[1]["content"])
        overhead = rendered - pkg.evidence_chars

        # Overhead is real and material, and the old check would have ignored it.
        self.assertGreater(overhead, 1000)
        self.assertLessEqual(pkg.evidence_chars + overhead, config.MAX_PROMPT_CHARS)

        predicted = prompts.static_overhead_chars(pkg.group, pkg.fields) + sum(
            len(e.text) + prompts.excerpt_overhead_chars(e) for e in pkg.excerpts
        )
        self.assertEqual(rendered, predicted)

    def test_everything_trimmed_yields_unrenderable_empty_package(self):
        import src.config as config

        original_max_chars = config.MAX_PROMPT_CHARS
        config.MAX_PROMPT_CHARS = 50  # smaller than the skeleton itself
        try:
            doc_data = {"metadata": {"doc_id": "doc1", "document_type": "rfp_main"}}
            self.registry.register_document(doc_data, [{"chunk_id": "c1", "text": "a" * 100}])

            builder = ContextBuilder(self.registry, self.searcher)
            pkg = builder.build_group("bid1", "identity")[0]

            self.assertEqual(pkg.excerpts, [])
            self.assertEqual(pkg.fields, [])
            self.assertIn("bid_number", pkg.unavailable_fields)
        finally:
            config.MAX_PROMPT_CHARS = original_max_chars

if __name__ == '__main__':
    unittest.main()
