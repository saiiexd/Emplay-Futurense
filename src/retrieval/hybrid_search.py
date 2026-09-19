import re
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from src.retrieval.embeddings import EmbeddingClient
from src.retrieval.queries import get_field_spec
from src.config import LEXICAL_FLOOR

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def compute_rrf(ranks: List[List[str]], k: int = 60) -> Dict[str, float]:
    rrf_scores = {}
    for rank_list in ranks:
        for idx, chunk_id in enumerate(rank_list):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (k + idx + 1)
    return rrf_scores

class HybridSearcher:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client
        self.chunks: Dict[str, dict] = {}
        self.chunk_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.embeddings: List[List[float]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_indexed = False

    def add_chunks(self, chunks: List[dict]):
        for chunk in chunks:
            if not chunk.get("text") or not chunk.get("chunk_id"):
                continue
            if chunk["chunk_id"] not in self.chunks:
                self.chunks[chunk["chunk_id"]] = chunk
        self._is_indexed = False

    def build_index(self):
        self.chunk_ids = sorted(list(self.chunks.keys()))
        self.corpus_texts = [self.chunks[cid]["text"] for cid in self.chunk_ids]
        
        if not self.corpus_texts:
            self.bm25 = None
            self.embeddings = []
            self._is_indexed = True
            return

        tokenized_corpus = [re.sub(r'[^\w\s]', ' ', text.lower()).split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.embeddings = self.embedding_client.embed_batch(self.corpus_texts)
        self._is_indexed = True

    def _score_bm25(self, query: str) -> Dict[str, float]:
        if not self.bm25:
            return {}
        tokenized_query = re.sub(r'[^\w\s]', ' ', query.lower()).split()
        scores = self.bm25.get_scores(tokenized_query)
        return {cid: float(score) for cid, score in zip(self.chunk_ids, scores) if score > 0}

    def _score_semantic(self, query: str) -> Dict[str, float]:
        if not self.embeddings:
            return {}
        query_emb = self.embedding_client.embed_text(query)
        scores = {}
        for cid, doc_emb in zip(self.chunk_ids, self.embeddings):
            sim = cosine_similarity(query_emb, doc_emb)
            if sim > 0:
                scores[cid] = sim
        return scores

    def search_field(self, field_name: str, doc_filter: Optional[str] = None, top_k: int = 10, lexical_floor: int = LEXICAL_FLOOR) -> List[dict]:
        if not self._is_indexed:
            self.build_index()

        if not self.chunks:
            return []

        field_spec = get_field_spec(field_name)
        if not field_spec:
            return []

        eligible_cids = self.chunk_ids
        if doc_filter:
            eligible_cids = [cid for cid in self.chunk_ids if self.chunks[cid].get("doc_id") == doc_filter or self.chunks[cid].get("source_filename") == doc_filter]

        lexical_rank_lists = []
        for alias in field_spec.bm25_aliases:
            bm25_scores = self._score_bm25(alias)
            valid_cids = [cid for cid in eligible_cids if cid in bm25_scores]
            ranked = sorted(valid_cids, key=lambda k: (-bm25_scores[k], k))
            if ranked:
                lexical_rank_lists.append(ranked)
        
        lexical_rrf_scores = compute_rrf(lexical_rank_lists, k=60)
        lexical_ranked = sorted(lexical_rrf_scores.keys(), key=lambda k: (-lexical_rrf_scores[k], k))

        semantic_scores = self._score_semantic(field_spec.semantic_query)
        valid_semantic_cids = [cid for cid in eligible_cids if cid in semantic_scores]
        semantic_ranked = sorted(valid_semantic_cids, key=lambda k: (-semantic_scores[k], k))

        html_priority_scores = {}
        lower_aliases = [a.lower() for a in field_spec.bm25_aliases]
        for cid in eligible_cids:
            chunk = self.chunks[cid]
            if chunk.get("source_format") == "html":
                section_id = str(chunk.get("section_identifier", "")).strip().lower()
                if len(section_id) >= 3:
                    for alias in lower_aliases:
                        if alias in section_id or section_id in alias:
                            html_priority_scores[cid] = 1.0
                            break
        
        html_ranked = sorted(html_priority_scores.keys(), key=lambda k: (-html_priority_scores[k], k))

        final_rank_lists = [lexical_ranked, semantic_ranked]
        if html_ranked:
            final_rank_lists.append(html_ranked)
            
        final_rrf_scores = compute_rrf(final_rank_lists, k=60)
        
        # Tie-breaker deterministic sort: (-score, cid)
        sorted_results = sorted(final_rrf_scores.keys(), key=lambda k: (-final_rrf_scores[k], k))
        
        selected_cids = sorted_results[:top_k]
        
        # Apply Lexical Floor
        # Append missing lexical top-floor chunks in lexical order
        if lexical_floor > 0 and lexical_ranked:
            for top_lex_cid in lexical_ranked[:lexical_floor]:
                if top_lex_cid not in selected_cids:
                    selected_cids.append(top_lex_cid)
        
        results = []
        for rank, cid in enumerate(selected_cids):
            chunk_data = dict(self.chunks[cid])
            chunk_data["retrieval_metadata"] = {
                "fused_score": final_rrf_scores.get(cid, 0.0),
                "overall_rank": rank + 1,
                "lexical_rank": lexical_ranked.index(cid) + 1 if cid in lexical_ranked else None,
                "semantic_rank": semantic_ranked.index(cid) + 1 if cid in semantic_ranked else None,
                "html_label_hit": cid in html_priority_scores,
                "lexical_floor_inserted": cid not in sorted_results[:top_k]
            }
            results.append(chunk_data)

        return results

    def search_exact_identifier(self, identifier: str, top_k: int = 5) -> List[dict]:
        if not self._is_indexed:
            self.build_index()
            
        bm25_scores = self._score_bm25(identifier)
        sorted_cids = sorted(bm25_scores.keys(), key=lambda k: (-bm25_scores[k], k))
        
        results = []
        for rank, cid in enumerate(sorted_cids[:top_k]):
            chunk_data = dict(self.chunks[cid])
            chunk_data["retrieval_metadata"] = {
                "fused_score": bm25_scores[cid],
                "overall_rank": rank + 1,
                "lexical_rank": rank + 1,
                "semantic_rank": None,
                "html_label_hit": False,
                "lexical_floor_inserted": False
            }
            results.append(chunk_data)
        return results
