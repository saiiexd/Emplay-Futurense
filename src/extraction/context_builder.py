import hashlib
from typing import List, Dict, Set, Optional
from collections import defaultdict

from src.schemas.candidates import BaseForbid
from src.schemas.enums import DocType
from src.schemas.field_catalog import FIELD_CATALOG
from src.retrieval.registry import DocumentRegistry
from src.retrieval.hybrid_search import HybridSearcher
from src.extraction import prompts
from src.extraction.rule_candidates import NAMED_FORM_PATTERN
import src.config as config

class EvidenceExcerpt(BaseForbid):
    handle: str
    doc_label: str
    follows: Optional[str] = None
    eligible_fields: List[str]
    text: str

class ContextPackage(BaseForbid):
    bid_id: str
    group: str
    batch_index: int
    batch_count: int
    fields: List[str]
    unavailable_fields: List[str]
    excerpts: List[EvidenceExcerpt]
    handle_map: Dict[str, str]
    evidence_chars: int
    trimmed_chunk_ids: List[str]
    max_output_tokens: int

class ContextBuilder:
    def __init__(self, registry: DocumentRegistry, searcher: HybridSearcher):
        self.registry = registry
        self.searcher = searcher
        self._doc_label_map: Dict[str, str] = {}
        self._handle_map: Dict[str, str] = {}
        
        doc_ids = sorted(self.registry.doc_refs.keys())
        for i, doc_id in enumerate(doc_ids, start=1):
            self._doc_label_map[doc_id] = f"D{i}"
            
        hash_to_chunks = defaultdict(list)
        for chunk_id in self.registry.chunk_metadata:
            sha = hashlib.sha256(chunk_id.encode('utf-8')).hexdigest()
            short_hash = sha[:8]
            hash_to_chunks[short_hash].append((chunk_id, sha))
            
        for short_hash, chunks in hash_to_chunks.items():
            if len(chunks) > 1:
                for chunk_id, sha in chunks:
                    self._handle_map[chunk_id] = "E" + sha[:12]
            else:
                self._handle_map[chunks[0][0]] = "E" + short_hash
                
        # Shared with the rule-candidate generator so both layers agree on what
        # counts as a named form.
        self._affidavit_regex = NAMED_FORM_PATTERN

    def doc_label_for(self, doc_id: str) -> str:
        return self._doc_label_map.get(doc_id, "D?")

    def handle_for(self, chunk_id: str) -> str:
        return self._handle_map.get(chunk_id, "")

    def build_all(self, bid_id: str) -> Dict[str, List[ContextPackage]]:
        return {g: self.build_group(bid_id, g) for g in config.GROUP_ORDER}

    def build_group(self, bid_id: str, group: str) -> List[ContextPackage]:
        F_g = [k for k in FIELD_CATALOG.keys() if FIELD_CATALOG[k].group.value == group]
        
        selected_chunk_ids: Set[str] = set()
        chunk_reasons: Dict[str, Set[str]] = defaultdict(set)
        
        doc_ids = sorted(self.registry.doc_refs.keys())
        
        for doc_id in doc_ids:
            doc_ref = self.registry.get_document_ref(doc_id)
            if not doc_ref or doc_ref.document_type == DocType.unknown:
                continue
                
            E_d = [f for f in F_g if doc_ref.document_type in FIELD_CATALOG[f].eligible_document_types]
            if not E_d:
                continue
                
            doc_chunk_ids = self.registry.doc_chunks.get(doc_id, [])
            if not doc_chunk_ids:
                continue
                
            # Rule B: Small Doc
            doc_text = self.registry.doc_text.get(doc_id, "")
            if doc_ref.document_type == DocType.affidavit:
                for cid in doc_chunk_ids:
                    chunk = self.registry.chunk_metadata.get(cid)
                    if chunk and self._affidavit_regex.search(chunk.get("text", "")):
                        selected_chunk_ids.add(cid)
                        chunk_reasons[cid].add("affidavit_anchor")
                        
            elif len(doc_text) <= config.SMALL_DOC_CHARS:
                for cid in doc_chunk_ids:
                    selected_chunk_ids.add(cid)
                    chunk_reasons[cid].add("whole_document")
            else:
                # Rule C: Large Doc
                for cid in doc_chunk_ids:
                    chunk = self.registry.chunk_metadata.get(cid)
                    if not chunk: continue
                    page_num = chunk.get("page_number") or chunk.get("metadata", {}).get("page_number")
                    if page_num in config.FRONT_MATTER_PAGES:
                        selected_chunk_ids.add(cid)
                        chunk_reasons[cid].add("front_matter")
                        
                for field in E_d:
                    results = self.searcher.search_field(
                        field, 
                        doc_filter=doc_id, 
                        top_k=config.RETRIEVAL_TOP_K, 
                        lexical_floor=config.LEXICAL_FLOOR
                    )
                    
                    for r in results:
                        cid = r["chunk_id"]
                        selected_chunk_ids.add(cid)
                        chunk_reasons[cid].add("retrieved")
                        
                        try:
                            idx = doc_chunk_ids.index(cid)
                            start = max(0, idx - config.NEIGHBOR_RADIUS)
                            end = min(len(doc_chunk_ids), idx + config.NEIGHBOR_RADIUS + 1)
                            for n_idx in range(start, end):
                                n_cid = doc_chunk_ids[n_idx]
                                selected_chunk_ids.add(n_cid)
                                chunk_reasons[n_cid].add("neighbor")
                        except ValueError:
                            pass
                            
                        if group in config.PAGE_EXPANSION_GROUPS:
                            chunk = self.registry.chunk_metadata.get(cid, {})
                            chunk_page = chunk.get("page_number") or chunk.get("metadata", {}).get("page_number")
                            if chunk_page is not None:
                                for pcid in doc_chunk_ids:
                                    pchunk = self.registry.chunk_metadata.get(pcid)
                                    if pchunk:
                                        ppage = pchunk.get("page_number") or pchunk.get("metadata", {}).get("page_number")
                                        if ppage == chunk_page:
                                            selected_chunk_ids.add(pcid)
                                            chunk_reasons[pcid].add("page_expansion")
                                            
        excerpts = []
        handle_map = {}
        evidence_chars = 0
        
        for doc_id in doc_ids:
            doc_chunk_ids = self.registry.doc_chunks.get(doc_id, [])
            doc_ref = self.registry.get_document_ref(doc_id)
            if not doc_ref: continue
            
            E_d = [f for f in F_g if doc_ref.document_type in FIELD_CATALOG[f].eligible_document_types]
            if not E_d: continue
            
            prev_handle = None
            for cid in doc_chunk_ids:
                if cid in selected_chunk_ids:
                    chunk = self.registry.chunk_metadata[cid]
                    text = chunk.get("text", "")
                    
                    handle = self.handle_for(cid)
                    handle_map[handle] = cid
                    evidence_chars += len(text)
                    
                    exc = EvidenceExcerpt(
                        handle=handle,
                        doc_label=self.doc_label_for(doc_id),
                        follows=prev_handle,
                        eligible_fields=E_d,
                        text=text
                    )
                    excerpts.append(exc)
                    prev_handle = handle
                    
        return self._batch_and_trim(bid_id, group, F_g, excerpts, handle_map, evidence_chars)

    def _batch_and_trim(self, bid_id: str, group: str, F_g: List[str], excerpts: List[EvidenceExcerpt], handle_map: Dict[str, str], total_chars: int) -> List[ContextPackage]:
        if not excerpts:
            pkg = ContextPackage(
                bid_id=bid_id,
                group=group,
                batch_index=0,
                batch_count=1,
                fields=[],
                unavailable_fields=F_g,
                excerpts=[],
                handle_map={},
                evidence_chars=0,
                trimmed_chunk_ids=[],
                max_output_tokens=config.GROUP_MAX_OUTPUT_TOKENS.get(group, 2000)
            )
            return [pkg]
            
        # The rendered prompt is more than the evidence text: it also carries the
        # system prompt, the field cards and one open/close tag pair per excerpt.
        # Budget against what is actually sent, otherwise a package that looks
        # small enough here still renders past MAX_PROMPT_CHARS.
        static_overhead = prompts.static_overhead_chars(group, F_g)
        evidence_budget = min(
            config.MAX_EVIDENCE_CHARS_PER_CALL,
            config.MAX_PROMPT_CHARS - static_overhead,
        )

        trimmed_chunk_ids = []
        kept: List[EvidenceExcerpt] = []
        used_chars = 0
        for exc in excerpts:
            # Cost of this excerpt in the rendered prompt, tags included.
            cost = len(exc.text) + prompts.excerpt_overhead_chars(exc)
            over_budget = used_chars + cost > evidence_budget
            over_count = len(kept) >= config.MAX_EXCERPTS_PER_CALL
            if over_budget or over_count:
                trimmed_chunk_ids.append(handle_map[exc.handle])
                continue
            kept.append(exc)
            used_chars += cost

        excerpts = kept
        total_chars = sum(len(exc.text) for exc in excerpts)

        # Nothing fits. Return the same shape as an empty group so callers keep
        # the invariant "fields is non-empty implies the package is renderable".
        if not excerpts:
            return [ContextPackage(
                bid_id=bid_id,
                group=group,
                batch_index=0,
                batch_count=1,
                fields=[],
                unavailable_fields=F_g,
                excerpts=[],
                handle_map={},
                evidence_chars=0,
                trimmed_chunk_ids=trimmed_chunk_ids,
                max_output_tokens=config.GROUP_MAX_OUTPUT_TOKENS.get(group, 2000)
            )]

        pkg = ContextPackage(
            bid_id=bid_id,
            group=group,
            batch_index=0,
            batch_count=1,
            fields=F_g,
            unavailable_fields=[],
            excerpts=excerpts,
            handle_map={exc.handle: handle_map[exc.handle] for exc in excerpts},
            evidence_chars=total_chars,
            trimmed_chunk_ids=trimmed_chunk_ids,
            max_output_tokens=config.GROUP_MAX_OUTPUT_TOKENS.get(group, 2000)
        )
        return [pkg]
