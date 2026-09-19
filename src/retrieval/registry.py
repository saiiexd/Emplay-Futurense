from typing import Dict, List, Optional
from src.schemas.models import DocumentRef
from src.schemas.enums import DocType

class DocumentRegistry:
    """
    Maintains a mapping of all indexed documents and their chunks to allow
    provenance tracing, neighbor lookups, and quote-to-page resolution without
    modifying the chunker directly.
    """
    def __init__(self):
        self.doc_refs: Dict[str, DocumentRef] = {} # doc_id -> DocumentRef
        self.doc_chunks: Dict[str, List[str]] = {} # doc_id -> list of chunk_ids
        self.chunk_metadata: Dict[str, dict] = {} # chunk_id -> chunk dict
        self.doc_text: Dict[str, str] = {}
        self.page_text: Dict[str, Dict[int, str]] = {}

    def register_document(self, doc_data: dict, chunks: List[dict]):
        """Registers a parsed document and its associated chunks."""
        meta = doc_data.get("metadata", {})
        doc_id = meta.get("doc_id", meta.get("source_filename", "unknown"))
        
        dtype_str = meta.get("document_type", "unknown")
        try:
            dtype = DocType(dtype_str)
        except ValueError:
            dtype = DocType.unknown

        doc_ref = DocumentRef(
            document_id=doc_id,
            filename=meta.get("source_filename", ""),
            document_type=dtype,
            doc_date=meta.get("doc_date"),
            addendum_number=meta.get("addendum_number"),
            page_index=None,
            printed_page_label=None
        )
        self.doc_refs[doc_id] = doc_ref
        
        full_text = []
        page_dict = {}
        for sec in doc_data.get("sections", []):
            text = sec.get("text", "")
            full_text.append(text)
            pn = sec.get("metadata", {}).get("page_number")
            if pn is not None:
                if pn not in page_dict:
                    page_dict[pn] = []
                page_dict[pn].append(text)
                
        self.doc_text[doc_id] = " ".join(full_text)
        self.page_text[doc_id] = {p: " ".join(texts) for p, texts in page_dict.items()}

        chunk_ids = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if cid:
                chunk_ids.append(cid)
                chunk["doc_id"] = doc_id
                self.chunk_metadata[cid] = chunk
                
        self.doc_chunks[doc_id] = chunk_ids

    def get_document_ref_by_filename(self, filename: str) -> Optional[DocumentRef]:
        for ref in self.doc_refs.values():
            if ref.filename == filename:
                return ref
        return None

    def get_document_ref(self, doc_id: str) -> Optional[DocumentRef]:
        return self.doc_refs.get(doc_id)

    def get_ordered_chunk_ids(self, doc_id: str) -> List[str]:
        return self.doc_chunks.get(doc_id, [])

    def provenance(self, chunk_id: str) -> Optional[DocumentRef]:
        """Returns provenance including doc info and printed page label for the chunk."""
        chunk = self.chunk_metadata.get(chunk_id)
        if not chunk:
            return None
        doc_id = chunk.get("doc_id")
        base_ref = self.doc_refs.get(doc_id)
        if not base_ref:
            return None
            
        pages = chunk.get("page_numbers", [])
        page_index = pages[0] if pages else None
        printed_label = str(page_index) if page_index else None
        
        return DocumentRef(
            document_id=base_ref.document_id,
            filename=base_ref.filename,
            document_type=base_ref.document_type,
            doc_date=base_ref.doc_date,
            addendum_number=base_ref.addendum_number,
            page_index=page_index,
            printed_page_label=printed_label
        )

    def resolve_page(self, chunk_id: str, quote: str) -> List[int]:
        """Quote-aware page resolution."""
        chunk = self.chunk_metadata.get(chunk_id)
        if not chunk:
            return []
        pages = chunk.get("page_numbers", [])
        if not pages:
            return []
        if len(pages) == 1:
            return pages
            
        doc_id = chunk.get("doc_id")
        p_texts = self.page_text.get(doc_id, {})
        
        matches = []
        for p in pages:
            if p in p_texts and quote in p_texts[p]:
                matches.append(p)
                
        if len(matches) == 1:
            return matches
        return pages

    def chunk_position(self, chunk_id: str) -> int:
        chunk = self.chunk_metadata.get(chunk_id)
        if not chunk:
            return -1
        doc_id = chunk.get("doc_id")
        ordered = self.doc_chunks.get(doc_id, [])
        try:
            return ordered.index(chunk_id)
        except ValueError:
            return -1

    def get_previous_chunk(self, chunk_id: str) -> Optional[dict]:
        pos = self.chunk_position(chunk_id)
        if pos > 0:
            doc_id = self.chunk_metadata[chunk_id]["doc_id"]
            ordered = self.doc_chunks.get(doc_id, [])
            return self.chunk_metadata.get(ordered[pos - 1])
        return None

    def get_next_chunk(self, chunk_id: str) -> Optional[dict]:
        pos = self.chunk_position(chunk_id)
        if pos >= 0:
            doc_id = self.chunk_metadata[chunk_id]["doc_id"]
            ordered = self.doc_chunks.get(doc_id, [])
            if pos < len(ordered) - 1:
                return self.chunk_metadata.get(ordered[pos + 1])
        return None
