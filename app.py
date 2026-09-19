"""Inspection dashboard for ingestion, chunking and retrieval.

A development aid, not the execution path. Extraction runs through main.py so
that results are reproducible and written to disk. This dashboard accepts any
PDF/HTML files and uses the deterministic fake embedding client, so it needs no
API key.
"""

import os
import sys
import tempfile
import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.parsers.bid_manager import BidManager
from src.retrieval.chunker import DocumentChunker
from src.retrieval.registry import DocumentRegistry
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.embeddings import FakeEmbeddingClient
from src.schemas.field_catalog import FIELD_CATALOG
from src.ui.components import highlight_text
from src.config import CHUNK_MAX, CHUNK_OVERLAP

def init_session_state():
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "registry" not in st.session_state:
        st.session_state.registry = None
    if "searcher" not in st.session_state:
        st.session_state.searcher = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}

def process_documents(uploaded_files):
    with st.spinner("Processing documents..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup backend abstractions
            bid_manager = BidManager(output_dir=os.path.join(tmpdir, "out"))
            chunker = DocumentChunker(max_chunk_size=CHUNK_MAX, overlap=CHUNK_OVERLAP)
            registry = DocumentRegistry()
            
            # For offline evaluation, use FakeEmbeddingClient to avoid API key requirements
            # In production, this would be an OpenAI client wrapped in CachedEmbeddingClient
            embedding_client = FakeEmbeddingClient(dim=1536) 
            searcher = HybridSearcher(embedding_client=embedding_client)
            
            # Save files
            for uf in uploaded_files:
                file_path = os.path.join(tmpdir, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                    
            # Parse all files
            bid_data = bid_manager.process_bid_directory(tmpdir)
            docs = bid_data.get("documents", [])
            
            all_chunks = []
            
            # Chunk and register
            for doc in docs:
                chunks = chunker.chunk_document(doc)
                registry.register_document(doc, chunks)
                all_chunks.extend(chunks)
                
            # Build searcher
            searcher.add_chunks(all_chunks)
            searcher.build_index()
            
            st.session_state.registry = registry
            st.session_state.searcher = searcher
            st.session_state.documents = docs
            st.session_state.chunks = all_chunks
            st.session_state.processed = True
            
            # Metrics
            st.session_state.metrics = {
                "doc_count": len(docs),
                "chunk_count": len(all_chunks)
            }

def main():
    st.set_page_config(page_title="RFP Document Intelligence", layout="wide", page_icon="📄")
    init_session_state()
    
    st.title("RFP Document Intelligence Dashboard")
    st.markdown("Observability and evaluation interface for the RFP parsing and extraction pipeline.")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Configuration & Upload")
        uploaded_files = st.file_uploader(
            "Upload Bid Documents (PDF/HTML)", 
            type=["pdf", "html", "htm"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                try:
                    process_documents(uploaded_files)
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    
        # This dashboard always uses the deterministic offline embedding client,
        # so it needs no API key. The CLI pipeline chooses via --embeddings.
        st.markdown("**Embedding Client:** Offline / Deterministic")

        try:
            from src.extraction.llm_client import build_provider, ProviderConfigError
            provider = build_provider()
            provider_name = getattr(provider, "name", "unknown").capitalize()
            model_name = getattr(provider, "model", "unknown")
            provider_display = f"{provider_name} — {model_name}"
        except ProviderConfigError:
            provider_display = "Extraction is not configured"
        except Exception:
            provider_display = "Extraction is not configured"

        st.markdown(f"**Configured LLM provider:** {provider_display}")
                    
    # --- MAIN CONTENT ---
    if st.session_state.processed:
        # Top level metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Parsed Documents", st.session_state.metrics["doc_count"])
        m_col2.metric("Total Chunks", st.session_state.metrics["chunk_count"])
        m_col3.metric("Mode", "Offline inspection")
        
        st.divider()
        
        tab_docs, tab_chunks, tab_retrieval, tab_extract, tab_output = st.tabs([
            "Documents", "Chunks", "Retrieval", "Extraction", "Final Output"
        ])
        
        # --- DOCUMENTS TAB ---
        with tab_docs:
            st.header("Parsed Documents")
            doc_records = []
            for d in st.session_state.documents:
                meta = d.get("metadata", {})
                doc_records.append({
                    "Document ID": meta.get("doc_id", "N/A"),
                    "Filename": meta.get("source_filename", "N/A"),
                    "Type": meta.get("document_type", "N/A"),
                    "Format": meta.get("source_format", "N/A"),
                    "Pages/Sections": meta.get("section_count", 0),
                    "Text Chars": meta.get("text_chars", 0),
                    "Addendum No": meta.get("addendum_number", "N/A")
                })
            if doc_records:
                st.dataframe(pd.DataFrame(doc_records), use_container_width=True)
            else:
                st.info("No documents parsed.")
                
        # --- CHUNKS TAB ---
        with tab_chunks:
            st.header("Document Chunks")
            search_query = st.text_input("Filter chunks by text (case-insensitive):")
            
            display_chunks = st.session_state.chunks
            if search_query:
                display_chunks = [c for c in display_chunks if search_query.lower() in c.get("text", "").lower()]
                
            st.write(f"Showing {len(display_chunks)} chunks.")
            
            for idx, chunk in enumerate(display_chunks[:50]): # Limit to 50 for UI performance
                with st.container():
                    st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id')}` | **Source:** `{chunk.get('source_filename')}` | **Pages:** `{chunk.get('page_numbers', [])}`")
                    
                    text = chunk.get("text", "")
                    if search_query:
                        text = highlight_text(text, search_query)
                        
                    st.info(text)
                    st.divider()
            if len(display_chunks) > 50:
                st.warning(f"And {len(display_chunks) - 50} more chunks not shown to preserve performance.")
                
        # --- RETRIEVAL TAB ---
        with tab_retrieval:
            st.header("Hybrid Retrieval Inspection")
            st.markdown(
                "Run the two-level RRF retrieval for any field in the catalog, or "
                "search the indexed chunks for a literal term."
            )

            mode = st.radio("Mode", ["Field from catalog", "Literal term"], horizontal=True)
            searcher = st.session_state.searcher

            if mode == "Field from catalog":
                field_name = st.selectbox("Field", list(FIELD_CATALOG.keys()))
                term = None
                run = st.button("Run Retrieval")
                results = searcher.search_field(field_name, top_k=6) if run else []
            else:
                term = st.text_input("Term to search for in the indexed chunks")
                run = st.button("Run Retrieval")
                results = searcher.search_exact_identifier(term, top_k=6) if (run and term) else []

            if run:
                st.write(f"Retrieved **{len(results)}** chunks.")

                for r in results:
                    meta = r.get("retrieval_metadata", {})
                    with st.container():
                        st.markdown(
                            f"**Rank {meta.get('overall_rank')}** "
                            f"(Score: {meta.get('fused_score', 0):.4f}) - "
                            f"Source: `{r.get('source_filename')}` Page: `{r.get('page_numbers', [])}`"
                        )

                        text = r.get("text", "")
                        if term:
                            text = highlight_text(text, term)

                        st.info(text)

                        with st.expander("Technical Metadata"):
                            st.json(meta)
                        st.divider()

        # --- EXTRACTION TAB ---
        with tab_extract:
            st.header("LLM Extraction")
            st.info(
                "Extraction is strictly executed by the CLI (via main.py) and is "
                "not invoked by this Streamlit process. This ensures runs are "
                "reproducible and outputs are safely written to disk."
            )
            st.code("python main.py --bid <bid_directory>", language="bash")
            st.markdown(
                "The engine sends one call per field group, parses the reply against a "
                "strict schema, grounds every quote against the indexed chunks, and "
                "resolves competing candidates deterministically. Candidates, evidence "
                "and rejection reasons are written to the diagnostics file."
            )

        # --- FINAL OUTPUT TAB ---
        with tab_output:
            st.header("Output Contract")
            st.markdown(
                "A run writes the answer to `<bid>.json` and the audit trail to "
                "`<bid>.diagnostics.json`. The answer contains exactly these fields, "
                "in this order; a field with no valid grounded candidate is `null`."
            )
            for field_name, spec in FIELD_CATALOG.items():
                with st.expander(spec.assignment_label, expanded=False):
                    st.write(f"**Internal name:** `{field_name}`")
                    st.write(f"**Definition:** {spec.definition}")
                    st.write(f"**Eligible document types:** "
                             f"{', '.join(d.value for d in spec.eligible_document_types)}")
    else:
        st.info("👈 Upload documents and click 'Process Documents' to begin.")

if __name__ == "__main__":
    main()
