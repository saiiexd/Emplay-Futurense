# Emplay RFP Document Intelligence System

## Executive Summary
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to extract 20 structured fields from heterogeneous procurement documents (PDFs and HTML) with strict evidence grounding and deterministic conflict resolution.

Procurement document processing is difficult because critical information is rarely contained within a single file. An original Request for Proposal (RFP) may establish a due date or requirement, which is later modified by one or more addenda, clarified in Q&A sections, and summarized in portal listings. Simple keyword extraction fails because terms like "due date" appear repeatedly across documents with conflicting values. A naive single-prompt LLM approach fails because models struggle to reliably process hundreds of pages of context, frequently hallucinate authority, and silently ignore amendments.

This system solves these problems by restricting the LLM to semantic candidate extraction and enforcing all authority, precedence, schema compliance, and formatting through deterministic Python logic. The input contract accepts directories of unstructured PDFs and HTML files; the output contract guarantees a highly structured, 20-field JSON document where every non-null value is traceable to a verbatim quote from an authoritative source document.

## Architecture

The pipeline processes documents through a strict sequence of deterministic and semantic operations:

1. **Ingestion and Normalization:** Documents (PDFs via PyMuPDF, HTML via BeautifulSoup) are parsed into a normalized textual representation. Noise such as HTML styles/scripts and PDF artifacts are stripped.
2. **Document Registry and Provenance:** Documents are classified (e.g., `rfp_main`, `addendum`, `portal_listing`, `specification`, `affidavit`) based on content and filename heuristics. Addendum numbers and amendment language are identified during this phase to establish later precedence.
3. **Chunking:** Text is split into deterministic ~1000-character chunks with a 200-character overlap to preserve semantic boundaries without exceeding model context windows.
4. **Lexical and Semantic Retrieval:** 
   - **BM25 Lexical Search:** Matches field-specific aliases against the corpus.
   - **Embedding Semantic Search:** Matches semantic intent using cosine similarity.
   - **Reciprocal Rank Fusion (RRF):** Fuses lexical and semantic scores, guaranteeing that exact identifier matches (e.g., part numbers) are not lost due to weak semantic similarity.
5. **Context Construction:** The system groups the 20 fields into six logical extraction groups. For each group, an isolated context budget is constructed using the highest-scoring retrieved chunks.
6. **LLM Extraction:** The LLM receives the constructed context and is prompted to extract specific fields. It is strictly constrained to output JSON candidates containing the value and a verbatim quote.
7. **Grounding Validation:** Every quote proposed by the LLM is deterministically verified against the original chunk text. If the quote is not present, the candidate is discarded. The system rejects candidates that fail evidence grounding.
8. **Deterministic Candidate Resolution:** When multiple documents propose conflicting values for a field (e.g., a new due date in an addendum vs. the original RFP), Python logic resolves the conflict using a strict tier-based precedence system. The LLM never independently decides which document is authoritative.
9. **Final Normalization and Output:** Validated, resolved candidates are serialized into the required JSON structure.

## Why This Architecture?
Retrieval is necessary for multi-document RFPs because full RFPs can easily exceed hundreds of pages, overflowing token limits and diluting the model's attention. Field-specific context is preferable to passing the entire corpus because different fields require different context; a contact's email address is located differently than a product specification.

The LLM is constrained to generating semantic candidates rather than making authority decisions because models are unreliable adjudicators of legal precedence. They often prefer whichever text they read last or hallucinate rules. Verbatim evidence grounding is required so that humans can instantly audit the extracted values. Addendum precedence is implemented deterministically because identifying whether an addendum amends an RFP is a structural logic problem, not a generative language problem.

## Extraction Groups
To prevent context dilution and adhere to token limits, the 20 fields are processed in six isolated groups:
1. **Identity:** Bid Number, Title
2. **Schedule:** Due Date, Pre Bid Meeting, Delivery Date
3. **Terms:** Bid Submission Type, Term of Bid, Bid Bond Requirement, Payment Terms
4. **Products:** Model_no, Part_no, Product, MFG for Registration, Contract or Cooperative to use
5. **Specifications:** Product Specification, Installation
6. **Contacts and Summary:** contact_info, company_name, Bid Summary, Any Additional Documentation Required

## Addendum Precedence and Resolution Strategy
Conflicts are resolved using a hierarchical precedence tier defined in `src/resolution/resolver.py`:
1. **Tier 40 (Amendment Addendum):** Addenda containing explicit amendment language (e.g., "the new due date is", "is hereby changed"). Higher addendum numbers override lower ones.
2. **Tier 30 (Main Solicitation):** Main solicitation documents (`rfp_main`).
3. **Tier 25 (Clarification Addendum):** Addenda that clarify or mention a field without amending it.
4. **Tier 20 (Specification):** Dedicated specification documents.
5. **Tier 10 (Portal Listing):** Portal/listing pages.
6. **Tier 5 (Affidavit):** Affidavits or attached forms.

An addendum only overrides the main RFP if it contains explicit amendment language near the extracted quote. Vendor questions in a Q&A addendum are treated as clarifications (Tier 25) and cannot override the primary document.

## Handling Missing Information
Fields lacking grounded evidence are explicitly represented as `null`. The system does not infer, guess, or inject placeholder text. A `null` accurately reflects that the information is absent from the provided documents.

## Heterogeneous Document Handling
The ingestion layer normalizes both PDF and HTML inputs into a common internal structure. Provenance is maintained at the chunk level; every chunk retains a reference to its source document, page index, and classification, ensuring that any extracted quote can be traced back to its exact origin.

## Input and Output Contract

### Input Directory Structure
The system expects input documents organized into bid-specific directories:
```
data/
└── input/
    ├── Bid1/
    │   ├── JA-207652 Student and Staff Computing Devices FINAL.pdf
    │   ├── Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf
    │   ├── Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf
    │   └── Student and Staff Computing Devices - Bid Information...html
    └── Bid2/
        ├── PORFP_-_Dell_Laptop_Final.pdf
        ├── Dell_Laptop_Specs.pdf
        ├── Contract_Affidavit.pdf
        ├── Mercury_Affidavit.pdf
        └── Dell Laptops w_Extended Warranty - Bid Information...html
```

### Output Artifacts
Results are written to `data/output/`:
- **`extracted_data.json`**: The primary aggregate deliverable containing structured data for all processed bids.
- **`Bid1.flat.json` / `Bid2.flat.json`**: Individual JSON outputs for each bid.
- **`Bid1.diagnostics.json` / `Bid2.diagnostics.json`**: Detailed audit trails showing the exact provenance, chunk IDs, and precedence rules applied for every extracted value. These files provide absolute transparency into the system's decisions.

### 20-Field Output Schema
The JSON output strictly adheres to the following 20 fields for each bid:
`Bid Number`, `Title`, `Due Date`, `Bid Submission Type`, `Term of Bid`, `Pre Bid Meeting`, `Installation`, `Bid Bond Requirement`, `Delivery Date`, `Payment Terms`, `Any Additional Documentation Required`, `MFG for Registration`, `Contract or Cooperative to use`, `Model_no`, `Part_no`, `Product`, `contact_info`, `company_name`, `Bid Summary`, `Product Specification`.

**Example Output (`Bid2` snippet from `extracted_data.json`):**
```json
{
  "Bid Number": "BPM044557",
  "Title": "Dell Laptops w/Extended Warranty",
  "Due Date": "06/10/2024",
  "Bid Submission Type": "Purchase Order Request for Proposal responses will only be accepted through the State's eMaryland Marketplace Advantage (eMMA) e-Procurement system. Bids will not be accepted by email, fax, U.S. Mail, or hand delivery.",
  "Term of Bid": null,
  "Pre Bid Meeting": null,
  "Installation": null,
  "Bid Bond Requirement": null,
  "Delivery Date": "Delivery within 45 days of Award.",
  "Payment Terms": "Email invoices to STOaccountspayable@treasurer.state.md.us. Invoice(s) shall be submitted within 10 days of delivering the equipment and shall include the contractor name, mailing address, social security number or Federal Tax ID number, phone number, the State's assigned PORFP number, date, invoice number, and amount due.",
  "Any Additional Documentation Required": "Mercury Affidavit; Contract Affidavit.",
  "MFG for Registration": "Dell",
  "Contract or Cooperative to use": null,
  "Model_no": null,
  "Part_no": null,
  "Product": "laptops",
  "contact_info": "Calvin.Kiser@maryland529.org",
  "company_name": "State of Maryland Treasurer's Office",
  "Bid Summary": "Office is in need of a refresh of laptops and must acquire enough...",
  "Product Specification": "Dell Latitude 5550 XCTO Base; Intel Core Ultra 5 125U processor..."
}
```
*(Note: Nulls indicate the information was strictly not found in the documents. The `Product Specification` and `Bid Summary` strings are abbreviated above for display, but are full length in the actual JSON artifact).*

## Project Structure
```
Project/
├── app.py                   # Streamlit observability dashboard
├── main.py                  # Primary backend extraction CLI
├── requirements.txt         # Dependency declarations
├── run.bat                  # Windows: offline validation + Streamlit
├── run.sh                   # macOS/Linux: offline validation + Streamlit
├── .env.example             # Configuration template
├── src/
│   ├── parsers/             # PDF/HTML parsing & normalization
│   ├── retrieval/           # Chunking, BM25, embeddings, hybrid search, RRF
│   ├── extraction/          # Context construction, prompts, LLM client
│   ├── resolution/          # Deterministic addendum precedence & resolution
│   ├── validation/          # Evidence grounding
│   ├── schemas/             # Pydantic data models & Field Catalog
│   └── ui/                  # Streamlit dashboard components
├── tests/                   # 220 offline deterministic tests
└── data/
    ├── input/               # Bid1 and Bid2 source PDFs & HTML
    └── output/              # Final JSON outputs & diagnostic audit logs
```

## Technology Stack
- **Parsing:** PyMuPDF, BeautifulSoup4
- **Retrieval:** rank_bm25, numpy (for cosine similarity)
- **Data Modeling & Schema:** Pydantic
- **LLM SDK:** google-generativeai
- **Frontend Dashboard:** Streamlit
- **Testing:** pytest

## Execution Instructions

### 1. Environment Setup (Windows)
Create the environment and install dependencies:
```powershell
cd "Project"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration and Security
The system requires an API key for live LLM extraction. Real credentials are never tracked in Git.
1. Copy `.env.example` to `.env` (which is correctly listed in `.gitignore`).
2. Open `.env` and set `GEMINI_API_KEY`.
3. Set `LLM_PROVIDER=gemini` (or `fake` for offline mode).

### 3. Backend Execution (Live LLM Extraction)
To execute the complete end-to-end extraction pipeline, you must provide paths to the bid directories:
```powershell
python main.py --bid data/input/Bid1 --bid data/input/Bid2
```
**What happens:** The system parses all documents, builds the chunk registry, executes the hybrid search, constructs prompts for the 6 extraction groups, queries the Gemini model (if configured), performs deterministic grounding and precedence resolution, and finally serializes the 20-field JSON to `data/output/extracted_data.json`.
*Note: If the API quota is exhausted, the system returns exit code `3` and logs a failure rather than writing null-filled data.*

### 4. Offline Deterministic Validation
You can run the entire test suite deterministically without network access:
```powershell
.\run.bat
```
*(On macOS/Linux, use `./run.sh`)*.
**What happens:** This runs all 220 unit and integration tests using a mocked `fake` LLM provider. Once tests complete successfully, it launches the Streamlit dashboard automatically.

### 5. Frontend Observability Dashboard
To launch the dashboard manually:
```powershell
streamlit run app.py
```
**What the Streamlit dashboard does:** It provides an observability interface to debug document ingestion, chunking, and hybrid retrieval. You can upload documents and see exactly how they are parsed and ranked by the BM25/Embedding system.
**What the Streamlit dashboard does NOT do:** It does **not** perform live LLM extraction. The dashboard is purely for introspection and evaluation of the deterministic RAG mechanics. The actual JSON artifacts must be built using the `main.py` CLI.

## Testing
The repository includes a rigorous suite of **220** tests (`pytest -q`).
These tests validate:
- Ingestion and chunk overlap boundaries.
- The hybrid RRF retrieval system.
- Context budget enforcement.
- Strict JSON schema adherence.
- Grounding verification logic.
- Addendum precedence conflict resolution.
- The complete end-to-end pipeline using the mock provider.
*Because the tests use the `fake` provider, they do not prove live LLM extraction accuracy, but they do definitively prove the structural soundness of the deterministic validation layers.*

## Limitations and Engineering Trade-offs
- **Live Provider Quota:** The live extraction pipeline depends on external API limits (Google Gemini). Quota exhaustion behaves as a loud failure rather than producing silent hallucinations. 
- **Tables and Spatial Formatting:** Complex nested tables in PDFs are flattened into linear text chunks by PyMuPDF. While sufficient for value extraction, complex multi-axis interpretation is limited.

## Assignment Alignment

### 1. Accuracy
The system ensures extraction accuracy by enforcing strict evidence grounding in `src/validation/grounding.py`. Rather than trusting the LLM to output a standalone value, the system requires the model to identify a verbatim quote. The Python layer validates this quote against the exact text of the source chunks; if it fails, the candidate is rejected. Conflicting accurate data across documents is resolved deterministically in `src/resolution/resolver.py`, preventing the model from arbitrarily choosing between valid but competing facts. The final output is enforced against the exact 20-field schema.

### 2. Robustness
Heterogeneous documents (PDFs and HTML) are normalized through a common parser interface (`src/parsers/`), removing styles and artifacts while preserving semantic boundaries. Document chunks retain their provenance metadata so that later stages can evaluate them correctly. The retrieval system uses Reciprocal Rank Fusion (RRF) to combine BM25 lexical search with embeddings, preventing exact part numbers from being lost to weak semantic scoring. By constructing separate context windows for the six extraction groups, the system isolates relevant evidence and robustly handles RFPs that exceed LLM token limits. Missing information is safely defaulted to `null` rather than generating hallucinated placeholders.

### 3. Code Quality
The codebase is structured with clear separation of concerns: parsing, retrieval, extraction, resolution, validation, and schema definitions each reside in dedicated modules under `src/`. Configuration is centralized, and credentials are intentionally kept out of version control via a git-ignored `.env` file (demonstrated by the `.env.example` template). The primary entry point (`main.py`) provides clear CLI arguments, and the system is covered by a 220-test offline suite. Vendor SDK logic is tightly encapsulated in `src/extraction/llm_client.py` rather than being scattered throughout the application.

### 4. Use of LLMs / NLP / RAG
The project implements a complete Retrieval-Augmented Generation pipeline. Lexical and semantic retrieval mechanisms gather field-specific context which is then passed to the LLM (configured for Google Gemini). The division of responsibility is explicit: the LLM performs the complex task of semantic candidate extraction (identifying meaning and quotes), while deterministic Python components handle chunking, evidence validation, schema enforcement, and document precedence.

### Assignment Deliverables

| Assignment Requirement | Repository Location | Purpose |
|------------------------|---------------------|---------|
| Python script / notebook | [`main.py`](main.py) and [`src/`](src/) | Primary extraction entry point and backend implementation |
| README | [`README.md`](README.md) | Setup, dependencies, architecture, and execution instructions |
| JSON output | [`data/output/extracted_data.json`](data/output/extracted_data.json) | Structured 20-field information extracted from the provided documents |

### Additional Evaluation Evidence
While not explicitly required by the core assignment, the repository includes a suite of 220 offline deterministic tests to validate pipeline behavior (`pytest -q`), detailed JSON diagnostic audit logs (`data/output/Bid1.diagnostics.json`), and an interactive Streamlit observability interface (`app.py`) for inspecting chunking and retrieval scoring.

## How to Evaluate This Repository
1. **Review the Architecture:** Read the architecture diagram and the "Why This Architecture?" section above.
2. **Execute Tests:** Run `pytest -q` to verify the 220 offline deterministic tests.
3. **Inspect the JSON:** Open `data/output/extracted_data.json` to verify strict adherence to the 20-field schema requirement.
4. **Inspect Resolution:** Review `src/resolution/resolver.py` to see the exact implementation of the addendum precedence tiers.
5. **Inspect Grounding:** Review `src/validation/grounding.py` to see how the system strictly rejects hallucinations.
6. **Interact with the RAG Pipeline:** Run `streamlit run app.py` and upload the RFP documents to explore how chunks are generated and scored before they ever reach an LLM.
