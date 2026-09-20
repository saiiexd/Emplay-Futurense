# RFP Document Intelligence

Extracts 20 structured fields from heterogeneous procurement documents
(PDF and HTML) using a RAG pipeline with LLM-powered semantic extraction,
strict evidence grounding, and deterministic conflict resolution.

---

## Assignment Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | **Python extraction script** | [`main.py`](main.py) — primary entry point that runs the full pipeline |
| 2 | **README with instructions** | This file |
| 3 | **JSON structured output** | [`data/output/extracted_data.json`](data/output/extracted_data.json) — aggregate file with both bids |
|   | Per-bid JSON | [`data/output/Bid1.flat.json`](data/output/Bid1.flat.json), [`data/output/Bid2.flat.json`](data/output/Bid2.flat.json) |
|   | Audit trails (diagnostics) | [`data/output/Bid1.diagnostics.json`](data/output/Bid1.diagnostics.json), [`data/output/Bid2.diagnostics.json`](data/output/Bid2.diagnostics.json) |

The assignment asks for "a JSON file containing the structured information
extracted from each provided document." The aggregate
[`extracted_data.json`](data/output/extracted_data.json) contains the result
for both supplied bid document sets in one file. The per-bid flat JSON files
contain the same values individually.

---

## Evaluation Criteria

### Accuracy

Every non-null value in the final JSON is traced to a verbatim quote from the
source documents. The grounding layer (`src/validation/grounding.py`) requires
that:

- The evidence handle resolves to a chunk that was actually supplied.
- The document type is eligible for that field per the catalog.
- The quote is found in the chunk text by exact, whitespace-normalized, case-insensitive, or compact matching.
- The extracted value is contained in its own quote.

Fields with no valid grounded candidate are `null` — never filled with
hallucinated or inferred content. The 20-field schema matches the assignment
table exactly, verified by automated JSON schema validation in the test suite.

### Robustness

The pipeline handles heterogeneous document sets without hardcoding:

- **PDF parsing** — page-by-page with PyMuPDF, including table extraction and low-text-page detection.
- **HTML parsing** — BeautifulSoup with script/style/nav/comment stripping.
- **Document classification** — `rfp_main`, `addendum`, `portal_listing`, `specification`, `affidavit` inferred from filenames and content.
- **Addendum handling** — addendum numbers extracted from document text; amendment cues detected near quotes to distinguish amendments from clarifications.
- **Missing information** — results in `null`, never in placeholders or fabricated values.
- **Malformed LLM responses** — re-sent with a correction prompt; after retries, reported rather than silently dropped.
- **Provider failures** — bounded retries with backoff; if all groups fail, no output is written (a null-filled file would misrepresent a failed run as a result).
- **220 automated tests** cover parsing, chunking, retrieval, context construction, prompt rendering, leakage protection, response parsing, grounding, normalization, precedence resolution, and the complete end-to-end pipeline.

### Code Quality

The codebase is organized around clear responsibilities:

- `src/parsers/` — ingestion (PDF, HTML) and document classification
- `src/retrieval/` — chunking, BM25 + embedding hybrid search, document registry
- `src/extraction/` — context building, prompt rendering, LLM client abstraction, response parsing, rule-based candidates
- `src/resolution/` — deterministic precedence and conflict resolution
- `src/validation/` — evidence grounding, canonical matching, normalization
- `src/schemas/` — Pydantic models, field catalog, enums

Provider abstraction (`src/extraction/llm_client.py`) isolates all vendor SDK
usage. No module outside that file imports a vendor SDK. Configuration uses
environment variables with no hardcoded secrets.

### Use of LLMs / NLP / RAG

The pipeline implements retrieval-augmented generation with these components:

1. **Document parsing and normalization** — PDF pages and HTML sections are extracted into a uniform structure.
2. **Chunking** — deterministic ~1000-character chunks with 200-character overlap, preserving page boundaries.
3. **Lexical retrieval (BM25)** — field-specific aliases from the catalog are used as BM25 queries; multiple alias rankings are fused with Reciprocal Rank Fusion.
4. **Embedding retrieval** — semantic queries from the catalog are embedded and compared via cosine similarity (configurable: fake for offline, OpenAI for production).
5. **RRF fusion** — lexical and semantic rankings are fused with RRF, with a lexical floor so exact identifier matches are never pushed out by weak semantic scores.
6. **Field-specific ContextBuilder** — selects evidence per field group, assigns opaque handles, enforces prompt budget, includes small documents whole and retrieves from large ones.
7. **Structured LLM extraction** — the model receives evidence excerpts with opaque handles (no document types, filenames, addendum numbers, or retrieval scores visible) and returns candidates with evidence references and verbatim quotes, validated against a strict JSON schema.
8. **Six extraction groups** — identity, schedule, terms, products, specifications, contacts/summary — each with a focused prompt and evidence set.
9. **Evidence grounding** — every candidate is validated against source text before it can influence any answer.
10. **Deterministic candidate resolution** — document provenance determines which candidate wins, using fixed precedence tiers. The LLM never chooses between conflicting documents.

The LLM is used for **semantic candidate extraction** only. It identifies values
and quotes them. All authority decisions (which document takes precedence, whether
an addendum amends or merely clarifies) are made by deterministic Python rules in
`src/resolution/resolver.py`. This prevents the model from hallucinating
authority or silently preferring whichever text it saw last.

---

## Architecture

```
bid folder
   ├─ BidManager            parse each PDF/HTML into normalized sections
   ├─ DocumentChunker       deterministic ~1000-char chunks with overlap
   ├─ DocumentRegistry      chunk → document provenance, addendum numbers
   ├─ HybridSearcher        BM25 + embeddings fused with RRF
   ├─ ContextBuilder        select evidence per field group, assign handles
   ├─ prompts.py            render messages + strict JSON schema
   ├─ ExtractionEngine      call the LLM provider, parse the reply
   ├─ GroundingValidator     verify every quote against real chunk text
   ├─ RuleCandidateGenerator deterministic candidates for self-named forms
   ├─ CandidateResolver     precedence and addendum resolution
   └─ final record          20-field JSON + separate diagnostics
```

### Addendum Resolution

Candidates are ranked by document provenance:

| Tier | Source |
|-----:|--------|
| 40 | addendum that **amends** the field (higher addendum number wins) |
| 30 | main solicitation document |
| 25 | addendum that mentions the field without amending it |
| 20 | specification document |
| 10 | portal / listing page |
|  5 | affidavit or attached form |

An addendum reaches tier 40 only when generic amendment language (e.g. "the new
due date", "is hereby amended", "extends the deadline", "delete ... insert")
appears near its quote. Question-and-answer text without such language is
recorded as a *clarification* and does not change the value. This is why a
vendor question inside an addendum cannot silently rewrite the solicitation.

When an amendment wins, the earlier value is retained as `superseded` in
diagnostics. Fields marked as aggregated (required documents, contacts, product
specification) collect evidence across eligible documents unless an addendum
amends the list, which replaces it.

### Why Resolution is Deterministic

Choosing between an original due date and an amended one is not a language
problem; it is a rule about document authority. A model asked to choose would be
unpredictable across runs, unable to explain itself in auditable terms, and prone
to preferring whichever text it saw last. Python applies fixed precedence rules
to provenance the registry already knows, so the same candidates always produce
the same answer and every decision can be replayed from the diagnostics file.

---

## Project Structure

```
Project/
├── main.py                  # Primary Python extraction entry point
├── app.py                   # Optional Streamlit observability dashboard
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template (placeholders only)
├── run.bat                  # Windows: offline validation + Streamlit
├── run.sh                   # macOS/Linux: offline validation + Streamlit
├── src/
│   ├── config.py            # Pipeline configuration constants
│   ├── parsers/             # PDF and HTML document ingestion
│   ├── retrieval/           # Chunking, BM25, embeddings, hybrid search
│   ├── extraction/          # Context builder, prompts, LLM client, engine
│   ├── resolution/          # Deterministic precedence and conflict resolution
│   ├── validation/          # Evidence grounding, canonical matching, normalization
│   ├── schemas/             # Pydantic models, field catalog, enums
│   └── ui/                  # Streamlit helper components
├── tests/                   # 220 automated tests (fully offline)
│   └── fixtures/            # Test fixtures with corpus-specific expectations
├── data/
│   ├── input/               # Supplied bid document corpus (tracked)
│   │   ├── Bid1/            # 4 documents: main RFP, 2 addenda, portal listing
│   │   └── Bid2/            # 5 documents: PORFP, spec sheet, 2 affidavits, portal
│   └── output/              # Generated structured JSON and diagnostics
│       ├── extracted_data.json      # Aggregate deliverable (both bids)
│       ├── Bid1.flat.json           # Bid 1 structured output (20 fields)
│       ├── Bid2.flat.json           # Bid 2 structured output (20 fields)
│       ├── Bid1.diagnostics.json    # Bid 1 audit trail
│       └── Bid2.diagnostics.json    # Bid 2 audit trail
└── docs/
    └── Assignment.pdf       # Original assignment specification
```

---

## Setup

```bash
cd Project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Dependencies

- **PyMuPDF** — PDF parsing with page-level text and table extraction
- **beautifulsoup4** — HTML parsing and noise removal
- **python-dotenv** — environment variable loading
- **pydantic** — strict data models and JSON schema validation
- **rank_bm25** — BM25 lexical retrieval
- **numpy** — embedding similarity computation
- **google-generativeai** — Google Gemini API client (for live extraction)
- **openai** — OpenAI embeddings (optional, for semantic retrieval channel)
- **streamlit** — optional observability dashboard
- **pytest** — test framework

---

## Running Extraction

### Live Extraction (requires Gemini API key and quota)

```bash
cd Project
copy .env.example .env        # then set a real GEMINI_API_KEY in .env
python main.py --bid data/input/Bid1 --bid data/input/Bid2
```

This runs the full pipeline: ingestion → chunking → retrieval → context
construction → six LLM extraction calls → parsing → grounding → precedence
resolution → final JSON output. Results are written to `data/output/`.

Point `--bid` at any folder of PDF/HTML documents; repeat it for more bids.

**Exit codes:** `0` all groups completed · `2` provider not configured (missing
key) · `3` the run finished but one or more groups failed at the provider.

### Offline Validation (no API key required)

```bash
cd Project
.\run.bat          # Windows
# ./run.sh         # macOS/Linux
```

This runs the full offline test suite (220 tests, no network calls) and then
starts the Streamlit dashboard. The test suite exercises the complete pipeline
with a deterministic fake provider, including ingestion, chunking, retrieval,
ContextBuilder, all six extraction groups, strict response parsing, grounding,
rule candidates, deterministic resolution, and final serialization.

### Live Extraction (explicit)

```powershell
cd Project
$env:LLM_PROVIDER = "gemini"
python main.py --bid data/input/Bid1 --bid data/input/Bid2
```

On macOS/Linux: `export LLM_PROVIDER=gemini` instead.

### Streamlit Dashboard Only

```bash
cd Project
streamlit run app.py
```

The dashboard is an observability interface for inspecting parsing, chunking,
and retrieval on uploaded files. It does not invoke live extraction. It needs
no API key.

---

## Configuring the Provider

Copy `.env.example` to `.env` (git-ignored) and configure:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_EXTRACTION_MODEL=models/gemini-flash-latest
```

| `LLM_PROVIDER` | Variables | Transport |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY`, `GEMINI_EXTRACTION_MODEL` | Google Gemini via `google-generativeai` SDK |
| `fake` | none | deterministic local provider for tests |

Exactly one provider is constructed; it validates **only its own** credentials.
There is no silent fallback. A misconfigured provider fails loudly. Provider
code is isolated in `src/extraction/llm_client.py`; nothing else imports a
vendor SDK. Decoding uses temperature 0. A run whose provider calls all fail
writes no output file at all, so a null-filled JSON can never be mistaken for
a result.

---

## Output Format

Each field maps to a string or `null`:

```
Bid Number, Title, Due Date, Bid Submission Type, Term of Bid, Pre Bid Meeting,
Installation, Bid Bond Requirement, Delivery Date, Payment Terms,
Any Additional Documentation Required, MFG for Registration,
Contract or Cooperative to use, Model_no, Part_no, Product, contact_info,
company_name, Bid Summary, Product Specification
```

A field with no valid grounded candidate is `null`. Values keep their source
wording; only whitespace-level cleanup is applied, and dates are never rewritten
or reinterpreted.

The diagnostics file contains the full audit trail: chosen candidate, resolution
rule, tier, document label, addendum number, evidence quotes and chunk ids,
competing and superseded candidates, clarifications, rejected candidates with
reasons, and per-group extraction status.

---

## Tests

```bash
pytest
```

220 tests, fully offline. No API key required, no network calls. The LLM is
replaced by a deterministic fake provider. Coverage includes:

- PDF and HTML parsing
- Document chunking with overlap
- Hybrid retrieval (BM25 + embedding + RRF)
- Context construction and prompt budget
- Prompt rendering and leakage protection
- Response parsing and strict schema validation
- Evidence grounding (quote verification)
- Value normalization
- Deterministic precedence and addendum resolution
- End-to-end pipeline with the supplied corpus
- Streamlit AppTest

---

## Validation of Submitted Artifacts

The submitted `Bid1.flat.json` and `Bid2.flat.json` were assembled and validated
through the deterministic extraction and resolution layer over the supplied
source corpus. The available live Gemini quota was exhausted during development,
so the final artifacts are not live-model output. This limitation is disclosed
here rather than hidden.

The validation performed:

- **Source-grounded semantic audit** — every non-null value was verified against
  the actual source documents to confirm it is stated verbatim.
- **Exact 20-field schema validation** — automated verification that both JSON
  files contain exactly the required fields in the correct order.
- **Automated tests** — 220 tests exercise the complete pipeline from ingestion
  through resolution.
- **Null correctness** — every `null` value was verified against the source
  corpus to confirm the information genuinely cannot be established.

No benchmark numbers, accuracy percentages, or generalization claims are made.
The retrieval aliases were tuned against the two supplied bids; results on those
bids are therefore not evidence of generalization.

---

## Limitations

- **Live extraction was not used for the submitted artifacts.** The provider path
  requires a working Gemini key and available quota.
- Printed page labels are not reliably detected; evidence provenance cites chunk
  and document identifiers rather than printed page numbers.
- Tables are extracted from PDFs but not separately chunked; their text reaches
  the model through normal page text.
- Retrieval aliases in the field catalog were tuned against the two supplied bids.
- `app.py` is an optional Streamlit dashboard for inspecting parsing, chunking,
  and retrieval. It is a development aid, not part of the extraction path.
