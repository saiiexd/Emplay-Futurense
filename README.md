# RFP Document Intelligence

Extracts 20 structured fields from a folder of heterogeneous procurement documents
(PDF and HTML) and writes them as JSON, keeping every answer traceable to the
sentence it came from.

## The problem

A public-sector bid is not one document. It is a main solicitation, one or more
addenda that amend it, a portal listing page, sometimes a manufacturer
specification sheet, and blank forms such as affidavits. The facts a buyer needs
are scattered across these, written in different words, and sometimes changed
after the fact: an addendum can move the due date or revise a requirement.

Two failure modes matter more than raw accuracy here:

1. **Invented answers.** A language model asked to fill 20 fields will fill them,
   whether or not the documents support it.
2. **Wrong document wins.** A portal page or a superseded paragraph can look just
   as convincing as the clause that actually applies.

This system is built around those two problems. Every non-null answer must quote
text that provably exists in a source chunk, and the decision about *which*
candidate wins is made by deterministic Python, never by the model.

## Supported inputs

- **PDF** - parsed page by page with PyMuPDF, including table extraction and
  detection of pages with little or no text layer.
- **HTML** - parsed with BeautifulSoup; scripts, styles and HTML comments are
  stripped so developer notes and analytics markup never become evidence.

Documents are classified as `rfp_main`, `addendum`, `portal_listing`,
`specification` or `affidavit`. Addendum numbers are read from the document text,
not from filenames.

## Pipeline

```
bid folder
   ├─ BidManager            parse each PDF/HTML into normalized sections
   ├─ DocumentChunker       deterministic ~1000-char chunks with overlap
   ├─ DocumentRegistry      chunk -> document provenance, addendum numbers
   ├─ HybridSearcher        BM25 + embeddings fused with RRF
   ├─ ContextBuilder        select evidence per field group, assign handles
   ├─ prompts.py            render messages + strict JSON schema
   ├─ ExtractionEngine      call the LLM provider, parse the reply
   ├─ GroundingValidator    verify every quote against real chunk text
   ├─ RuleCandidateGenerator deterministic candidates for self-named forms
   ├─ CandidateResolver     precedence and addendum resolution
   └─ final record          20-field JSON + separate diagnostics
```

### Retrieval

Retrieval selects *which evidence the model may see*; it is not used to rank
answers. Per field, BM25 runs over the catalog's aliases and an embedding query
runs over a natural-language description of the field; the two are fused with
Reciprocal Rank Fusion, with a lexical floor so an exact identifier match cannot
be pushed out by a weak semantic score.

Small documents are included whole, so nothing depends on ranking for them.
Retrieval only does real work on the large main solicitation. Ordering is fully
deterministic, with ties broken by chunk id.

### The LLM

The model does one job: **semantic candidate extraction**. For a group of related
fields it receives evidence excerpts and returns candidate values, each with an
evidence handle and a verbatim quote.

It is deliberately not told document types, filenames, addendum numbers, page
numbers or retrieval scores. It cannot see which document is authoritative, so it
cannot be tempted to rank them. Its reply is validated against a strict JSON
schema whose enums are limited to the field names and evidence handles of that
specific call.

### Grounding

Before a candidate can influence any answer:

- its evidence handle must resolve to a chunk that was actually supplied,
- the document type must be eligible for that field per the catalog,
- the quote must be found in the chunk by exact/whitespace/case/compact matching,
- the value must be contained in its own quote.

Matching is literal. There is no fuzzy or semantic similarity anywhere in
grounding, because a near-match is exactly how a fabricated value would slip
through. Rejected candidates keep their rejection code and stay in diagnostics.

### Deterministic addendum resolution

Candidates are ranked by document provenance:

| Tier | Source |
|-----:|--------|
| 40 | addendum that **amends** the field (higher addendum number wins) |
| 30 | main solicitation |
| 25 | addendum that mentions the field without amending it |
| 20 | specification document |
| 10 | portal / listing page |
|  5 | affidavit or attached form |

An addendum reaches tier 40 only when generic amendment language ("the new due
date", "is hereby amended", "extends the deadline", "delete ... insert") appears
near its quote. Question-and-answer text without such language is recorded as a
*clarification*: it is kept for audit but does not change the value. This is why a
vendor question inside an addendum cannot silently rewrite the solicitation, and
why a non-amending addendum sits *below* the document it was issued against.

When an amendment wins, the earlier value is retained as `superseded` rather than
discarded. Fields the catalog marks as aggregated (required documents, contacts,
product specification) collect evidence across eligible documents instead of
picking one - unless an addendum amends the list, which replaces it.

No value, date, filename or identifier from any specific corpus appears in the
resolution logic. Corpus-specific expectations used by regression tests live in
`tests/fixtures/`.

## Output

`data/output/<bid>.json` - the answer, exactly these 20 keys in this order:

```
Bid Number, Title, Due Date, Bid Submission Type, Term of Bid, Pre Bid Meeting,
Installation, Bid Bond Requirement, Delivery Date, Payment Terms,
Any Additional Documentation Required, MFG for Registration,
Contract or Cooperative to use, Model_no, Part_no, Product, contact_info,
company_name, Bid Summary, Product Specification
```

A field with no valid grounded candidate is `null`. Values keep their source
wording; only whitespace-level cleanup is applied, and dates are never rewritten
or reinterpreted against the current date - the corpus contains valid historical
2024 solicitations.

`data/output/<bid>.diagnostics.json` - the audit trail, kept separate from the
answer: chosen candidate, resolution rule, tier, document label, addendum number,
evidence quotes and chunk ids, competing and superseded candidates,
clarifications, rejected candidates with reasons, and per-group extraction status.

`data/output/<bid>.flat.json` contains the assignment-facing flat artifact. The
submitted Bid1 and Bid2 flat artifacts were assembled and validated from the
supplied source corpus through the deterministic extraction and resolution layer
because the available live Gemini quota was exhausted; they are not live-model
output.

## Setup

```bash
cd Project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Configuring the LLM provider

Copy `.env.example` to `.env` (git-ignored) and set one provider. **OpenRouter is
the active default**; OpenAI and native Mistral are supported alternatives.

```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EXTRACTION_MODEL=qwen/qwen3-30b-a3b:free
```

| `LLM_PROVIDER` | Variables it reads | Transport |
|---|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_EXTRACTION_MODEL` | OpenAI-compatible API at `https://openrouter.ai/api/v1`, via the `openai` SDK |
| `openai` | `OPENAI_API_KEY`, `OPENAI_EXTRACTION_MODEL` | OpenAI directly |
| `mistral` | `MISTRAL_API_KEY`, `MISTRAL_EXTRACTION_MODEL`, `MISTRAL_BASE_URL` | native `mistralai` SDK |

Exactly one provider is constructed, and it validates **only its own**
credentials: selecting `openrouter` never requires an OpenAI or Mistral key, and
vice versa. There is no silent fallback between providers - a misconfigured
provider fails loudly.

Provider code is isolated in `src/extraction/llm_client.py`; nothing else imports
a vendor SDK. Decoding uses temperature 0 and a fixed seed. Extraction requests
carry the strict JSON schema built from the field catalog. Without a usable key
the program exits with a clear message rather than guessing, and **a run whose
provider calls all fail writes no output file at all**, so a null-filled JSON can
never be mistaken for a result.

> **OpenRouter keys:** use a normal inference key from
> <https://openrouter.ai/keys>. A *provisioning/management* key authenticates
> against `/auth/key` but returns `401 User not found` on `/chat/completions`.

## Running the whole system

From `Project/`, with `.env` configured, this one command runs everything -
ingestion, chunking, retrieval, context construction, the six LLM extraction
calls, parsing, grounding, precedence resolution, and final JSON output - for
both supplied bids:

```bash
python main.py --bid ../Bid1 --bid ../Bid2
```

Point `--bid` at any folder of PDF/HTML documents; repeat it for more bids.

It writes four files into `data/output/`:

```
Bid1.json                 Bid1.diagnostics.json
Bid2.json                 Bid2.diagnostics.json
```

and prints a per-field summary showing, for each of the 20 fields, whether it
resolved, which rule decided it, and which document it came from.

**Exit codes:** `0` all groups completed · `2` provider not configured (missing
key) · `3` the run finished but one or more groups failed at the provider, so
the output is incomplete rather than a genuine "not found".

> [!WARNING]
> **Live API Constraint:** The pipeline executes 6 concurrent group calls per bid. The Gemini Free Tier enforces a strict rate limit of 15 Requests Per Minute (RPM) which causes `RateLimitError` and drops extraction groups, preventing successful local JSON generation with free keys. The pipeline handles this gracefully by aborting the failed extraction groups (exit code 3) without manufacturing fake JSON. To fully execute live extraction across all bids, a paid tier or higher-quota Gemini API key is strictly required.

Options: `--out` (default `data/output`), `--embeddings fake|openai`
(default `fake`, which keeps runs cheap and deterministic; `openai` enables the
real semantic retrieval channel), `--max-correction-retries`.

### First run, from a clean checkout

```bash
cd Project
python -m venv venv
venv\Scripts\activate                      # Windows; use source venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env                     # then put a real key in .env
python main.py --bid ../Bid1 --bid ../Bid2
```

### One-Click Execution (Backend + Frontend)

For convenience, you can run both the extraction backend and the Streamlit frontend dashboard in a single command using the provided helper scripts:

**On Windows (PowerShell/CMD):**
```cmd
.\run.bat
```

**On macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

These scripts will automatically execute the backend extraction against the sample bids and then launch the interactive dashboard in your browser.

If the model returns malformed JSON, the original contract is re-sent once with a
correction message. Transient provider errors retry with bounded backoff. A
response that cannot be parsed is reported, not silently dropped.

## Tests

```bash
pytest
```

249 tests, fully offline. They need no API key and make no network calls; the LLM
is replaced by a deterministic fake provider. Coverage includes parsing,
chunking, retrieval, context construction, prompt rendering and leakage
protection, response parsing, grounding, normalization, precedence, and the
20-field assembly. Several tests run against the supplied corpus to check real
structural behaviour, including an addendum that amends a field.

## Two design decisions worth explaining

**Why group-level extraction.** The 20 fields are extracted in six calls
(identity, schedule, terms, products, specifications, contacts/summary) rather
than one call for everything or one per field. One call for 20 fields makes the
context enormous and degrades recall on the fields at the end. One call per field
means 20x the cost and latency and loses the context that related fields share -
a date sits next to the deadline it belongs to. Grouping related fields keeps
each prompt focused while reusing the same evidence.

**Why resolution is deterministic.** Choosing between an original due date and an
amended one is not a language problem; it is a rule about document authority. A
model asked to choose would be unpredictable across runs, unable to explain
itself in auditable terms, and prone to preferring whichever text it saw last.
Python applies fixed precedence rules to provenance the registry already knows,
so the same candidates always produce the same answer and every decision can be
replayed from the diagnostics file.

## Status and limitations

- **Live extraction accuracy is unverified.** The pipeline has been executed
  against the configured OpenRouter endpoint, but every request was rejected
  (`401`), so the model never returned a candidate. The provider path, the
  strict-schema request and the failure handling are therefore exercised;
  extraction quality is not. Everything else described here is verified offline
  with a deterministic fake provider over the real documents, from ingestion
  through resolution. No accuracy figure is claimed because none has been
  measured. Running the system with a working inference key is the one
  outstanding step.
- Printed page labels are not reliably detected, so evidence provenance cites
  chunk and document identifiers rather than printed page numbers.
- Tables are extracted from PDFs but are not separately chunked; their text
  reaches the model through normal page text.
- Retrieval aliases in the field catalog were tuned against the two supplied
  bids. Results on those two bids are therefore not evidence of generalization.
- `app.py` (with helpers in `src/ui/`) is an optional Streamlit dashboard for
  inspecting parsing, chunking and retrieval on uploaded files. It is a
  development aid, not part of the extraction path, and it needs no API key:
  `streamlit run app.py`. Extraction itself runs from `main.py` so that results
  are reproducible and written to disk.
