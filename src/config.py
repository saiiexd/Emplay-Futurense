CHUNK_MAX = 1000
CHUNK_OVERLAP = 200
SMALL_DOC_CHARS = 15000
FRONT_MATTER_PAGES = (1, 2)
RETRIEVAL_TOP_K = 6
LEXICAL_FLOOR = 2
NEIGHBOR_RADIUS = 1

# Explicit Groups based on catalog
PAGE_EXPANSION_GROUPS = ("products", "specifications")
GROUP_ORDER = ["identity", "schedule", "terms", "products", "specifications", "contacts_summary"]

MAX_EVIDENCE_CHARS_PER_CALL = 100_000
MAX_EXCERPTS_PER_CALL = 120
MAX_PROMPT_CHARS = 110_000
HANDLE_HEX_LEN = 8
MAX_QUOTE_CHARS = 600
MAX_VALUE_CHARS = 600
MAX_EVIDENCE_PER_CANDIDATE = 3
MAX_CANDIDATES_PER_RESPONSE = 200
MAX_REASON_CHARS = 200
MAX_CORRECTION_CHARS = 500

GROUP_MAX_OUTPUT_TOKENS = {
    "identity": 2000,
    "schedule": 3000,
    "terms": 4000,
    "products": 3000,
    "specifications": 12000,
    "contacts_summary": 3000
}

RETRY_PARAMETERS = {
    "max_retries": 3,
    "backoff_factor": 2.0
}

CACHE_ROOT = "data/cache"
PROMPT_VERSION = "p4.2.0"

# --- Stage 4.3 LLM extraction ------------------------------------------------
# Provider selection and credentials come from the environment. No secret is
# ever stored here; only the names of the variables to read.
LLM_PROVIDER_ENV = "LLM_PROVIDER"

# Google Gemini API
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_EXTRACTION_MODEL"
DEFAULT_GEMINI_MODEL = "models/gemini-flash-latest"

DEFAULT_LLM_PROVIDER = "gemini"

# Deterministic decoding: temperature 0 and a fixed seed so repeated runs of the
# same package produce the same extraction as far as the provider allows.
LLM_TEMPERATURE = 0.0
LLM_SEED = 7
LLM_TIMEOUT_S = 90

# Bounded retries. Correction retries re-send the same contract plus the
# parser's complaint; transient retries cover rate limits and 5xx responses.
LLM_MAX_CORRECTION_RETRIES = 1
LLM_MAX_TRANSIENT_RETRIES = RETRY_PARAMETERS["max_retries"]
LLM_BACKOFF_FACTOR = RETRY_PARAMETERS["backoff_factor"]

OUTPUT_ROOT = "data/output"
