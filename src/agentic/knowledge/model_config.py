"""
Knowledge module model & pipeline configuration.

Single source of truth for all AI defaults used across knowledge
pipelines: indexing, retrieval, reranking, embedding, and chunking.
To change any default, update the constants below — all Python code
references these (including the host's own strategy registry).

Any frontend defaults the host maintains must be updated separately
to match.
"""

import os

# Indexing: tree building, ToC detection, summary generation, section splitting
PAGEINDEX_INDEXING_MODEL = "gpt-5-mini"

# Retrieval: LLM-based document selection and node selection (tree search)
PAGEINDEX_RETRIEVAL_MODEL = "gpt-5-mini"


# =============================================================================
# PageIndex Pipeline Thresholds
# =============================================================================

# --- ToC Detection ---
# Number of pages to scan from the start of the document when looking for a
# table of contents. Also used as the upper bound for retry scans.
# Used in: _pageindex_lib/page_index.py → find_toc_pages(), check_toc()
PAGEINDEX_TOC_CHECK_PAGE_NUM = 15

# --- Post-ToC Heading Scan ---
# Number of pages to scan *after* the detected ToC when looking for section
# headings to calibrate the page-number offset (physical_index − page).
# Used in: _pageindex_lib/page_index.py → process_toc_with_page_numbers()
PAGEINDEX_TOC_OFFSET_SCAN_PAGES = 50

# Maximum consecutive non-ToC pages tolerated inside a ToC sequence before
# concluding the ToC has ended. Handles blank/separator pages between ToC pages.
# Used in: _pageindex_lib/page_index.py → find_toc_pages()
PAGEINDEX_TOC_GAP_TOLERANCE = 3

# --- ToC Extraction LLM Input ---
# Maximum tokens to send to the LLM in a single call when extracting or
# generating the ToC from page text. If the document exceeds this, pages are
# split into groups processed sequentially (generate_toc_init → generate_toc_continue).
# Used in: _pageindex_lib/page_index.py → page_list_to_group_text()
PAGEINDEX_TOC_MAX_TOKENS_PER_CHUNK = 32000

# --- Large-Node Recursive Splitting (page-aware pipeline) ---
# After the tree is built, any single node spanning more than this many pages
# AND exceeding PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE tokens gets recursively
# re-processed via meta_processor(mode='process_no_toc') to split it into children.
# Used in: _pageindex_lib/page_index.py → process_large_node_recursively()
# Used by the page-aware pipeline (tree_parser → process_large_node_recursively())
PAGEINDEX_MAX_PAGE_NUM_EACH_NODE = 16
PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE = 16000

# --- Markdown Pipeline: Leaf-Node Splitting ---
# Tier 1: Leaf nodes above this token count are always sent to the LLM for
# structure inference (split into sub-sections).
# Used in: _pageindex_lib/page_index_md.py → split_large_sections()
# Used by the markdown pipeline (md_to_tree → split_large_sections())
PAGEINDEX_MAX_NODE_TOKENS = 16000

# Tier 2: Leaf nodes above this token count AND with >= PAGEINDEX_MIN_PARAGRAPH_COUNT
# blank-line-separated paragraphs are also sent for splitting.
# Used in: _pageindex_lib/page_index_md.py → split_large_sections()
PAGEINDEX_MIN_SPLIT_TOKENS = 16000
PAGEINDEX_MIN_PARAGRAPH_COUNT = 4

# --- Thinning (optional subtree merging in markdown pipeline) ---
# When thinning is enabled, subtrees with cumulative token count below this
# are merged into their parent node.
# Used in: _pageindex_lib/page_index_md.py → tree_thinning_for_index()
# By default, tree thinning is turned off; all nodes are preserved as is.
PAGEINDEX_MIN_TOKEN_THRESHOLD = 0

# --- Summary Generation ---
# Nodes with fewer tokens than this use their raw text as the summary
# (no LLM call). Above this, the LLM generates a concise summary.
# Used in: _pageindex_lib/page_index_md.py → get_node_summary()
PAGEINDEX_SUMMARY_TOKEN_THRESHOLD = 1500

# --- Document Description ---
# Maximum input tokens for the document-description prompt.
# The structure is truncated beyond this to stay within TPM / context limits.
PAGEINDEX_DOC_DESCRIPTION_MAX_TOKENS = 80_000

# --- LLM Concurrency ---
# Maximum concurrent LLM calls during PageIndex indexing.

# Controls an asyncio.Semaphore inside _llm_completion() to prevent
# OpenAI 429 rate-limit errors when processing large documents.
# Applies to both page_index and graph_index (Stage 1) strategies.
PAGEINDEX_LLM_MAX_CONCURRENT = 7


# =============================================================================
# Reranker defaults
# =============================================================================
# When reranking is enabled, the retrieval pipeline uses a two-stage approach:
#
#   Stage 1 (initial retrieval): Fetch `RERANKER_CANDIDATE_COUNT` candidate
#            chunks using the configured retrieval method (vector, hybrid,
#            full_text). This is a wider net to give the reranker enough
#            candidates to work with.
#
#   Stage 2 (reranking): The reranker model re-scores all candidates by
#            semantic relevance to the query. The top `top_k` results are
#            then returned to the caller.
#
# The user-facing `top_k` parameter controls how many chunks are ultimately
# returned — this stays the same whether reranking is on or off. The only
# thing that changes internally is that Stage 1 fetches more candidates
# (RERANKER_CANDIDATE_COUNT) to feed the reranker.
#
# These defaults can be overridden per-KB via retrieval_config.reranker.

RERANKER_DEFAULT_MODEL = "cohere/rerank-english-v3.0"

# Number of candidate chunks to retrieve in Stage 1 (before reranking).
# Higher values give the reranker more candidates to choose from, improving
# recall, but increase latency and cost. 20 is a good default balance.
RERANKER_CANDIDATE_COUNT = 20


# =============================================================================
# Query Enrichment defaults
# =============================================================================
QUERY_ENRICHMENT_DEFAULT_MODEL = "gpt-5-mini"
QUERY_ENRICHMENT_TEMPERATURE = 0
QUERY_ENRICHMENT_MAX_TOKENS = None  # No limit by default

# =============================================================================
# Metadata Enrichment defaults
# =============================================================================

# Default LLM model for metadata field extraction during enrichment.
METADATA_ENRICHMENT_DEFAULT_MODEL = "gpt-5-mini"

# Maximum input characters for item text sent to the enrichment LLM (text-only path).
# 0 means no truncation — the full item text is sent to the model.
METADATA_ENRICHMENT_MAX_INPUT_CHARS = 0

# Maximum input characters for item text when images are also provided (multimodal path).
# Multimodal calls include page images alongside text, so a lower limit may help
# keep the request within model context limits and reduce cost.
# 0 means no truncation.
METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL = 0

# Maximum number of page images to include in a single multimodal enrichment call.
# Additional images beyond this cap are silently dropped.
METADATA_ENRICHMENT_MAX_IMAGES = 10

# Default LLM response token budget for metadata enrichment calls.
METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS = 2000

# Maximum number of concurrent LLM calls during enrichment.
METADATA_ENRICHMENT_MAX_CONCURRENT = 10

# Number of items to process per database commit batch.
METADATA_ENRICHMENT_BATCH_SIZE = 50

# Maximum retry attempts when the LLM returns unparseable JSON.
METADATA_ENRICHMENT_MAX_RETRIES = 3


# =============================================================================
# Full Document strategy defaults
# =============================================================================
FULLDOC_SUMMARY_MODEL = "gpt-5-mini"
FULLDOC_SUMMARY_INPUT_CHARS = 128_000  # ~32K tokens at ~4 chars/token
FULLDOC_SUMMARY_MAX_TOKENS = 8000
FULLDOC_EMBEDDING_MODEL = "text-embedding-3-small"


# =============================================================================
# GraphIndex strategy defaults
# =============================================================================

# LLM model for ToC building (tree structure, section splitting, summary generation).
# Uses the same model as PageIndex since the ToC pipeline is shared.
GRAPHINDEX_INDEXING_MODEL = "gpt-5-mini"

# LLM model for internal referenced_nodes enrichment.
# Identifies cross-section references within each document during indexing.
GRAPHINDEX_ENRICHMENT_MODEL = "gpt-5-mini"

# Embedding model for node summary embeddings.
# Summaries + metadata are embedded for vector similarity retrieval.
GRAPHINDEX_EMBEDDING_MODEL = "text-embedding-3-small"

# Maximum number of concurrent LLM calls during referenced_nodes enrichment.
GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT = 7
GRAPHINDEX_ENRICHMENT_BATCH_SIZE = (
    50  # nodes per batch; bounds per-batch retained memory + billing granularity
)

# Maximum tokens for LLM response during referenced_nodes enrichment.
GRAPHINDEX_ENRICHMENT_MAX_TOKENS = 5000

# Maximum input characters for node text sent to the enrichment LLM.
# 0 means no truncation (node texts are already length-bounded by the indexing pipeline).
GRAPHINDEX_ENRICHMENT_MAX_INPUT_CHARS = 0

# Whether to include node summaries in the ToC context for enrichment prompts.
# Off by default — titles + node IDs are sufficient for cross-reference detection,
# and summaries can bloat prompts to 60k+ tokens on large documents.
GRAPHINDEX_ENRICHMENT_TOC_INCLUDE_SUMMARIES = False

# Maximum retry attempts when the LLM returns unparseable JSON.
GRAPHINDEX_ENRICHMENT_MAX_JSON_RETRIES = 3


# =============================================================================
# Chunk Embed strategy defaults
# =============================================================================
CHUNK_EMBED_EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_EMBED_DEFAULT_CHUNK_SIZE = 2000
CHUNK_EMBED_DEFAULT_OVERLAP = 50


# =============================================================================
# Doc2JSON strategy defaults
# =============================================================================
# LLM model for sliding window extraction (summary + JSON extraction)
# Must be multimodal-capable when use_images=True
DOC2JSON_EXTRACTION_MODEL = "gpt-5-mini"

# Embedding model for summary embeddings
DOC2JSON_EMBEDDING_MODEL = "text-embedding-3-small"

# Sliding window configuration (in tokens) - used when use_images=False
DOC2JSON_DEFAULT_WINDOW_SIZE = 4000
DOC2JSON_DEFAULT_WINDOW_OVERLAP = 200

# Image-based extraction (multimodal mode)
DOC2JSON_USE_IMAGES = False  # Extract from page images instead of text
DOC2JSON_DEFAULT_PAGES_PER_WINDOW = 3  # Pages per LLM call when using images

# LLM output token limits
DOC2JSON_EXTRACTION_MAX_TOKENS = 4000
DOC2JSON_SUMMARY_MAX_TOKENS = 2000

# Maximum retry attempts for JSON parsing failures
DOC2JSON_MAX_RETRIES = 3

# Maximum tokens per single embedding API request.
# OpenAI's text-embedding-3-small allows max 300k tokens/request.
# Use 250k to leave margin for overhead.
EMBEDDING_MAX_TOKENS_PER_BATCH = 200_000


# =============================================================================
# Agent context window defaults
# =============================================================================
# Default maximum tokens for the formatted context sent to the agent LLM.
DEFAULT_MAX_CONTEXT_TOKENS = 32000


# =============================================================================
# Hybrid search defaults
# =============================================================================
# Default weight for vector similarity in hybrid RRF fusion.
# keyword_weight = 1.0 - HYBRID_DEFAULT_VECTOR_WEIGHT.
HYBRID_DEFAULT_VECTOR_WEIGHT = 0.5


# =============================================================================
# Agent defaults
# =============================================================================
# Default LLM model for agent execution when no model is specified.
AGENT_DEFAULT_MODEL = "gpt-4o-mini"


# =============================================================================
# Extraction defaults
# =============================================================================
# Default extraction method for PDFs. "auto" uses the fallback chain below.
EXTRACTION_DEFAULT_METHOD = "auto"  # "auto", "mistral", "opendataloader", "paddleocr", "lighton", "llamaparse", "fitz", "pdfplumber"

# Ordered fallback chain used when extraction_model is "auto".
# paddleocr, mistral and llamaparse are NOT in the chain — user-selectable only.
EXTRACTION_FALLBACK_CHAIN = ["lighton", "opendataloader", "fitz", "pdfplumber"]

# --- PaddleOCR-VL configuration ---
PADDLEOCR_DEFAULT_BASE_URL = "https://s1naa1u9e7v9f7ub.aistudio-app.com"
PADDLEOCR_LAYOUT_PARSING_PATH = "/layout-parsing"
PADDLEOCR_FILE_TYPE_PDF = 0
PADDLEOCR_FILE_TYPE_IMAGE = 1
PADDLEOCR_TIMEOUT = 120  # seconds — large PDFs can take time
PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY = False
PADDLEOCR_USE_DOC_UNWARPING = False
PADDLEOCR_USE_CHART_RECOGNITION = False

# --- LightOnOCR configuration ---
LIGHTON_DEFAULT_BASE_URL = "https://openai.inference.de-txl.ionos.com/v1"
LIGHTON_DEFAULT_MODEL = "lightonai/LightOnOCR-2-1B"
LIGHTON_MAX_TOKENS = 4096
LIGHTON_TEMPERATURE = 0.2
LIGHTON_TOP_P = 0.9
LIGHTON_TIMEOUT = 60  # seconds per page

# --- LlamaParse (LlamaCloud) configuration ---
# Uses the v2 Parse API with the most-advanced "Agentic Plus" tier (the standard
# for advanced OCR). Flow: upload file -> get file_id, POST /api/v2/parse with
# the tier, poll the job, then fetch per-page markdown. User-selectable only
# ("llamaparse"); billed as advanced OCR. agentic_plus = 45 credits/page
# ($0.05625 at $1.25/1k credits) — the cost basis the advanced_ocr price is set
# against. (NOTE: v1 parse_document_with_agent is 90 credits/page — do NOT use
# it, it would double the cost.)
LLAMAPARSE_DEFAULT_BASE_URL = "https://api.cloud.llamaindex.ai"
LLAMAPARSE_TIER = "agentic_plus"  # most advanced tier; v2 auto-selects the model
# Pinned to a dated agentic_plus version so output schema + COGS stay reproducible
# (the 2x markup is anchored to the current tier cost; "latest" would let upstream
# re-tier our COGS silently). Override via the LLAMAPARSE_VERSION env when bumping
# — an invalid value makes the API echo the list of valid versions in its 4xx body.
LLAMAPARSE_VERSION = os.getenv("LLAMAPARSE_VERSION", "2026-06-11")
LLAMAPARSE_TIMEOUT = 120  # seconds per HTTP request
LLAMAPARSE_POLL_INTERVAL = 5  # seconds between job-status polls
# agentic_plus (full-document agent + LVM) is slow on large docs. The Celery
# extraction task allows up to 6h, so give the job generous headroom rather than
# spuriously timing out and falling back to local extraction.
LLAMAPARSE_MAX_POLL_SECONDS = 1800  # 30 min


# =============================================================================
# Global LiteLLM debug configuration
# =============================================================================
import litellm  # noqa: E402

if os.environ.get("LITELLM_DEBUG", "").lower() in ("1", "true"):
    litellm._turn_on_debug()

# -----------------------------------------------------------------------------
# Disable LiteLLM's aiohttp transport (default-on since ~1.74; we run 1.83.14).
# -----------------------------------------------------------------------------
# The aiohttp transport caches a ClientSession/connector bound to the event loop
# it was first created on. Indexing runs inside Celery tasks via asyncio.run(),
# which creates a FRESH event loop per task — so once a worker has handled a
# task, the cached session is bound to a now-dead loop and the next
# litellm.acompletion fails instantly ("Timeout on reading data from socket",
# time taken=0.0s, "coroutine ... was never awaited"). Combined with
# num_retries=0 in the PageIndex pipeline, one stale-session blip fails the
# whole indexing job. Disabling the aiohttp transport falls back to httpx, which
# builds per-request clients and is event-loop-safe across asyncio.run() calls.
# Set LITELLM_AIOHTTP_TRANSPORT=1 to re-enable aiohttp if a future litellm fixes
# the loop affinity and we want the perf back. The hasattr guard avoids silently
# creating a dead attribute if litellm ever renames the flag — that rename is
# caught loudly by test_model_config_resiliency.py (which asserts the attribute
# exists), so this degrades gracefully in prod while CI fails on the regression.
if (
    hasattr(litellm, "disable_aiohttp_transport")
    and os.environ.get("LITELLM_AIOHTTP_TRANSPORT", "").lower() not in ("1", "true")
):
    litellm.disable_aiohttp_transport = True


# -----------------------------------------------------------------------------
# PageIndex / GraphIndex LLM call resiliency
# -----------------------------------------------------------------------------
def _int_env(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* (with a warning) on a
    malformed value rather than raising at import time. These constants live in
    a foundational module imported by ~all of agentic.knowledge (retrieval,
    chat, query enrichment — not just indexing), so a bad operator-set value
    must not take the whole module down."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning(
            "Invalid %s=%r (not an int); falling back to %d", name, raw, default
        )
        return default


# Per-call timeout (seconds) for the tree-building / ToC / summary LLM calls in
# _pageindex_lib/utils.py::_llm_completion. The previous hardcoded 60s was too
# low for reasoning models (gpt-5* / o-series) doing large ToC extraction on big
# documents — they legitimately take 60-90s+, which tripped litellm.Timeout and,
# with num_retries=0 (PR #445), failed the ENTIRE indexing job on a single slow
# response. Env-overridable: PAGEINDEX_LLM_TIMEOUT.
PAGEINDEX_LLM_TIMEOUT = _int_env("PAGEINDEX_LLM_TIMEOUT", 300)

# litellm retry count for those calls. Default 1 gives a single retry on a
# transient timeout/5xx without the unbounded re-send of PR #445's original 3
# (each retry re-sends a large prompt — the bounded-memory concern). Set to 0 to
# fully restore #445's no-retry behavior. Env-overridable: PAGEINDEX_LLM_NUM_RETRIES.
PAGEINDEX_LLM_NUM_RETRIES = _int_env("PAGEINDEX_LLM_NUM_RETRIES", 1)
