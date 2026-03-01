"""
Knowledge module model & pipeline configuration.

Central defaults for all models used in knowledge pipelines.
To change the default model for indexing, retrieval, or reranking,
update the constants below — all Python code references these.

Frontend defaults (knowledge-bases/page.tsx, [kb_id]/page.tsx,
constants.ts) must be updated separately to match.
"""

# Indexing: tree building, ToC detection, summary generation, section splitting
PAGEINDEX_INDEXING_MODEL = "gpt-5-mini"

# Retrieval: LLM-based document selection and node selection (tree search)
PAGEINDEX_RETRIEVAL_MODEL = "gpt-4.1-mini"


# =============================================================================
# PageIndex Pipeline Thresholds
# =============================================================================

# --- ToC Detection ---
# Number of pages to scan from the start of the document when looking for a
# table of contents. Also used as the upper bound for retry scans.
# Used in: _pageindex_lib/page_index.py → find_toc_pages(), check_toc()
PAGEINDEX_TOC_CHECK_PAGE_NUM = 15

# --- ToC Extraction LLM Input ---
# Maximum tokens to send to the LLM in a single call when extracting or
# generating the ToC from page text. If the document exceeds this, pages are
# split into groups processed sequentially (generate_toc_init → generate_toc_continue).
# Used in: _pageindex_lib/page_index.py → page_list_to_group_text()
PAGEINDEX_TOC_MAX_TOKENS_PER_CHUNK = 16000

# --- Large-Node Recursive Splitting (page-aware pipeline) ---
# After the tree is built, any single node spanning more than this many pages
# AND exceeding PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE tokens gets recursively
# re-processed via meta_processor(mode='process_no_toc') to split it into children.
# Used in: _pageindex_lib/page_index.py → process_large_node_recursively()
PAGEINDEX_MAX_PAGE_NUM_EACH_NODE = 10
PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE = 16000

# --- Markdown Pipeline: Leaf-Node Splitting ---
# Tier 1: Leaf nodes above this token count are always sent to the LLM for
# structure inference (split into sub-sections).
# Used in: _pageindex_lib/page_index_md.py → split_large_sections()
PAGEINDEX_MAX_NODE_TOKENS = 1500

# Tier 2: Leaf nodes above this token count AND with >= PAGEINDEX_MIN_PARAGRAPH_COUNT
# blank-line-separated paragraphs are also sent for splitting.
# Used in: _pageindex_lib/page_index_md.py → split_large_sections()
PAGEINDEX_MIN_SPLIT_TOKENS = 500
PAGEINDEX_MIN_PARAGRAPH_COUNT = 4

# --- Thinning (optional subtree merging in markdown pipeline) ---
# When thinning is enabled, subtrees with cumulative token count below this
# are merged into their parent node.
# Used in: _pageindex_lib/page_index_md.py → tree_thinning_for_index()
PAGEINDEX_MIN_TOKEN_THRESHOLD = 2500

# --- Summary Generation ---
# Nodes with fewer tokens than this use their raw text as the summary
# (no LLM call). Above this, the LLM generates a concise summary.
# Used in: _pageindex_lib/page_index_md.py → get_node_summary()
PAGEINDEX_SUMMARY_TOKEN_THRESHOLD = 200

# --- Small-Sibling Merging (page-aware pipeline post-processing) ---
# Leaf nodes with fewer tokens than this are merged into an adjacent sibling
# or absorbed into their parent to reduce ToC noise.
# Used in: _pageindex_lib/page_index_md.py → _merge_small_siblings()
PAGEINDEX_MIN_MERGE_TOKENS = 200


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


# =============================================================================
# Metadata Enrichment defaults
# =============================================================================
# Default LLM model for metadata field extraction during enrichment.
METADATA_ENRICHMENT_DEFAULT_MODEL = "gpt-4.1-mini"


# =============================================================================
# Full Document strategy defaults
# =============================================================================
FULLDOC_SUMMARY_MODEL = "gpt-4.1-mini"
FULLDOC_SUMMARY_INPUT_CHARS = 128_000  # ~32K tokens at ~4 chars/token
FULLDOC_SUMMARY_MAX_TOKENS = 8000
FULLDOC_EMBEDDING_MODEL = "text-embedding-3-large"


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
GRAPHINDEX_EMBEDDING_MODEL = "text-embedding-3-large"

# Maximum number of concurrent LLM calls during referenced_nodes enrichment.
GRAPHINDEX_ENRICHMENT_MAX_CONCURRENT = 10


# =============================================================================
# Hybrid search defaults
# =============================================================================
# Default weight for vector similarity in hybrid RRF fusion.
# keyword_weight = 1.0 - HYBRID_DEFAULT_VECTOR_WEIGHT.
HYBRID_DEFAULT_VECTOR_WEIGHT = 0.5
