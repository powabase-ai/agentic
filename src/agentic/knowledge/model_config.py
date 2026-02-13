"""
PageIndex LLM model & pipeline configuration.

Central defaults for all LLM models used in the PageIndex pipeline.
To change the default model for indexing or retrieval, update the
constants below — all Python code references these.

Frontend defaults (knowledge-bases/page.tsx, [kb_id]/page.tsx) must
be updated separately to match.
"""

# Indexing: tree building, ToC detection, summary generation, section splitting
PAGEINDEX_INDEXING_MODEL = "gpt-4.1-mini"

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
