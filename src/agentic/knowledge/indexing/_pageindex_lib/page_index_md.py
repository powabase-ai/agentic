"""
Markdown and page-aware indexing pipelines for PageIndex.

This module contains two entry points:

1. **md_to_tree()** — The markdown pipeline. Parses markdown headers (#, ##, etc.)
   into a hierarchical tree, then uses LLM calls to split oversized leaf nodes.
   Used when documents arrive as a single markdown string.

2. **md_to_tree_from_pages()** — The page-aware pipeline. Delegates to the
   reference PageIndex tree_parser() for full ToC detection/verification/correction,
   then adds Agentic-specific post-processing (parent text trimming, small sibling
   merging). Used when per-page text arrays are available from PDF extractors.

Both return the same format: {"doc_name": "...", "structure": [...tree...]}
where each node in the tree carries title, node_id, text, and optional summaries.

The tree is later split into toc_structure + sections by the outer page_index.py
wrapper via _extract_sections_from_structure().

Key functions (in pipeline order for the markdown pipeline):
    extract_nodes_from_markdown()  — Regex scan for # headers
    extract_node_text_content()    — Assign text slices between headers
    update_node_list_with_text_token_count() — Compute cumulative token counts
    tree_thinning_for_index()      — Merge small subtrees (optional)
    build_tree_from_nodes()        — Stack-based nesting by header level
    split_large_sections()         — LLM-split oversized leaf nodes
    write_node_id()                — Sequential ID assignment (from utils)
    generate_summaries_for_structure_md() — Concurrent per-node LLM summaries

Agentic-specific additions (not in the reference PageIndex library):
    _trim_parent_text()     — Prevents parent/child text duplication for retrieval
    _merge_small_siblings() — Merges tiny leaf nodes into adjacent siblings
"""

import asyncio
import json
import logging
import re
import os
try:
    from .utils import *
except ImportError:
    from utils import *

logger = logging.getLogger(__name__)


# =============================================================================
# Summary Generation
# =============================================================================

async def get_node_summary(node, summary_token_threshold=PAGEINDEX_SUMMARY_TOKEN_THRESHOLD, model=None):
    """Generate a summary for a single tree node.

    Optimization: if the node's text is already short enough (below
    summary_token_threshold), the text itself serves as the summary and
    no LLM call is made. This avoids wasting API calls on nodes that
    are already concise enough to be their own summary.

    Args:
        node: Tree node dict with "text" key.
        summary_token_threshold: Token count below which the raw text is
            returned as-is instead of generating an LLM summary.
        model: LLM model identifier for summary generation.

    Returns:
        Summary string — either the raw text (if short) or LLM-generated.
    """
    node_text = node.get('text')
    num_tokens = count_tokens(node_text, model=model)
    if num_tokens < summary_token_threshold:
        # Text is short enough to be its own summary — skip the LLM call
        return node_text
    else:
        # Text is long — ask the LLM to produce a concise summary
        return await generate_node_summary(node, model=model)


async def generate_summaries_for_structure_md(structure, summary_token_threshold, model=None):
    """Generate summaries for all nodes in the tree concurrently.

    Flattens the tree into a list, fires off one get_node_summary() task
    per node via asyncio.gather (fully concurrent), then assigns the results
    back to the appropriate field:

    - **Leaf nodes** (no children) get a "summary" field — describes what
      this section contains.
    - **Parent nodes** (has children) get a "prefix_summary" field — describes
      what topics are covered under this branch. Named "prefix" because the
      parent's own text is typically just the intro prefix before children.

    This distinction matters for retrieval: the LLM sees prefix_summary to
    understand branch scope, and summary to understand leaf content.

    Args:
        structure: The tree structure (list of root nodes). Mutated in-place.
        summary_token_threshold: Passed through to get_node_summary().
        model: LLM model identifier.

    Returns:
        The same structure list, now with summary/prefix_summary fields added.
    """
    # structure_to_list() (from utils) flattens the tree into a pre-order list
    # of all node dicts, preserving references so mutations affect the tree.
    nodes = structure_to_list(structure)
    tasks = [get_node_summary(node, summary_token_threshold=summary_token_threshold, model=model) for node in nodes]

    # Fire all summary requests concurrently — this is the most LLM-intensive
    # step, making one call per node (minus those below the token threshold).
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        if not node.get('nodes'):
            # Leaf node — "summary" describes this section's content
            node['summary'] = summary
        else:
            # Parent node — "prefix_summary" describes the branch scope
            node['prefix_summary'] = summary
    return structure


# =============================================================================
# Step 1: Header Extraction
# =============================================================================

def extract_nodes_from_markdown(markdown_content):
    """Scan markdown text for headers and return a flat list of header nodes.

    Uses regex to find lines matching `^#{1,6} title` (ATX-style markdown
    headers). Correctly ignores headers inside fenced code blocks (``` ... ```)
    to avoid false positives from code examples.

    Args:
        markdown_content: Raw markdown text string.

    Returns:
        Tuple of (node_list, lines) where:
        - node_list: List of dicts with keys:
            - "node_title": The header text (without # prefix)
            - "line_num": 1-based line number where the header appears
            - "level": Header depth (1 for #, 2 for ##, etc.)
        - lines: The markdown split into a list of lines (used by later steps
          for text slicing).

        If the document has no headers, node_list will be empty and the caller
        should fall back to wrapping the entire document as a single root node.
    """
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []

    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()

        # Toggle code block state on ``` delimiters.
        # Headers inside code blocks are not real headers.
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue

        if not stripped_line:
            continue

        # Only match headers when outside code blocks
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                level = len(match.group(1))  # Number of # characters = depth
                title = match.group(2).strip()
                node_list.append({
                    'node_title': title, 'line_num': line_num, 'level': level,
                })

    return node_list, lines


# =============================================================================
# Step 2: Text Assignment
# =============================================================================

def extract_node_text_content(node_list, markdown_lines):
    """Assign text content to each header node by slicing between headers.

    Each node gets all the text from its header line up to (but not including)
    the next header's line. This means:
    - Leaf nodes get exactly their section content.
    - Parent nodes get their content PLUS all descendant content (overlapping).

    This overlap is acceptable because:
    - In the markdown pipeline, the tree is built purely from header nesting,
      and each node's text is self-contained for summary generation.
    - The overlap doesn't cause problems for retrieval because sections are
      fetched individually by node_id.

    Args:
        node_list: Flat list of header dicts from extract_nodes_from_markdown().
            Each must have "node_title", "line_num", and optionally "level".
        markdown_lines: The full markdown split into lines.

    Returns:
        List of processed node dicts with keys: title, line_num, level, text.
        The text is the joined lines from this header to the next header.
    """
    all_nodes = []
    for node in node_list:
        # Determine header level — prefer pre-computed level from extraction,
        # fall back to re-parsing the header line (for backwards compatibility
        # with node lists that don't include level).
        if 'level' in node:
            level = node['level']
        else:
            line_content = markdown_lines[node['line_num'] - 1]
            header_match = re.match(r'^(#{1,6})', line_content)
            if header_match is None:
                logger.warning(f"Line {node['line_num']} does not contain a valid header: '{line_content}'")
                continue
            level = len(header_match.group(1))

        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': level,
        }
        all_nodes.append(processed_node)

    # Assign text: each node gets lines from its header to the next header.
    # The last node gets everything to the end of the document.
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1  # Convert to 0-based index
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1  # Up to next header
        else:
            end_line = len(markdown_lines)  # Last node: to end of file

        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
    return all_nodes


# =============================================================================
# Optional: Thinning (merge small subtrees)
# =============================================================================

def update_node_list_with_text_token_count(node_list, model=None):
    """Compute cumulative token counts for each node (own text + all descendants).

    This is a pre-processing step for tree_thinning_for_index(). Each node's
    text_token_count represents the TOTAL tokens if this node and all its
    children were merged into a single section. This lets the thinning step
    decide which subtrees are small enough to collapse.

    The flat node_list uses "level" to determine parent/child relationships:
    nodes with higher level numbers that appear after a lower-level node are
    its descendants (until a node with equal or lower level is encountered).

    Processing is done bottom-up (end to beginning) so children are counted
    before their parents.

    Args:
        node_list: Flat list of node dicts with "text", "level" keys.
        model: Model identifier for token counting.

    Returns:
        Copy of node_list with "text_token_count" added to each node.
    """

    def find_all_children(parent_index, parent_level, node_list):
        """Find all direct and indirect descendants of a node in the flat list.

        Walks forward from parent_index, collecting all nodes with deeper
        levels until hitting a node at the same or shallower level (which
        would be a sibling or ancestor, not a descendant).
        """
        children_indices = []

        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']

            # Same or shallower level means we've left this subtree
            if current_level <= parent_level:
                break

            children_indices.append(i)

        return children_indices

    result_list = node_list.copy()

    # Process bottom-up so children's token counts are ready when we reach parents
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        current_level = current_node['level']

        children_indices = find_all_children(i, current_level, result_list)

        # Cumulative text = this node's text + all descendants' text
        node_text = current_node.get('text', '')
        total_text = node_text

        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text

        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)

    return result_list


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    """Merge subtrees that are too small to be useful as separate sections.

    When a parent node's cumulative token count (itself + all descendants) is
    below min_node_token, all its children are absorbed into the parent:
    - Children's text is concatenated into the parent's text
    - Children are removed from the list
    - Parent's token count is updated

    This reduces tree granularity for documents with many tiny sections,
    producing fewer but more substantial sections for retrieval.

    Processing is bottom-up so the deepest small subtrees are merged first,
    which may cause their parents to grow large enough to survive thinning.

    Args:
        node_list: Flat list with "text_token_count" from
            update_node_list_with_text_token_count().
        min_node_token: Minimum cumulative tokens to keep a subtree intact.
            Subtrees below this are merged into their parent.
        model: Model identifier for token counting.

    Returns:
        Thinned copy of the node list with small subtrees merged.
    """
    def find_all_children(parent_index, parent_level, node_list):
        children_indices = []

        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']

            if current_level <= parent_level:
                break

            children_indices.append(i)

        return children_indices

    result_list = node_list.copy()
    nodes_to_remove = set()

    # Process bottom-up: deepest nodes first
    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue

        current_node = result_list[i]
        current_level = current_node['level']

        total_tokens = current_node.get('text_token_count', 0)

        # If this subtree is below the threshold, absorb all children
        if total_tokens < min_node_token:
            children_indices = find_all_children(i, current_level, result_list)

            # Collect children's text and mark them for removal
            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '')
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)

            # Merge children's text into this node
            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith('\n'):
                        merged_text += '\n\n'
                    merged_text += child_text

                result_list[i]['text'] = merged_text
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)

    # Remove absorbed children (in reverse order to preserve indices)
    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)

    return result_list


# =============================================================================
# Step 3: Build Hierarchical Tree
# =============================================================================

def build_tree_from_nodes(node_list):
    """Convert a flat header list into a nested tree using header levels.

    Uses a stack-based algorithm where:
    - Each node is pushed onto the stack with its level
    - Before pushing, all nodes on the stack with level >= current are popped
      (they can't be parents of this node)
    - If the stack is empty after popping, this is a root node
    - Otherwise, the top of the stack is this node's parent

    Example: Given levels [1, 2, 3, 2, 1], the tree structure would be:
        Node(level=1)
        ├── Node(level=2)
        │   └── Node(level=3)
        └── Node(level=2)
        Node(level=1)

    Each node in the output tree has:
        - title: Section heading text
        - node_id: Temporary sequential ID (will be overwritten by write_node_id)
        - text: Full text content (including descendants' text — see
          extract_node_text_content)
        - line_num: Source line number for citation
        - nodes: List of child nodes (empty list for leaves)

    Args:
        node_list: Flat list of node dicts with "title", "text", "line_num",
            "level" keys (from extract_node_text_content).

    Returns:
        List of root-level tree node dicts with nested "nodes" children.
    """
    if not node_list:
        return []

    stack = []       # Stack of (tree_node_dict, level) tuples
    root_nodes = []  # Nodes with no parent
    node_counter = 1

    for node in node_list:
        current_level = node['level']

        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),  # Temporary ID: "0001", "0002", ...
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []  # Children will be appended here
        }
        node_counter += 1

        # Pop all stack entries with level >= current.
        # A level-2 node can't be a parent of another level-2 node,
        # so we pop until we find a strictly shallower level.
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            # No parent on the stack — this is a root node
            root_nodes.append(tree_node)
        else:
            # Top of stack is the parent — append as child
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)

        # Push this node so it can be a parent for deeper nodes
        stack.append((tree_node, current_level))

    return root_nodes


# =============================================================================
# Step 4: Large-Section Splitting (LLM-powered)
#
# For leaf nodes that are too large (e.g. a section with 3000 tokens and no
# sub-headers), the LLM is asked to infer hierarchical structure. This mirrors
# the PDF pipeline's approach:
# - Tags text with <line_X> markers (like PDF's <physical_index_X>)
# - Asks LLM for hierarchical structure indices ("1", "1.1", "1.2")
# - Builds a multi-level sub-tree from the LLM's response
# =============================================================================

async def _infer_section_structure(node, model=None):
    """Ask the LLM to identify hierarchical sub-sections within a large text block.

    This is the core LLM call for the markdown pipeline's section splitting.
    The approach mirrors the PDF pipeline's generate_toc_init():

    1. Tags each line with `<line_X>content<line_X>` markers so the LLM can
       reference specific line positions in its response.
    2. Sends a prompt asking the LLM to produce a flat list of sections with
       hierarchical structure indices (e.g. "1", "1.1", "1.2", "2") and
       start_line numbers.
    3. Validates the response: must be a list of 2+ dicts with title and
       start_line, sorted by start_line, with the first entry at line 1.

    The LLM acts as a gatekeeper: if the text doesn't have meaningful internal
    structure, it returns fewer than 2 items and the node is left unsplit.
    This means false-positive triggers (from the token/paragraph thresholds)
    only cost one wasted LLM call.

    Args:
        node: Tree node dict with "text" and "title" keys.
        model: LLM model identifier.

    Returns:
        List of dicts [{"structure": "1", "title": "...", "start_line": int}, ...]
        with at least 2 items, or empty list if no structure was found.
    """
    text = node.get("text", "")
    title = node.get("title", "Untitled")
    lines = text.split("\n")
    logger.info(f"[_infer_section_structure] '{title}': {len(lines)} lines, asking LLM for structure")

    # Tag each line with markers so the LLM can reference positions.
    # Format: <line_1>actual content here<line_1>
    tagged_lines = []
    for i, line in enumerate(lines):
        tagged_lines.append(f"<line_{i+1}>{line}<line_{i+1}>")
    tagged_text = "\n".join(tagged_lines)

    prompt = f"""You are an expert in extracting hierarchical tree structure. Your task is to generate the tree structure for the document section titled "{title}".

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, generate a concise, descriptive title based on the content of each section.

The provided text contains tags like <line_X> to indicate line numbers.

For the start_line, identify the line number where each section begins using the <line_X> tags.

Rules:
- The first entry MUST have start_line 1
- Generate 2-8 top-level sections with subsections where content clearly has distinct sub-topics
- Add subsections where there are clear section separators or topic boundaries
- Avoid grouping multiple subsections together that should have been split
- Be particularly sharp about where a section or subsection should start and end; apply the following sense checks when determining the range of a section or subsection
  - Section start_lines should be monotonically increasing: the next section's start line should be >= the start line of all previous sections
  - Ranges should never overlap between sections or subsections (they can start on the same page, but later sections should never start before previous sections)
  - The overall range covered by child sections of a root section should not extend beyond the start_line of the next root section (see the previous overlap rule)
- Prefer splitting at paragraph boundaries
- Titles should be concise and descriptive

The response should be in the following format:
[
    {{"structure": "1", "title": "...", "start_line": 1}},
    {{"structure": "1.1", "title": "...", "start_line": 15}},
    {{"structure": "1.2", "title": "...", "start_line": 30}},
    {{"structure": "2", "title": "...", "start_line": 45}}
]

Directly return the final JSON structure. Do not output anything else.

Text:
{tagged_text}"""

    result = await _llm_json(model, prompt)

    # --- Validation ---
    # Must be a list with at least 2 sections (splitting into 1 is pointless)
    if not isinstance(result, list) or len(result) < 2:
        return []

    valid_items = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if "title" not in item or "start_line" not in item:
            continue
        try:
            item["start_line"] = int(str(item["start_line"]).strip())
        except (ValueError, TypeError):
            continue
        # Assign a default structure index if the LLM omitted it
        if "structure" not in item:
            item["structure"] = str(len(valid_items) + 1)
        valid_items.append(item)

    if len(valid_items) < 2:
        logger.info(f"[_infer_section_structure] '{title}': LLM returned {len(valid_items)} valid sections (not enough to split)")
        return []

    logger.info(f"[_infer_section_structure] '{title}': LLM returned {len(valid_items)} valid sections")
    # Sort by position and force the first entry to start at line 1
    # (the LLM sometimes starts at a later line, losing the beginning)
    valid_items.sort(key=lambda x: x["start_line"])
    if valid_items[0]["start_line"] != 1:
        valid_items[0]["start_line"] = 1

    return valid_items


def _structure_list_to_tree(structure_list, text_lines, base_line_num):
    """Convert a flat structure-indexed list into a nested tree with text slices.

    This is the line-based equivalent of the PDF pipeline's list_to_tree() +
    add_node_text(). It takes the LLM's flat output (structure indices like
    "1", "1.1", "1.2", "2") and builds a proper nested tree, assigning text
    to each node by slicing the original lines.

    Structure indices encode hierarchy: "1.2.3" means section 3 under
    subsection 2 under section 1. Parent-child relationships are determined
    by stripping the last component (e.g. parent of "1.2.3" is "1.2").

    After building the tree, parent nodes have their text trimmed to only
    the "intro" portion before the first child starts. This prevents the
    same text appearing in both parent and child nodes.

    Args:
        structure_list: List of dicts with "structure", "title", "start_line"
            from _infer_section_structure(). Must be sorted by start_line.
        text_lines: The original text split into lines (from the parent node).
        base_line_num: The line number offset of the parent node in the
            original document. Added to local line indices to produce
            absolute line_num values for citation.

    Returns:
        List of root-level tree node dicts with nested "nodes" children.
        Each node has: title, text, line_num, nodes.
    """
    total_lines = len(text_lines)

    # Compute end_line for each item: text runs from start_line to end_line.
    # Last item extends to the end of the text block.
    for i, item in enumerate(structure_list):
        if i + 1 < len(structure_list):
            item["end_line"] = structure_list[i + 1]["start_line"] - 1
        else:
            item["end_line"] = total_lines

    def get_parent_structure(structure):
        """Get the parent's structure index by removing the last component.
        e.g. "1.2.3" -> "1.2", "1" -> None (root level)."""
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    # Build tree from structure indices (mirrors utils.list_to_tree)
    nodes = {}       # structure_index -> node dict
    root_nodes = []  # Nodes without a parent in the structure

    for item in structure_list:
        structure = item["structure"]
        start = item["start_line"] - 1  # Convert to 0-based for slicing
        end = item["end_line"]          # end_line is inclusive in line count

        node = {
            "title": item["title"],
            "text": "\n".join(text_lines[start:end]).strip(),
            "line_num": base_line_num + start,  # Absolute line number
            "nodes": [],
        }

        nodes[structure] = node

        # Find this node's parent by stripping the last index component
        parent_structure = get_parent_structure(structure)
        if parent_structure and parent_structure in nodes:
            nodes[parent_structure]["nodes"].append(node)
        else:
            # No parent found — this is a root-level section
            root_nodes.append(node)

    # Trim parent text to avoid duplication with children.
    # A parent's text initially spans from its start to its end (which
    # includes all children's text). We trim it to only the lines before
    # the first child starts — the "intro" portion.
    for item in structure_list:
        structure = item["structure"]
        node = nodes[structure]
        if node["nodes"]:
            node_start = item["start_line"] - 1
            first_child_start = node["nodes"][0]["line_num"] - base_line_num
            if first_child_start > node_start:
                # Parent text = only the intro lines before first child
                node["text"] = "\n".join(
                    text_lines[node_start:first_child_start]
                ).strip()
            else:
                # First child starts at the same line — parent gets no text
                node["text"] = ""

    return root_nodes


def _apply_inferred_structure(node, structure_list):
    """Graft LLM-inferred sub-sections onto a leaf node, making it a parent.

    Takes the validated structure_list from _infer_section_structure() and
    converts it into a nested sub-tree via _structure_list_to_tree(), then
    attaches the sub-tree as children of the original node.

    After this, the node is no longer a leaf — it has children, and its own
    text is reduced to just the first line (typically the header).

    Args:
        node: The leaf node dict to be split. Mutated in-place.
        structure_list: LLM-inferred structure from _infer_section_structure().
    """
    text = node.get("text", "")
    lines = text.split("\n")
    base_line_num = node.get("line_num", 1)

    # Build the sub-tree from the structure indices
    children = _structure_list_to_tree(structure_list, lines, base_line_num)

    if children:
        node["nodes"] = children
        # The parent node's text becomes just the header/first line.
        # All substantive content is now in the children.
        node["text"] = lines[0] if lines else ""


async def split_large_sections(tree_nodes, model=None, max_node_tokens=PAGEINDEX_MAX_NODE_TOKENS, min_split_tokens=PAGEINDEX_MIN_SPLIT_TOKENS, min_paragraph_count=PAGEINDEX_MIN_PARAGRAPH_COUNT):
    """Recursively walk the tree and split oversized leaf nodes via LLM inference.

    This is the main orchestrator for section splitting. It checks each leaf
    node against two triggering criteria:

    **Tier 1 — Always split** (max_node_tokens, default 1500):
        Any leaf node with more than this many tokens is always sent to the
        LLM for structure inference. This catches obviously large sections.

    **Tier 2 — Structural signal** (min_split_tokens + min_paragraph_count):
        Leaf nodes between min_split_tokens (500) and max_node_tokens (1500)
        that also have >= min_paragraph_count (4) blank-line-separated
        paragraphs. This catches smaller nodes that show signs of internal
        structure (multiple paragraphs = likely multiple topics).

    The LLM is the final gatekeeper: _infer_section_structure() returns []
    if it can't find meaningful structure, so false positives from the
    heuristic triggers only waste one LLM call per node.

    After splitting a node, the function recurses into the newly created
    children, so multi-level splitting is possible (e.g. a 5000-token leaf
    gets split into 3 sections, one of which is still 2000 tokens and gets
    split again).

    Args:
        tree_nodes: List of tree node dicts. Mutated in-place.
        model: LLM model identifier for structure inference.
        max_node_tokens: Token threshold above which a leaf always gets split.
        min_split_tokens: Lower token threshold for structural-signal splitting.
        min_paragraph_count: Minimum paragraph blocks to trigger tier-2 split.
    """
    logger.info(f"[split_large_sections] Processing {len(tree_nodes)} nodes")
    for node in tree_nodes:
        text = node.get("text", "")
        token_count = count_tokens(text, model=model)

        # Only split leaf nodes (nodes with children are already structured)
        if not node.get("nodes"):
            # Tier 1: always try if above the high threshold
            should_split = token_count > max_node_tokens

            # Tier 2: try if above low threshold AND has multiple paragraphs
            if not should_split and token_count > min_split_tokens:
                blocks = [b for b in text.split('\n\n') if b.strip()]
                should_split = len(blocks) >= min_paragraph_count

            if should_split:
                structure_list = await _infer_section_structure(node, model=model)
                if structure_list and len(structure_list) >= 2:
                    logger.info(f"[split_large_sections] Splitting '{node.get('title', '')}' ({token_count} tokens): "
                                f"LLM returned {len(structure_list)} sub-sections")
                    _apply_inferred_structure(node, structure_list)
                else:
                    logger.info(f"[split_large_sections] No split for '{node.get('title', '')}' ({token_count} tokens): "
                                f"LLM found no meaningful sub-structure")

        # Recurse into children — including any newly created ones from
        # the splitting above. This enables multi-level splitting.
        if node.get("nodes"):
            await split_large_sections(
                node["nodes"], model=model, max_node_tokens=max_node_tokens,
                min_split_tokens=min_split_tokens, min_paragraph_count=min_paragraph_count,
            )


# =============================================================================
# Utility: Clean tree for output (strips internal fields)
# =============================================================================

def clean_tree_for_output(tree_nodes):
    """Create a clean copy of the tree with only output-relevant fields.

    Strips any internal/temporary fields (like "level", "text_token_count")
    and keeps only: title, node_id, text, line_num, nodes.

    Used for standalone output (e.g. the __main__ block) but NOT used in the
    Agentic pipeline — the outer page_index.py uses format_structure() from
    utils instead, which is more configurable.

    Args:
        tree_nodes: List of tree node dicts.

    Returns:
        New list of cleaned node dicts (deep copy of relevant fields).
    """
    cleaned_nodes = []

    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num']
        }

        if node['nodes']:
            cleaned_node['nodes'] = clean_tree_for_output(node['nodes'])

        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes


# =============================================================================
# Main Entry Point: Markdown Pipeline
# =============================================================================

async def md_to_tree(md_path=None, if_thinning=False, min_token_threshold=PAGEINDEX_MIN_TOKEN_THRESHOLD, if_add_node_summary='no', summary_token_threshold=PAGEINDEX_SUMMARY_TOKEN_THRESHOLD, model=None, if_add_doc_description='no', if_add_node_text='no', if_add_node_id='yes', if_split_large_sections=True, max_node_tokens=PAGEINDEX_MAX_NODE_TOKENS, min_split_tokens=PAGEINDEX_MIN_SPLIT_TOKENS, min_paragraph_count=PAGEINDEX_MIN_PARAGRAPH_COUNT, md_content=None, doc_name=None, llm_max_concurrent=None):
    """Build a hierarchical tree from a markdown file or string.

    This is the main entry point for the markdown pipeline. It orchestrates
    the full sequence:

        1. extract_nodes_from_markdown() — Regex scan for # headers
        2. extract_node_text_content() — Assign text slices between headers
        3. (optional) thinning — Merge small subtrees
        4. build_tree_from_nodes() — Stack-based nesting by header level
        5. split_large_sections() — LLM-split oversized leaf nodes
        6. write_node_id() — Assign sequential IDs ("0001", "0002", ...)
        7. format_structure() — Reorder/filter node fields for output
        8. generate_summaries_for_structure_md() — Concurrent LLM summaries

    If the markdown has NO headers at all, the entire document is wrapped as
    a single root node so that split_large_sections() can still break it into
    meaningful sections via LLM inference.

    Args:
        md_path: Path to the markdown file to index. Either md_path or
            md_content must be provided.
        if_thinning: If True, merge subtrees below min_token_threshold tokens.
            This reduces granularity for documents with many tiny sections.
        min_token_threshold: Token threshold for thinning (only if if_thinning).
        if_add_node_summary: "yes" to generate LLM summaries per node.
        summary_token_threshold: Nodes with fewer tokens than this use their
            raw text as the summary (no LLM call). Default: 200.
        model: LLM model identifier for splitting and summary generation.
        if_add_doc_description: "yes" to generate an overall document description.
        if_add_node_text: "yes" to include section text in the output tree.
            The outer page_index.py always passes "yes" because it needs text
            for SectionData objects.
        if_add_node_id: "yes" to assign sequential node IDs.
        if_split_large_sections: Whether to LLM-split oversized leaf nodes.
        max_node_tokens: Token threshold above which a leaf always gets split.
        min_split_tokens: Lower token threshold for structural-signal splitting.
        min_paragraph_count: Minimum paragraph blocks for tier-2 splitting.
        md_content: Markdown content string. If provided, used directly instead
            of reading from md_path.
        doc_name: Document name override. If not provided, derived from md_path
            filename or defaults to "Untitled".

    Returns:
        Dict with:
        - "doc_name": Filename without extension (or provided doc_name)
        - "structure": List of root tree nodes with title, node_id, text,
          line_num, summary/prefix_summary, and nested nodes.
        - "doc_description": (optional) LLM-generated document description.
    """
    if md_content is not None:
        markdown_content = md_content
        if doc_name is None:
            doc_name = "Untitled"
    elif md_path is not None:
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        if doc_name is None:
            doc_name = os.path.splitext(os.path.basename(md_path))[0]
    else:
        raise ValueError("Either md_path or md_content must be provided")

    if llm_max_concurrent:
        init_llm_semaphore(llm_max_concurrent)

    # --- Step 1: Extract headers ---
    logger.info("Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    if node_list:
        # --- Step 2: Assign text slices ---
        logger.info("Extracting text content from nodes...")
        nodes_with_content = extract_node_text_content(node_list, markdown_lines)

        # --- Optional: Thinning ---
        # Merge subtrees that are too small to be useful as separate sections.
        # Requires two passes: first compute cumulative token counts, then merge.
        if if_thinning:
            nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, model=model)
            logger.info("Thinning nodes...")
            nodes_with_content = tree_thinning_for_index(nodes_with_content, min_token_threshold, model=model)

        # --- Step 3: Build nested tree ---
        logger.info("Building tree from nodes...")
        tree_structure = build_tree_from_nodes(nodes_with_content)
    else:
        # FALLBACK: No markdown headers found.
        # Wrap the entire document as a single root node so the LLM splitting
        # step can still break it into logical sections. Without this, a
        # plain-text document would produce no tree at all.
        logger.info("No markdown headers found, creating root node for splitting...")
        tree_structure = [{
            'title': doc_name,
            'text': markdown_content.strip(),
            'line_num': 1,
            'nodes': [],
        }]

    # --- Step 4: Split large leaf nodes ---
    # Walk the tree and use LLM to infer sub-sections for oversized leaves.
    # This is especially important for the no-header fallback (single root node)
    # where the entire document text is in one leaf.
    if if_split_large_sections:
        logger.info(f"Splitting large sections (threshold: {max_node_tokens} tokens)...")
        await split_large_sections(tree_structure, model=model, max_node_tokens=max_node_tokens, min_split_tokens=min_split_tokens, min_paragraph_count=min_paragraph_count)

    # --- Step 5: Assign final node IDs ---
    # write_node_id() (from utils.py) walks the tree in pre-order and assigns
    # sequential zero-padded IDs: "0001", "0002", "0003", etc.
    # This overwrites the temporary IDs from build_tree_from_nodes().
    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    # --- Steps 6-7: Format and generate summaries ---
    logger.info("Formatting tree structure...")

    if if_add_node_summary == 'yes':
        # Include text in the formatted structure because summaries are
        # generated FROM the text. Text may be stripped afterwards.
        tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes'])

        # Generate summaries for every node concurrently.
        # Leaf nodes get "summary", parent nodes get "prefix_summary".
        logger.info("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(tree_structure, summary_token_threshold=summary_token_threshold, model=model)

        if if_add_node_text == 'no':
            # Strip text from the output if not requested.
            # (The outer page_index.py always requests text, so this branch
            # is only hit when md_to_tree is called standalone.)
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes'])

        if if_add_doc_description == 'yes':
            logger.info("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = await generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': doc_name,
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        # No summaries — just format the tree with or without text
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes'])
        else:
            tree_structure = format_structure(tree_structure, order = ['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes'])

    return {
        'doc_name': doc_name,
        'structure': tree_structure,
    }


# =============================================================================
# Page-Aware Pipeline Helpers (Agentic-specific additions)
#
# These functions are NOT in the reference PageIndex library. They add
# retrieval-optimized post-processing after the reference pipeline builds
# the tree structure.
# =============================================================================

def _trim_parent_text(tree_nodes, page_list):
    """Trim parent-node text to only the intro portion before the first child.

    **Problem**: add_node_text() (from the reference library) assigns each node
    the full concatenated text of its start_index..end_index page range. For
    parent nodes, this page range encompasses all children's pages too, so
    the parent's text is a superset of its children's text — exact duplication.

    **Solution**: Walk the tree and for each parent node, re-assign its text
    to only the pages from parent.start_index to (first_child.start_index - 1).
    This is the "intro" or "prefix" portion — any text on pages before the
    first child begins. If the first child starts on the same page as the
    parent, the parent gets empty text.

    This is critical for retrieval quality: without trimming, fetching a parent
    section would return all children's text duplicated, wasting context window.

    Args:
        tree_nodes: List of tree node dicts with "start_index", "end_index",
            "text", and "nodes" keys. Mutated in-place.
        page_list: List of (page_text, token_count) tuples from the PDF
            extractor. Used to re-extract text for the trimmed page range.
    """
    for node in tree_nodes:
        children = node.get("nodes")
        if not children:
            continue  # Leaf nodes — text is already correct

        first_child_start = children[0].get("start_index")
        node_start = node.get("start_index")

        if first_child_start is not None and node_start is not None:
            if first_child_start > node_start:
                # Parent has intro pages before the first child.
                # Re-extract text for just those prefix pages.
                node["text"] = get_text_of_pdf_pages(
                    page_list, node_start, first_child_start - 1
                )
                node["end_index"] = first_child_start - 1
            else:
                # First child starts on the same page as parent.
                # No intro text — parent gets empty string.
                node["text"] = ""
                node["end_index"] = node_start

        # Recurse into children (they may also be parents)
        _trim_parent_text(children, page_list)


def _merge_small_siblings(tree_nodes, model=None, min_merge_tokens=PAGEINDEX_MIN_MERGE_TOKENS):
    """Merge tiny leaf nodes into an adjacent sibling or parent.

    **Problem**: The tree may contain very small leaf sections (e.g. a one-line
    "Acknowledgments" section with 20 tokens). These are too small to be useful
    as standalone retrieval units and add noise to the ToC.

    **Solution**: Walk the tree and for each leaf node whose text is below
    min_merge_tokens (default 200):

    1. **Merge backward** (preferred): If the previous sibling exists, append
       this node's text to it and extend its end_index. The small node is removed.

    2. **Merge forward**: If no previous sibling but a next sibling exists,
       prepend this node's text to the next sibling. The small node is removed
       and the next sibling inherits its title and start_index.

    3. **Absorb into parent**: If this is the only child, merge its text into
       the parent. The child is removed, and the parent may become a leaf.

    After processing all children, if the children list is empty (all were
    absorbed), the "nodes" key is deleted from the parent, making it a leaf.
    Otherwise, recurse into remaining children.

    Only leaf nodes are candidates for merging — nodes with their own children
    are skipped (they have structural significance regardless of text size).

    Args:
        tree_nodes: List of tree node dicts. Mutated in-place.
        model: Model identifier for token counting.
        min_merge_tokens: Minimum tokens for a leaf to survive as standalone.
    """
    for node in tree_nodes:
        children = node.get("nodes")
        if not children:
            continue

        i = 0
        while i < len(children):
            child = children[i]

            # Skip non-leaf nodes — they have structural significance
            if child.get("nodes"):
                i += 1
                continue

            text = child.get("text", "")
            # Skip if the leaf is large enough to be a standalone section
            if count_tokens(text, model=model) >= min_merge_tokens:
                i += 1
                continue

            # --- This leaf is too small — merge it ---

            if i > 0:
                # MERGE BACKWARD: append to previous sibling
                prev = children[i - 1]
                prev["text"] = prev.get("text", "") + "\n\n" + text
                # Extend previous sibling's page range to cover this node
                if child.get("end_index"):
                    prev["end_index"] = child["end_index"]
                children.pop(i)
                # Don't increment i — next child shifted into this position

            elif len(children) > 1:
                # MERGE FORWARD: prepend to next sibling
                nxt = children[i + 1]
                nxt["text"] = text + "\n\n" + nxt.get("text", "")
                # Next sibling inherits this node's title and start_index
                nxt["title"] = child["title"]
                if child.get("start_index"):
                    nxt["start_index"] = child["start_index"]
                children.pop(i)
                # Don't increment i — next child shifted into this position

            else:
                # ABSORB INTO PARENT: only child, merge into parent
                node["text"] = node.get("text", "") + "\n\n" + text
                if child.get("end_index"):
                    node["end_index"] = child["end_index"]
                children.pop(i)
                # Don't increment i — list is now empty

        if not children:
            # All children were absorbed — parent becomes a leaf node
            del node["nodes"]
        else:
            # Recurse into remaining children (they may also have small leaves)
            _merge_small_siblings(children, model=model, min_merge_tokens=min_merge_tokens)


def _count_tree_nodes(nodes):
    return sum(1 + _count_tree_nodes(n.get('nodes', [])) for n in nodes)


# =============================================================================
# Main Entry Point: Page-Aware Pipeline
# =============================================================================

async def md_to_tree_from_pages(
    page_texts,
    doc_name,
    model=None,
    if_add_node_summary="no",
    summary_token_threshold=None,
    if_add_node_text="no",
    if_add_node_id="yes",
    if_split_large_sections=True,
    max_node_tokens=PAGEINDEX_MAX_NODE_TOKENS,
    min_split_tokens=PAGEINDEX_MIN_SPLIT_TOKENS,
    min_paragraph_count=PAGEINDEX_MIN_PARAGRAPH_COUNT,
    # Parameters for the reference PageIndex tree_parser:
    toc_check_page_num=PAGEINDEX_TOC_CHECK_PAGE_NUM,
    toc_offset_scan_pages=PAGEINDEX_TOC_OFFSET_SCAN_PAGES,
    max_page_num_each_node=PAGEINDEX_MAX_PAGE_NUM_EACH_NODE,
    max_token_num_each_node=PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE,
    toc_gap_tolerance=PAGEINDEX_TOC_GAP_TOLERANCE,
    llm_max_concurrent=None,
):
    """Build a hierarchical tree from per-page text using the full ToC pipeline.

    This is the entry point for the page-aware pipeline, used for PDF-sourced
    documents where per-page text arrays are available from extractors (Fitz,
    Mistral OCR, pdfplumber).

    Instead of parsing markdown headers, this pipeline delegates to tree_parser()
    from the reference PageIndex library (page_index.py), which performs:

        1. **ToC detection**: Scans the first N pages for a table of contents
        2. **ToC verification**: Checks that detected ToC entries actually
           appear in the document at their stated page numbers
        3. **ToC correction**: Uses LLM to fix incorrect page numbers
        4. **Fallback**: If no ToC is found, uses LLM to infer structure
           directly from the page content (process_no_toc mode)
        5. **Large-node splitting**: Recursively breaks oversized nodes
           (> max_page_num_each_node pages or > max_token_num_each_node tokens)

    After tree_parser returns the structure (with start_index/end_index page
    ranges but no text), this function adds three Agentic-specific steps:

        5. **add_node_text()** — Assigns each node the concatenated text of
           its start_index..end_index page range.
        6. **_trim_parent_text()** — Trims parent nodes to only the intro
           pages before their first child (prevents text duplication).
        7. **_merge_small_siblings()** — Merges tiny leaf nodes into adjacent
           siblings (reduces noise in the ToC).

    Then optionally generates summaries and formats the output.

    Args:
        page_texts: List of per-page text strings from PDF extractors.
            page_texts[0] is page 1's text, page_texts[1] is page 2's, etc.
        doc_name: Human-readable document name.
        model: LLM model identifier for structure detection and summaries.
        if_add_node_summary: "yes" to generate per-node summaries.
        summary_token_threshold: Below this, text IS the summary (no LLM call).
        if_add_node_text: "yes" to include text in the output tree.
        if_add_node_id: "yes" to assign sequential node IDs.
        if_split_large_sections: Passed through but not directly used here
            (tree_parser handles splitting internally).
        max_node_tokens: Not directly used (tree_parser uses token thresholds).
        toc_check_page_num: Number of pages to scan for ToC (default 20).
        max_page_num_each_node: Max pages per node before recursive splitting.
        max_token_num_each_node: Max tokens per node before recursive splitting.

    Returns:
        Dict with "doc_name", "structure", and optionally "doc_description".
        Same format as md_to_tree().
    """
    import logging
    from types import SimpleNamespace
    from .page_index import tree_parser

    _logger = logging.getLogger(__name__)

    # --- Step 1: Build page_list ---
    # tree_parser expects a list of (text, token_count) tuples.
    # Token counts are pre-computed here to avoid redundant counting later.
    page_list = [(text, count_tokens(text, model=model)) for text in page_texts]

    # --- Step 2: Build config object ---
    # tree_parser expects an "opt" namespace with configuration attributes.
    # We wrap our parameters in a SimpleNamespace for compatibility.
    opt = SimpleNamespace(
        model=model or DEFAULT_MODEL,
        toc_check_page_num=toc_check_page_num,
        toc_offset_scan_pages=toc_offset_scan_pages,
        toc_gap_tolerance=toc_gap_tolerance,
        max_page_num_each_node=max_page_num_each_node,
        max_token_num_each_node=max_token_num_each_node,
        if_add_node_id=if_add_node_id,
        if_add_node_text=if_add_node_text,
        if_add_node_summary=if_add_node_summary,
        if_add_doc_description="no",
    )

    if llm_max_concurrent:
        init_llm_semaphore(llm_max_concurrent)

    _logger.info(
        f"Page-aware pipeline: {len(page_texts)} pages, model={opt.model}, "
        f"toc_check_page_num={toc_check_page_num}, toc_offset_scan_pages={toc_offset_scan_pages}"
    )

    # --- Step 3: Run the reference PageIndex pipeline ---
    # tree_parser() returns a tree structure where each node has:
    # title, start_index, end_index, and nested nodes — but NO text yet.
    # The start_index/end_index are 1-based page numbers.
    tree_structure = await tree_parser(page_list, opt)
    _logger.info(f"[md_to_tree_from_pages] tree_parser complete: {_count_tree_nodes(tree_structure)} nodes, {len(tree_structure)} roots")

    # --- Step 4: Assign node IDs ---
    # write_node_id() walks in pre-order and assigns "0001", "0002", etc.
    if if_add_node_id == "yes":
        write_node_id(tree_structure)
    _logger.info(f"[md_to_tree_from_pages] Node IDs assigned")

    # --- Step 5: Add text from page ranges ---
    # add_node_text() (from utils) concatenates page_list[start-1..end-1]
    # text for each node. After this, every node has a "text" field.
    add_node_text(tree_structure, page_list)
    _logger.info(f"[md_to_tree_from_pages] Text assigned to all nodes from page ranges")

    # --- Step 6: Agentic post-processing ---
    # _trim_parent_text: Prevents parent/child text duplication by trimming
    # parent text to only the intro pages before the first child.
    _trim_parent_text(tree_structure, page_list)
    _logger.info(f"[md_to_tree_from_pages] Parent text trimmed")

    # _merge_small_siblings: Absorbs tiny leaf nodes (<200 tokens) into
    # adjacent siblings to reduce ToC noise.
    _merge_small_siblings(tree_structure, model=model)
    _logger.info(f"[md_to_tree_from_pages] Small siblings merged. Tree now: {_count_tree_nodes(tree_structure)} nodes")

    # --- Step 7: Generate summaries and format output ---
    if if_add_node_summary == "yes":
        # Format with text included (summaries are generated from text)
        tree_structure = format_structure(
            tree_structure,
            order=[
                "title", "node_id", "summary", "prefix_summary",
                "text", "start_index", "end_index", "line_num", "nodes",
            ],
        )
        _logger.info("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=model,
        )
        _logger.info(f"[md_to_tree_from_pages] Summaries generated")
        if if_add_node_text == "no":
            # Strip text after summary generation if not requested
            tree_structure = format_structure(
                tree_structure,
                order=[
                    "title", "node_id", "summary", "prefix_summary",
                    "start_index", "end_index", "line_num", "nodes",
                ],
            )
        # Generate overall document description from the tree structure
        clean_structure = create_clean_structure_for_description(tree_structure)
        doc_description = await generate_doc_description(clean_structure, model=model)
        return {
            "doc_name": doc_name,
            "doc_description": doc_description,
            "structure": tree_structure,
        }
    else:
        # No summaries — format with or without text
        order_fields = [
            "title", "node_id", "summary", "prefix_summary",
            "text" if if_add_node_text == "yes" else None,
            "start_index", "end_index", "line_num", "nodes",
        ]
        order_fields = [f for f in order_fields if f]
        tree_structure = format_structure(tree_structure, order=order_fields)

    return {
        "doc_name": doc_name,
        "structure": tree_structure,
    }


# =============================================================================
# Standalone execution for testing
# =============================================================================

if __name__ == "__main__":
    import os
    import json

    # MD_NAME = 'Detect-Order-Construct'
    MD_NAME = 'cognitive-load'
    MD_PATH = os.path.join(os.path.dirname(__file__), '..', 'tests/markdowns/', f'{MD_NAME}.md')


    MODEL="gpt-4.1"
    IF_THINNING=False
    THINNING_THRESHOLD=5000
    SUMMARY_TOKEN_THRESHOLD=200
    IF_SUMMARY=True

    tree_structure = asyncio.run(md_to_tree(
        md_path=MD_PATH,
        if_thinning=IF_THINNING,
        min_token_threshold=THINNING_THRESHOLD,
        if_add_node_summary='yes' if IF_SUMMARY else 'no',
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD,
        model=MODEL))

    print('\n' + '='*60)
    print('TREE STRUCTURE')
    print('='*60)
    print_json(tree_structure)

    print('\n' + '='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(tree_structure['structure'])

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'{MD_NAME}_structure.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)

    print(f"\nTree structure saved to: {output_path}")
