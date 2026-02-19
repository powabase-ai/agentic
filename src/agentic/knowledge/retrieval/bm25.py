"""
BM25 scoring utilities for full-text search.

Pure functions for computing BM25 relevance scores from PostgreSQL tsvector data.
These are stateless and reusable by any retriever that needs keyword-based scoring.

BM25 improves over TF-IDF (PostgreSQL's ts_rank) with:
- Term frequency saturation: diminishing returns for repeated terms
- Document length normalization: shorter docs with same term density score higher
"""

import math
import re


def parse_tsvector(tsvector_str: str) -> dict[str, int]:
    """
    Parse a PostgreSQL tsvector text representation into term frequencies.

    PostgreSQL tsvectors look like: ``'cat':1,5 'dog':2 'run':3,4,7``
    Each term is followed by a colon and a comma-separated list of positions.
    The number of positions equals the term frequency in the document.

    Args:
        tsvector_str: The text representation of a tsvector
            (from ``to_tsvector('english', text)::text``)

    Returns:
        Dict mapping stemmed terms to their frequency in the document.

    Examples:
        >>> parse_tsvector("'cat':1,5 'dog':2")
        {'cat': 2, 'dog': 1}
        >>> parse_tsvector("'machin':3 'learn':1,4,7")
        {'machin': 1, 'learn': 3}
        >>> parse_tsvector("")
        {}
    """
    if not tsvector_str or not tsvector_str.strip():
        return {}

    term_freqs: dict[str, int] = {}
    # Match 'term':pos1,pos2,... patterns
    for match in re.finditer(r"'([^']+)':([0-9,]+)", tsvector_str):
        term = match.group(1)
        positions = match.group(2).split(",")
        term_freqs[term] = len(positions)

    return term_freqs


def bm25_score(
    query_terms: list[str],
    doc_term_freqs: dict[str, int],
    doc_len: int,
    avg_doc_len: float,
    total_docs: int,
    doc_freqs: dict[str, int],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    Compute the BM25 relevance score for a single document.

    Uses the standard BM25 formula::

        score = Σ IDF(qi) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |D| / avgdl))

    where::

        IDF(qi) = ln((N - df + 0.5) / (df + 0.5) + 1)

    Args:
        query_terms: List of stemmed query terms (from PostgreSQL's to_tsvector).
        doc_term_freqs: Term frequency map for the document
            (from :func:`parse_tsvector`).
        doc_len: Length of the document (character count of text).
        avg_doc_len: Average document length across the corpus.
        total_docs: Total number of documents in the corpus.
        doc_freqs: Dict mapping each query term to the number of documents
            containing that term.
        k1: Term frequency saturation parameter. Higher values give more weight
            to term frequency. Default 1.2 (standard BM25).
        b: Length normalization parameter (0 = no normalization, 1 = full).
            Default 0.75 (standard BM25).

    Returns:
        BM25 relevance score (non-negative float). Higher means more relevant.

    Examples:
        >>> bm25_score(
        ...     query_terms=["machin", "learn"],
        ...     doc_term_freqs={"machin": 3, "learn": 2},
        ...     doc_len=500,
        ...     avg_doc_len=1000.0,
        ...     total_docs=100,
        ...     doc_freqs={"machin": 10, "learn": 30},
        ... )  # doctest: +SKIP
        3.45...
    """
    if avg_doc_len <= 0:
        return 0.0

    score = 0.0
    for term in query_terms:
        tf = doc_term_freqs.get(term, 0)
        if tf == 0:
            continue

        df = doc_freqs.get(term, 0)
        # IDF with the "+1" variant to ensure non-negative values
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        # BM25 term frequency component with saturation and length normalization
        tf_component = (tf * (k1 + 1.0)) / (
            tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
        )

        score += idf * tf_component

    return score
