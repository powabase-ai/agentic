"""
Result fusion utilities for combining multiple retrieval signals.

Reciprocal Rank Fusion (RRF) is a rank-based method for merging results from
different retrieval systems. Unlike score-based fusion (weighted sum), RRF is
agnostic to score scales — making it robust when combining signals with
incompatible ranges (e.g., cosine similarity 0–1 vs BM25 scores 1–20).

Reference:
    Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and Individual Rank Learning Methods."
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievedItem


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedItem]],
    weights: list[float] | None = None,
    top_k: int = 10,
    k: int = 60,
) -> list[RetrievedItem]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    For each document ``d`` appearing in any result list::

        rrf_score(d) = Σ weight_i × 1 / (k + rank_i(d))

    where ``rank_i(d)`` is the 0-based rank of ``d`` in result list ``i``
    (documents absent from a list are simply not scored for that list).

    Args:
        result_lists: List of ranked result lists to fuse. Each inner list
            should be sorted by relevance (best first).
        weights: Optional per-list weights. If None, all lists are weighted
            equally (weight=1.0). Length must match ``result_lists``.
        top_k: Maximum number of results to return.
        k: RRF constant controlling how much lower-ranked results are
            discounted. Default 60 (per original paper recommendation).

    Returns:
        Fused list of RetrievedItem, sorted by RRF score descending,
        truncated to ``top_k``.

    Examples:
        >>> from agentic.knowledge.models import RetrievedItem
        >>> vector_results = [
        ...     RetrievedItem(item_id="a", text="...", score=0.95),
        ...     RetrievedItem(item_id="b", text="...", score=0.80),
        ... ]
        >>> keyword_results = [
        ...     RetrievedItem(item_id="b", text="...", score=12.5),
        ...     RetrievedItem(item_id="c", text="...", score=8.3),
        ... ]
        >>> fused = reciprocal_rank_fusion(
        ...     [vector_results, keyword_results],
        ...     weights=[0.7, 0.3],
        ...     top_k=5,
        ... )
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"result_lists length ({len(result_lists)})"
        )

    # Accumulate RRF scores. Keep the first-seen item for each ID
    # (preserves the item from the highest-weighted list).
    scores: dict[str, float] = {}
    items: dict[str, RetrievedItem] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, item in enumerate(result_list):
            rrf = weight * (1.0 / (k + rank))
            scores[item.item_id] = scores.get(item.item_id, 0.0) + rrf
            if item.item_id not in items:
                items[item.item_id] = item

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)

    # Normalize to [0, 1] so the top result has score 1.0.
    # This is purely cosmetic — ranking is unchanged.
    max_score = scores[sorted_ids[0]] if sorted_ids else 1.0
    if max_score > 0:
        for item_id in sorted_ids:
            scores[item_id] /= max_score

    return [
        replace(items[item_id], score=scores[item_id])
        for item_id in sorted_ids[:top_k]
    ]
