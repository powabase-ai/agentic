"""
Source-diversity post-processing for retrieval results.

After an algorithm has scored/reranked a candidate pool, ``apply_source_limits``
re-selects the final ``top_k`` while enforcing per-source (per-document) limits:

- ``max_per_source``: a hard cap — at most N items from any one source.
- ``min_per_source``: a diversity floor — visit sources in best-score order and
  give each up to N of its top items before any source receives extra, then fill
  the remaining slots greedily by score.

Both limits are opt-in: ``None`` or ``0`` disables them. With neither set this is
a plain ``items[:top_k]`` truncation, so existing behaviour is preserved.

Input order does not matter: the function sorts a copy by score descending before
grouping/selecting (callers may pass a merged pool that isn't globally sorted). It
is a pure transform — it never mutates the input items or the input list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.knowledge.models import RetrievedItem


def _norm_limit(value: int | None) -> int | None:
    """Treat None/0/negative as 'disabled'."""
    if value is None:
        return None
    return value if value > 0 else None


def _source_key(item: RetrievedItem) -> str:
    """Group key for an item. Items without a source_id are their own singleton
    source (keyed by item_id) so the cap/floor never lumps them together."""
    if item.source_id is not None:
        return item.source_id
    return f"\x00item:{item.item_id}"


def _score(item: RetrievedItem) -> float:
    """Sort key tolerant of a missing/None score (treated as lowest)."""
    return item.score if item.score is not None else 0.0


def apply_source_limits(
    items: list[RetrievedItem],
    top_k: int,
    min_per_source: int | None = None,
    max_per_source: int | None = None,
) -> list[RetrievedItem]:
    """Select the final ``top_k`` items subject to per-source limits.

    Args:
        items: Candidate items (any order — sorted internally by score). For
            limits to have any effect this should be a pool larger than ``top_k``.
        top_k: Maximum number of items to return.
        min_per_source: Diversity floor — each matched source gets up to this many
            of its top items before any source exceeds it. None/0 disables.
        max_per_source: Hard cap on items per source. None/0 disables.

    Returns:
        Selected items sorted by score descending, length ``<= top_k``. When a
        ``max_per_source`` cap leaves fewer than ``top_k`` eligible items, the
        result is shorter — items are never fabricated.

    Notes:
        - If ``min_per_source > max_per_source`` the floor is clamped to the cap.
        - When ``min_per_source * num_sources > top_k`` the floor is best-effort:
          sources are visited in best-score order until ``top_k`` is exhausted.
    """
    min_per_source = _norm_limit(min_per_source)
    max_per_source = _norm_limit(max_per_source)

    if top_k <= 0:
        return []

    # Be robust about input order: callers may hand us a merged pool (e.g. a
    # diversity back-fill appended to the main results) that isn't globally
    # score-sorted. Sort a copy so grouping/selection sees score-desc order
    # without mutating the caller's list.
    items = sorted(items, key=_score, reverse=True)

    # Fast path: no limits → plain truncation (preserves legacy behaviour).
    if min_per_source is None and max_per_source is None:
        return items[:top_k]

    # A floor larger than the cap can never be satisfied; clamp it.
    if min_per_source is not None and max_per_source is not None:
        min_per_source = min(min_per_source, max_per_source)

    # Group items per source, preserving the incoming (score-desc) order within
    # each group. ``order`` is best-score-first because groups are created on
    # first appearance in the score-desc input.
    groups: dict[str, list[RetrievedItem]] = {}
    order: list[str] = []
    for item in items:
        key = _source_key(item)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    taken: dict[str, int] = {key: 0 for key in order}
    selected: list[RetrievedItem] = []
    selected_ids: set[str] = set()

    def _take(item: RetrievedItem, key: str) -> None:
        selected.append(item)
        selected_ids.add(item.item_id)
        taken[key] += 1

    # Phase A — diversity floor.
    if min_per_source is not None:
        for key in order:
            if len(selected) >= top_k:
                break
            for item in groups[key]:
                if len(selected) >= top_k or taken[key] >= min_per_source:
                    break
                _take(item, key)

    # Phase B — relevance fill. Walk the original score-desc order, adding any
    # not-yet-selected item whose source is below its cap, until top_k.
    for item in items:
        if len(selected) >= top_k:
            break
        if item.item_id in selected_ids:
            continue
        key = _source_key(item)
        if max_per_source is not None and taken[key] >= max_per_source:
            continue
        _take(item, key)

    # Restore global score-desc order for the returned slice.
    selected.sort(key=_score, reverse=True)
    return selected[:top_k]
