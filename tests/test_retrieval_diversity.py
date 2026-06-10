"""
Tests for per-source retrieval limits (agentic.knowledge.retrieval.diversity).
"""

from agentic.knowledge.models import RetrievedItem
from agentic.knowledge.retrieval import apply_source_limits


def _item(item_id: str, source_id: str | None, score: float) -> RetrievedItem:
    return RetrievedItem(
        item_id=item_id,
        text=f"text-{item_id}",
        score=score,
        source_id=source_id,
    )


def _pool() -> list[RetrievedItem]:
    """A score-desc candidate pool skewed toward source 'A'."""
    return [
        _item("a1", "A", 0.95),
        _item("a2", "A", 0.90),
        _item("a3", "A", 0.85),
        _item("b1", "B", 0.80),
        _item("a4", "A", 0.75),
        _item("c1", "C", 0.70),
        _item("b2", "B", 0.65),
        _item("c2", "C", 0.60),
    ]


def _sources(items: list[RetrievedItem]) -> list[str | None]:
    return [it.source_id for it in items]


def _ids(items: list[RetrievedItem]) -> list[str]:
    return [it.item_id for it in items]


class TestApplySourceLimits:
    def test_no_limits_is_plain_truncation(self):
        pool = _pool()
        result = apply_source_limits(pool, top_k=3)
        assert _ids(result) == ["a1", "a2", "a3"]

    def test_zero_limits_disabled(self):
        pool = _pool()
        result = apply_source_limits(pool, top_k=3, min_per_source=0, max_per_source=0)
        assert _ids(result) == ["a1", "a2", "a3"]

    def test_max_only_caps_each_source(self):
        pool = _pool()
        result = apply_source_limits(pool, top_k=5, max_per_source=2)
        # No source may appear more than twice; fill remaining by score.
        counts = {s: _sources(result).count(s) for s in set(_sources(result))}
        assert all(c <= 2 for c in counts.values())
        assert len(result) == 5
        # Highest two of A, then B/C fill in score order.
        assert _ids(result) == ["a1", "a2", "b1", "c1", "b2"]

    def test_max_too_small_returns_fewer(self):
        # Only sources A,B,C exist; cap of 1 => at most 3 results even though top_k=5.
        pool = _pool()
        result = apply_source_limits(pool, top_k=5, max_per_source=1)
        assert _ids(result) == ["a1", "b1", "c1"]

    def test_min_floor_guarantees_diversity(self):
        # Without limits, top_3 would be all A. min_per_source=1 forces B and C in.
        pool = _pool()
        result = apply_source_limits(pool, top_k=3, min_per_source=1)
        assert set(_sources(result)) == {"A", "B", "C"}
        # Each source represented by its best-scored chunk, returned score-desc.
        assert _ids(result) == ["a1", "b1", "c1"]

    def test_min_floor_then_relevance_fill(self):
        # top_k=5, min=1: cover A,B,C first (1 each), then fill 2 more by score.
        pool = _pool()
        result = apply_source_limits(pool, top_k=5, min_per_source=1)
        assert len(result) == 5
        assert {"A", "B", "C"}.issubset(set(_sources(result)))
        # Floor picks a1,b1,c1; fill picks next-best a2,a3.
        assert _ids(result) == ["a1", "a2", "a3", "b1", "c1"]

    def test_min_and_max_together(self):
        pool = _pool()
        result = apply_source_limits(pool, top_k=5, min_per_source=1, max_per_source=2)
        counts = {s: _sources(result).count(s) for s in set(_sources(result))}
        assert all(c <= 2 for c in counts.values())
        assert {"A", "B", "C"}.issubset(set(_sources(result)))
        assert len(result) == 5

    def test_min_clamped_to_max(self):
        # min(3) > max(1): floor clamps to 1, so each source gets exactly 1.
        pool = _pool()
        result = apply_source_limits(pool, top_k=5, min_per_source=3, max_per_source=1)
        assert _ids(result) == ["a1", "b1", "c1"]

    def test_min_times_sources_exceeds_top_k_is_best_effort(self):
        # min=2, top_k=3: can't give every source 2; cover best sources first.
        pool = _pool()
        result = apply_source_limits(pool, top_k=3, min_per_source=2)
        assert len(result) == 3
        # A gets its floor of 2 (best source), then 1 slot left -> next source B.
        assert _ids(result) == ["a1", "a2", "b1"]

    def test_items_without_source_are_singletons(self):
        pool = [
            _item("x1", None, 0.9),
            _item("x2", None, 0.8),
            _item("a1", "A", 0.7),
        ]
        # max_per_source=1 must NOT collapse the two source-less items together.
        result = apply_source_limits(pool, top_k=3, max_per_source=1)
        assert _ids(result) == ["x1", "x2", "a1"]

    def test_result_sorted_by_score_desc(self):
        pool = _pool()
        result = apply_source_limits(pool, top_k=6, min_per_source=1, max_per_source=3)
        scores = [it.score for it in result]
        assert scores == sorted(scores, reverse=True)

    def test_does_not_mutate_input(self):
        pool = _pool()
        original = _ids(pool)
        apply_source_limits(pool, top_k=3, min_per_source=1, max_per_source=2)
        assert _ids(pool) == original

    def test_top_k_zero_returns_empty(self):
        assert apply_source_limits(_pool(), top_k=0, max_per_source=2) == []
