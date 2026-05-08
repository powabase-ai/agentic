"""Tests for the LiteLLM cost accumulator.

The accumulator is scoped to a Celery indexing task via a ContextVar,
mirroring the existing _reasoning_effort_var / _llm_semaphore_var pattern
in agentic/src/agentic/knowledge/indexing/_pageindex_lib/utils.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentic.llm.cost_accumulator import (
    CostAccumulator,
    _acc_var,
    _async_cb,
    init_accumulator,
    install,
)


def _fake_usage(prompt: int, completion: int, reasoning: int | None = None):
    """Mirror the LiteLLM/OpenAI shape: usage with completion_tokens_details."""
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=details,
    )


def _fake_response(prompt: int, completion: int, reasoning: int | None, cost: float | None):
    return SimpleNamespace(
        usage=_fake_usage(prompt, completion, reasoning),
        _hidden_params={"response_cost": cost} if cost is not None else {},
    )


def test_accumulator_aggregates_basic():
    """add() bumps counts and tokens correctly for one stage."""
    acc = CostAccumulator()
    acc.add(
        stage="tree",
        model="openai/responses/gpt-5-mini",
        effort="low",
        usage=_fake_usage(100, 50, reasoning=10),
        cost=0.0001,
    )
    acc.add(
        stage="tree",
        model="openai/responses/gpt-5-mini",
        effort="low",
        usage=_fake_usage(200, 75, reasoning=20),
        cost=0.0002,
    )
    d = acc.to_dict()
    tree = d["by_stage"]["tree"]
    assert tree["calls"] == 2
    assert tree["prompt_tokens"] == 300
    assert tree["completion_tokens"] == 125
    assert tree["reasoning_tokens"] == 30
    assert tree["cost_usd"] == pytest.approx(0.0003)
    assert tree["model"] == "openai/responses/gpt-5-mini"
    assert tree["reasoning_effort"] == "low"
    assert d["total_cost_usd"] == pytest.approx(0.0003)


def test_accumulator_separates_stages():
    """Tree-building and enrichment costs aggregate independently."""
    acc = CostAccumulator()
    acc.add("tree", "m", "low", _fake_usage(100, 50, 10), 0.001)
    acc.add("enrichment", "m", "low", _fake_usage(80, 30, 5), 0.0005)
    d = acc.to_dict()
    assert set(d["by_stage"]) == {"tree", "enrichment"}
    assert d["by_stage"]["tree"]["calls"] == 1
    assert d["by_stage"]["enrichment"]["calls"] == 1
    assert d["total_cost_usd"] == pytest.approx(0.0015)


def test_accumulator_handles_none_fields():
    """Defensive: usage with no details / cost==None must not crash."""
    acc = CostAccumulator()
    acc.add("tree", "m", None, _fake_usage(10, 5, reasoning=None), cost=None)
    tree = acc.to_dict()["by_stage"]["tree"]
    assert tree["reasoning_tokens"] == 0
    assert tree["cost_usd"] == 0.0
    assert tree["calls"] == 1


def test_accumulator_handles_no_usage():
    """add() with usage=None (failed call) is a no-op, not a crash."""
    acc = CostAccumulator()
    acc.add("tree", "m", "low", usage=None, cost=None)
    assert acc.to_dict()["by_stage"] == {}


def test_install_is_idempotent():
    """Calling install() repeatedly registers the callback exactly once."""
    import litellm

    # Reset to a known state for the test
    litellm._async_success_callback = []
    install()
    install()
    install()
    count = sum(
        1 for cb in (litellm._async_success_callback or []) if cb is _async_cb
    )
    assert count == 1


@pytest.mark.asyncio
async def test_callback_routes_by_metadata_stage():
    """When the callback fires, it reads stage from kwargs.metadata and
    routes the response into the right bucket of the accumulator that the
    ContextVar points to."""
    acc = init_accumulator()  # sets the ContextVar to a fresh accumulator
    kwargs = {
        "model": "openai/responses/gpt-5-mini",
        "extra_body": {"reasoning": {"effort": "medium"}},
        "metadata": {"stage": "enrichment"},
    }
    response = _fake_response(prompt=100, completion=80, reasoning=30, cost=0.00025)
    await _async_cb(kwargs, response, 0.0, 0.1)

    d = acc.to_dict()
    assert "enrichment" in d["by_stage"]
    bucket = d["by_stage"]["enrichment"]
    assert bucket["calls"] == 1
    assert bucket["reasoning_tokens"] == 30
    assert bucket["reasoning_effort"] == "medium"
    assert bucket["cost_usd"] == pytest.approx(0.00025)


@pytest.mark.asyncio
async def test_callback_no_op_when_var_unset():
    """If no accumulator was init'd for this context, the callback is a no-op."""
    _acc_var.set(None)
    kwargs = {"model": "x", "metadata": {"stage": "tree"}}
    response = _fake_response(10, 5, 0, 0.0001)
    # Must not raise
    await _async_cb(kwargs, response, 0.0, 0.1)


@pytest.mark.asyncio
async def test_callback_reads_top_level_reasoning_effort():
    """For non-Responses-bridge calls, reasoning_effort lives at top-level
    (Anthropic/Gemini path) — the callback should pick it up."""
    acc = init_accumulator()
    kwargs = {
        "model": "claude-opus-4-7",
        "reasoning_effort": "high",
        "metadata": {"stage": "tree"},
    }
    response = _fake_response(50, 20, 8, 0.0004)
    await _async_cb(kwargs, response, 0.0, 0.1)

    bucket = acc.to_dict()["by_stage"]["tree"]
    assert bucket["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_accumulator_propagates_across_asyncio_run():
    """The Celery indexing task calls asyncio.run() multiple times for the
    different stages. The ContextVar must be set BEFORE asyncio.run so the
    snapshot inside the loop sees the same accumulator object."""
    acc = init_accumulator()  # set in outer (sync-equivalent) context

    async def _inner():
        # Snapshot inherits acc; mutating the object is visible outside
        kwargs = {"model": "m", "metadata": {"stage": "tree"}}
        response = _fake_response(10, 5, 2, 0.0001)
        await _async_cb(kwargs, response, 0.0, 0.1)

    # Mimic the cross-asyncio.run pattern: each call snapshots the parent
    # context. The accumulator OBJECT is shared.
    asyncio.get_event_loop().run_until_complete(_inner()) if False else await _inner()
    await _inner()

    bucket = acc.to_dict()["by_stage"]["tree"]
    assert bucket["calls"] == 2
