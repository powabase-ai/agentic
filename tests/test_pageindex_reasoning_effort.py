"""Tests that reasoning_effort is plumbed from a ContextVar into the
_llm_completion call. Mirrors the existing _llm_semaphore_var pattern in
agentic/src/agentic/knowledge/indexing/_pageindex_lib/utils.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic.knowledge.indexing._pageindex_lib.utils import (
    _llm_completion,
    init_reasoning_effort,
)


def _fake_response(text: str = "ok"):
    class _C:
        message = type("M", (), {"content": text})
        finish_reason = "stop"
    class _R:
        choices = [_C()]
    return _R()


@pytest.mark.asyncio
async def test_no_reasoning_effort_omits_kwargs():
    """When the ContextVar is unset, _llm_completion must NOT pass
    reasoning_effort or extra_body to litellm.acompletion."""
    init_reasoning_effort(None)
    with patch(
        "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
        new=AsyncMock(return_value=_fake_response()),
    ) as mock_acompletion:
        await _llm_completion(model="gpt-4o", prompt="hello")
    kwargs = mock_acompletion.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_for_anthropic_passes_top_level():
    """For non-OpenAI providers, the param goes through as top-level
    reasoning_effort (LiteLLM normalizes to provider-native shape)."""
    init_reasoning_effort("low")
    with patch(
        "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
        new=AsyncMock(return_value=_fake_response()),
    ) as mock_acompletion:
        await _llm_completion(model="anthropic/claude-opus-4-7", prompt="hi")
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("reasoning_effort") == "low"
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_for_openai_routes_through_responses():
    """For OpenAI reasoning models, the model is rewritten to the Responses
    bridge and the effort is packed into extra_body.reasoning."""
    init_reasoning_effort("low")
    with (
        patch(
            "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
            new=AsyncMock(return_value=_fake_response()),
        ) as mock_acompletion,
        patch(
            "agentic.llm.routing.litellm.supports_reasoning",
            return_value=True,
        ),
    ):
        await _llm_completion(model="openai/gpt-5-mini", prompt="hi")
    kwargs = mock_acompletion.await_args.kwargs
    # Top-level reasoning_effort must be absent (would be silently dropped
    # on the Responses bridge per routing.py docstring).
    assert "reasoning_effort" not in kwargs
    assert kwargs["model"] == "openai/responses/gpt-5-mini"
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "low", "summary": "detailed"}
    }


@pytest.mark.asyncio
async def test_init_reasoning_effort_clears_to_none():
    """Calling init_reasoning_effort(None) after a previous set must clear
    state — sequential indexing tasks must not leak effort across runs."""
    init_reasoning_effort("high")
    init_reasoning_effort(None)
    with patch(
        "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
        new=AsyncMock(return_value=_fake_response()),
    ) as mock_acompletion:
        await _llm_completion(model="gpt-4o", prompt="hello")
    kwargs = mock_acompletion.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_extractor_reads_reasoning_effort_from_extra():
    """PageIndexExtractor.extract() must call init_reasoning_effort with the
    value from config.extra['reasoning_effort'] before invoking the pipeline."""
    from agentic.knowledge.indexing import PageIndexAlgorithm
    from agentic.knowledge.models import IndexingConfig

    captured: dict[str, object] = {}

    async def fake_md_to_tree(**kwargs):
        # Read the contextvar at call time — this is what the pipeline does.
        from agentic.knowledge.indexing._pageindex_lib.utils import (
            _reasoning_effort_var,
        )
        captured["effort"] = _reasoning_effort_var.get()
        return {"doc_name": "x", "structure": []}

    config = IndexingConfig(
        strategy="page_index",
        extra={
            "source_name": "doc",
            "model": "gpt-5-mini",
            "reasoning_effort": "low",
            "if_add_node_summary": "no",
        },
    )

    with patch(
        "agentic.knowledge.indexing._pageindex_lib.page_index_md.md_to_tree",
        new=AsyncMock(side_effect=fake_md_to_tree),
    ):
        await PageIndexAlgorithm().aindex(
            content="# Hello\n\nworld\n",
            config=config,
            source_id="src1",
        )

    assert captured["effort"] == "low"


@pytest.mark.asyncio
async def test_extractor_omits_reasoning_effort_when_unset():
    """If extra has no reasoning_effort key, the contextvar must remain None."""
    from agentic.knowledge.indexing import PageIndexAlgorithm
    from agentic.knowledge.indexing._pageindex_lib.utils import (
        _reasoning_effort_var,
        init_reasoning_effort,
    )
    from agentic.knowledge.models import IndexingConfig

    init_reasoning_effort(None)  # ensure clean state
    captured: dict[str, object] = {}

    async def fake_md_to_tree(**kwargs):
        captured["effort"] = _reasoning_effort_var.get()
        return {"doc_name": "x", "structure": []}

    config = IndexingConfig(
        strategy="page_index",
        extra={
            "source_name": "doc",
            "model": "gpt-5-mini",
            "if_add_node_summary": "no",
        },
    )

    with patch(
        "agentic.knowledge.indexing._pageindex_lib.page_index_md.md_to_tree",
        new=AsyncMock(side_effect=fake_md_to_tree),
    ):
        await PageIndexAlgorithm().aindex(
            content="# Hello\n\nworld\n",
            config=config,
            source_id="src1",
        )

    assert captured["effort"] is None
