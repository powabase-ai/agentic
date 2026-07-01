"""Tests that _llm_completion tags every litellm call with
metadata={"stage": "tree"}, so the LiteLLM cost-accumulator callback can
route the cost into the right per-stage bucket on the indexed_source row.
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
async def test_llm_completion_tags_metadata_stage_tree():
    """Every tree-building call must carry metadata={"stage": "tree"}."""
    init_reasoning_effort(None)
    with patch(
        "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
        new=AsyncMock(return_value=_fake_response()),
    ) as mock_acompletion:
        await _llm_completion(model="gpt-4o", prompt="hi")
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("metadata") == {"stage": "tree"}


@pytest.mark.asyncio
async def test_metadata_present_with_reasoning_effort_set():
    """Adding metadata must not break the existing reasoning_effort plumbing."""
    init_reasoning_effort("low")
    with patch(
        "agentic.knowledge.indexing._pageindex_lib.utils.litellm.acompletion",
        new=AsyncMock(return_value=_fake_response()),
    ) as mock_acompletion:
        await _llm_completion(model="anthropic/claude-opus-4-7", prompt="hi")
    kwargs = mock_acompletion.await_args.kwargs
    assert kwargs.get("metadata") == {"stage": "tree"}
    assert kwargs.get("thinking") == {"type": "adaptive", "display": "summarized"}
    assert kwargs.get("output_config") == {"effort": "low"}
