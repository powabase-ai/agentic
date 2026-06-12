"""Behavioral tests for reasoning-effort routing in tree_search retrieval.

The string-match guard in the project-service wiring test only proves the
*tokens* are present; this proves the *behavior* — specifically the
route-then-kwargs ORDERING that agentic.llm.routing exists to enforce:

  - OpenAI reasoning models must route through the Responses bridge AND pack
    the effort into extra_body['reasoning'] (top-level reasoning_effort is
    silently dropped on that bridge — the exact bug this guards).
  - Non-OpenAI providers (Anthropic) pass a top-level reasoning_effort, no
    route change.
  - No configured effort => no model rewrite and no reasoning kwargs at all.

Driven through TreeSearchAlgorithm.select_documents with litellm.acompletion
mocked so we capture the outgoing kwargs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from agentic.knowledge.retrieval.tree_search import TreeSearchAlgorithm


def _fake_resp(content: str = '["d0"]'):
    return type(
        "R",
        (),
        {
            "choices": [
                type("C", (), {"message": type("M", (), {"content": content})()})()
            ]
        },
    )()


def _run_select_documents(config: dict):
    algo = TreeSearchAlgorithm()
    mock = AsyncMock(return_value=_fake_resp())
    with patch("litellm.acompletion", new=mock):
        asyncio.run(
            algo.select_documents(
                query="what is the refund policy",
                toc_records=[{"doc_name": "Doc", "structure": []}],
                config=config,
            )
        )
    return mock.await_args.kwargs


def test_no_effort_leaves_model_and_kwargs_untouched():
    kwargs = _run_select_documents({"retrieval_model": "gpt-5-mini"})
    assert kwargs["model"] == "gpt-5-mini"
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_anthropic_effort_passes_top_level_no_route():
    kwargs = _run_select_documents(
        {"retrieval_model": "anthropic/claude-opus-4-6", "retrieval_reasoning_effort": "low"}
    )
    assert kwargs["model"] == "anthropic/claude-opus-4-6"
    assert kwargs.get("reasoning_effort") == "low"
    assert "extra_body" not in kwargs


def test_openai_effort_routes_through_responses_with_extra_body():
    # supports_reasoning is patched for hermeticity; the routing helper gates on
    # it. The assertion is the route-then-pack ordering: extra_body is built
    # against the ROUTED model, and top-level reasoning_effort is NOT sent.
    with (
        patch("agentic.llm.routing.litellm.supports_reasoning", return_value=True),
        patch.dict("os.environ", {"OPENAI_REASONING_SUMMARY": ""}),
    ):
        kwargs = _run_select_documents(
            {"retrieval_model": "openai/gpt-5-mini", "retrieval_reasoning_effort": "medium"}
        )
    assert kwargs["model"] == "openai/responses/gpt-5-mini"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert "reasoning_effort" not in kwargs
