"""Tests for Agent.__init__ reasoning_effort precheck and per-model cache."""

from __future__ import annotations

import logging
from unittest.mock import patch

from agentic.agent.agent import Agent


def test_init_with_no_reasoning_effort_caches_none():
    agent = Agent(model="anthropic/claude-opus-4-7")
    assert agent._effort_cache == {"anthropic/claude-opus-4-7": None}


def test_init_with_reasoning_effort_supported_model_caches_value():
    with patch("litellm.supports_reasoning", return_value=True):
        agent = Agent(model="anthropic/claude-opus-4-7", reasoning_effort="medium")
    assert agent._effort_cache == {"anthropic/claude-opus-4-7": "medium"}


def test_init_with_reasoning_effort_unsupported_model_drops_with_breadcrumb(caplog):
    caplog.set_level(logging.INFO)
    with patch("litellm.supports_reasoning", return_value=False):
        agent = Agent(model="openai/gpt-4o", reasoning_effort="medium")
    assert agent._effort_cache == {"openai/gpt-4o": None}
    breadcrumbs = [r for r in caplog.records if "reasoning_effort_dropped" in r.message]
    assert len(breadcrumbs) == 1
    rec = breadcrumbs[0]
    assert rec.model == "openai/gpt-4o"
    assert rec.requested_effort == "medium"
    assert rec.reason == "model_does_not_support_reasoning"


def test_init_with_supports_reasoning_raising_treats_as_unsupported():
    with patch("litellm.supports_reasoning", side_effect=Exception("unknown model")):
        agent = Agent(model="custom/exotic-model", reasoning_effort="high")
    assert agent._effort_cache == {"custom/exotic-model": None}


def test_resolved_effort_for_caches_lazily_on_fallback():
    """Fallback to a different model triggers one new precheck and caches."""
    with patch("litellm.supports_reasoning", return_value=True) as mock_supports:
        agent = Agent(model="anthropic/claude-opus-4-7", reasoning_effort="medium")
        mock_supports.assert_called_once_with(model="anthropic/claude-opus-4-7")
        # Same model: cache hit, no new call
        assert agent._resolved_effort_for("anthropic/claude-opus-4-7") == "medium"
        assert mock_supports.call_count == 1
        # Different model: cache miss, one new call
        assert agent._resolved_effort_for("openai/gpt-5.4") == "medium"
        assert mock_supports.call_count == 2
        # Same fallback again: cache hit
        assert agent._resolved_effort_for("openai/gpt-5.4") == "medium"
        assert mock_supports.call_count == 2
