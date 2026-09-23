"""Every agent model call carries an explicit timeout.

Without one, litellm falls back to its `request_timeout` default of 6000s. A
provider that accepts the request and never sends a byte then holds the run for
100 minutes: the run's own `timeout_seconds` and a client abort both set
`abort_signal`, which is only read between stream chunks, so neither can end a
call that is still waiting for its first one.
"""

import asyncio
from unittest.mock import patch

import pytest

from agentic import Agent


class _Stop(Exception):
    """Raised by the fake provider once it has seen the call's kwargs."""


def _capture(seen: dict):
    def _fake(**kwargs):
        seen.update(kwargs)
        raise _Stop()

    return _fake


def _capture_async(seen: dict):
    async def _fake(**kwargs):
        seen.update(kwargs)
        raise _Stop()

    return _fake


def _run(agent: Agent) -> dict:
    seen: dict = {}
    with patch("agentic.agent.agent.litellm.completion", side_effect=_capture(seen)):
        agent.run("hi")
    return seen


def _stream(agent: Agent) -> dict:
    seen: dict = {}
    with patch("agentic.agent.agent.litellm.completion", side_effect=_capture(seen)):
        try:  # stream() reports a provider error itself; only the kwargs matter
            list(agent.stream("hi"))
        except _Stop:
            pass
    return seen


def _arun(agent: Agent) -> dict:
    seen: dict = {}
    with patch("agentic.agent.agent.litellm.acompletion", side_effect=_capture_async(seen)):
        asyncio.run(agent.arun("hi"))
    return seen


def _astream(agent: Agent) -> dict:
    seen: dict = {}

    async def _drain():
        async for _ in agent.astream("hi"):
            pass

    with patch("agentic.agent.agent.litellm.acompletion", side_effect=_capture_async(seen)):
        with pytest.raises(_Stop):
            asyncio.run(_drain())
    return seen


CALL_SITES = [_run, _stream, _arun, _astream]


@pytest.mark.parametrize("call", CALL_SITES, ids=lambda f: f.__name__.strip("_"))
def test_every_agent_call_is_bounded_by_default(call, monkeypatch):
    monkeypatch.delenv("AGENT_LLM_TIMEOUT_SECONDS", raising=False)
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert seen["timeout"] == 600.0


@pytest.mark.parametrize("call", CALL_SITES, ids=lambda f: f.__name__.strip("_"))
def test_the_bound_is_read_from_the_environment_per_call(call, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "42")
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert seen["timeout"] == 42.0


@pytest.mark.parametrize("call", CALL_SITES, ids=lambda f: f.__name__.strip("_"))
def test_zero_leaves_the_call_to_litellms_own_default(call, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "0")
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert "timeout" not in seen
