"""Every streaming agent model call carries an explicit timeout.

Without one, litellm falls back to its `request_timeout` default of 6000s. A
provider that accepts the request and never sends a byte then holds the run for
100 minutes: the run's own `timeout_seconds` and a client abort both set
`abort_signal`, which is only read between stream chunks, so neither can end a
call that is still waiting for its first one.

On a streaming call the timeout bounds each read, not the whole answer. A
non-streaming call sends nothing until the answer is complete, so the same
bound would cap the whole generation; those calls keep litellm's default.
"""

import asyncio
import logging
from unittest.mock import patch

import litellm
import pytest

from agentic import Agent
from agentic.execution.status import ExecutionStatus


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


def _run_async(coro):
    # A private loop: asyncio.run() would leave this thread with no current
    # event loop, breaking later tests that call asyncio.get_event_loop().
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
    with patch(
        "agentic.agent.agent.litellm.acompletion", side_effect=_capture_async(seen)
    ):
        _run_async(agent.arun("hi"))
    return seen


def _astream(agent: Agent) -> dict:
    seen: dict = {}

    async def _drain():
        async for _ in agent.astream("hi"):
            pass

    with patch(
        "agentic.agent.agent.litellm.acompletion", side_effect=_capture_async(seen)
    ):
        with pytest.raises(_Stop):
            _run_async(_drain())
    return seen


STREAMING = [_run, _stream, _astream]  # _run: with AGENT_LLM_STREAMING_ENABLED=true
NON_STREAMING = [_run, _arun]  # _run: with AGENT_LLM_STREAMING_ENABLED=false
_ids = lambda f: f.__name__.strip("_")  # noqa: E731


@pytest.fixture
def streaming_on(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")


@pytest.mark.parametrize("call", STREAMING, ids=_ids)
def test_every_streaming_call_is_bounded_by_default(call, monkeypatch, streaming_on):
    monkeypatch.delenv("AGENT_LLM_TIMEOUT_SECONDS", raising=False)
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert seen["timeout"] == 300.0


@pytest.mark.parametrize("call", STREAMING, ids=_ids)
def test_the_bound_is_read_from_the_environment_per_call(
    call, monkeypatch, streaming_on
):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "42")
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert seen["timeout"] == 42.0


@pytest.mark.parametrize("value", ["0", "-5"])
@pytest.mark.parametrize("call", STREAMING, ids=_ids)
def test_zero_or_less_leaves_the_call_to_litellms_own_default(
    call, value, monkeypatch, streaming_on
):
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", value)
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert "timeout" not in seen


@pytest.mark.parametrize("value", ["", "abc"])
def test_a_malformed_value_warns_and_keeps_the_default(
    value, monkeypatch, streaming_on, caplog
):
    # An empty string is what a compose `${VAR:-}` passthrough produces.
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", value)
    with caplog.at_level(logging.WARNING, logger="agentic.agent.agent"):
        seen = _run(Agent(model="openai/gpt-4o-mini"))
    assert seen["timeout"] == 300.0
    assert "AGENT_LLM_TIMEOUT_SECONDS" in caplog.text


@pytest.mark.parametrize("call", NON_STREAMING, ids=_ids)
def test_a_non_streaming_call_keeps_litellms_default(call, monkeypatch):
    # conftest pins AGENT_LLM_STREAMING_ENABLED=false; arun() never streams.
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "42")
    seen = call(Agent(model="openai/gpt-4o-mini"))
    assert "timeout" not in seen


def test_a_timed_out_call_fails_the_run_without_falling_back(monkeypatch, streaming_on):
    # litellm appends "time taken=<s> seconds" to a Timeout on the OpenAI path.
    # Matched as text, a "500" there read as a server error and sent the run
    # to the fallback model.
    err = litellm.Timeout(
        message="Request timed out. - timeout value=500.0, time taken=500.03 seconds",
        model="gpt-4o-mini",
        llm_provider="openai",
    )
    with patch("agentic.agent.agent.litellm.completion", side_effect=err) as fake:
        out = Agent(model="openai/gpt-4o-mini").run(
            "hi", fallback_model="openai/gpt-4o"
        )
    assert out.status == ExecutionStatus.FAILED
    assert fake.call_count == 1
