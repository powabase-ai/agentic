"""What the Agent actually sends to the newest models, with litellm's HTTP
layer intercepted (no network).

These models reject parameters older ones accepted: Claude 5.5 / Fable 5.1
400 on any temperature, GPT-5.6 / GPT-6 reject a non-default temperature
while reasoning, and each takes its own subset of effort levels. A user's
saved temperature or effort must not turn into a provider error.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic import Agent


class _Captured(Exception):
    pass


def _sent_bodies(agent_call):
    """Run ``agent_call`` with every outgoing HTTP POST recorded and failed,
    and return the recorded JSON bodies. The Agent's own error handling
    swallows the failure; the bodies are what litellm put on the wire."""
    bodies: list[dict] = []

    def fake_post(self, url, data=None, json_=None, headers=None, **kw):
        body = kw.get("json") if kw.get("json") is not None else json_
        if body is None:
            body = json.loads(data)
        bodies.append(body)
        raise _Captured(url)

    def wrapped(self, url, data=None, json=None, headers=None, **kw):
        return fake_post(self, url, data=data, json_=json, headers=headers, **kw)

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", wrapped):
        try:
            agent_call()
        except Exception:
            pass
    assert bodies, "no request reached the HTTP layer"
    return bodies


def _effort(body: dict):
    return (
        (body.get("output_config") or {}).get("effort")
        or (body.get("reasoning") or {}).get("effort")
        or body.get("reasoning_effort")
    )


@pytest.mark.parametrize(
    "model,requested,sent",
    [
        ("claude-sonnet-5-5", "minimal", "low"),
        ("claude-opus-5-5", "medium", "medium"),
        ("claude-fable-5-1", "none", "low"),
        ("gpt-6-astra", "minimal", "low"),
        ("gpt-5.6", "high", "high"),
    ],
)
def test_run_drops_temperature_and_normalizes_effort(model, requested, sent):
    agent = Agent(
        model=model,
        system_prompt="test",
        temperature=0.3,
        reasoning_effort=requested,
        api_key="sk-test",
    )
    body = _sent_bodies(lambda: agent.run("hi"))[0]

    assert "temperature" not in body
    assert _effort(body) == sent


def test_run_normalizes_kimi_k3_effort_and_keeps_its_temperature():
    """Kimi K3 takes low/high/max only, but does take a temperature."""
    agent = Agent(
        model="openrouter/moonshotai/kimi-k3",
        system_prompt="test",
        temperature=0.3,
        reasoning_effort="medium",
        api_key="sk-test",
    )
    body = _sent_bodies(lambda: agent.run("hi"))[0]

    assert body["temperature"] == 0.3
    assert _effort(body) == "high"


def test_run_keeps_temperature_where_the_model_accepts_it():
    """drop_params removes only what the model rejects. (An Anthropic model:
    litellm's OpenAI chat route goes through the OpenAI SDK, past the
    intercepted HTTP layer.)"""
    agent = Agent(
        model="claude-haiku-4-5",
        system_prompt="test",
        temperature=0.3,
        api_key="sk-test",
    )
    body = _sent_bodies(lambda: agent.run("hi"))[0]

    assert body["temperature"] == 0.3


def test_stream_drops_temperature_on_claude_5_5():
    agent = Agent(
        model="claude-sonnet-5-5",
        system_prompt="test",
        temperature=0.3,
        reasoning_effort="minimal",
        api_key="sk-test",
    )

    def consume():
        for _ in agent.stream("hi"):
            pass

    body = _sent_bodies(consume)[0]

    assert "temperature" not in body
    assert _effort(body) == "low"


def _async_response():
    message = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=message, finish_reason="stop")
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return MagicMock(choices=[choice], usage=usage)


@pytest.mark.asyncio
async def test_arun_asks_litellm_to_drop_unsupported_params():
    with patch("agentic.agent.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=_async_response())
        agent = Agent(model="claude-sonnet-5-5", system_prompt="t", temperature=0.3)
        await agent.arun("hi")

    assert mock_litellm.acompletion.call_args.kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_astream_asks_litellm_to_drop_unsupported_params():
    async def empty_stream():
        if False:
            yield None

    with patch("agentic.agent.agent.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=empty_stream())
        agent = Agent(model="claude-sonnet-5-5", system_prompt="t", temperature=0.3)
        async for _ in agent.astream("hi"):
            pass

    assert mock_litellm.acompletion.call_args.kwargs["drop_params"] is True
