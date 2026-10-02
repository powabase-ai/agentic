"""What the Agent actually sends to the newest models, intercepted at
``httpx.Client.send`` / ``httpx.AsyncClient.send`` -- below both litellm's own
HTTP handler and the OpenAI SDK, so every route is covered. No network.

These models reject parameters older ones accepted: Claude 5.5 / Fable 5.1
400 on any temperature, GPT-5.6 / GPT-6 reject a non-default temperature
while reasoning, and each takes its own subset of effort levels. A user's
saved temperature or effort must not turn into a provider error -- and
nothing else in the request may be dropped to get there.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from agentic import Agent, ExecutionStatus
from agentic.agent.tools import BuiltinTool


class _Captured(Exception):
    pass


def _intercept():
    sent: list[tuple[str, dict]] = []

    def send(self, request, *args, **kwargs):
        # Model calls only; the agent's context-window lookup also GETs a
        # model registry.
        if request.method == "POST":
            sent.append((str(request.url), json.loads(request.content or b"{}")))
        raise _Captured(str(request.url))

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    patches = (
        patch.object(httpx.Client, "send", send),
        patch.object(httpx.AsyncClient, "send", asend),
    )
    return sent, patches


def _sent(agent_call):
    """Run ``agent_call`` with every outgoing request recorded and failed;
    return (url, body) of the first. The Agent's own error handling swallows
    the failure."""
    sent, (p1, p2) = _intercept()
    with p1, p2:
        try:
            agent_call()
        except Exception:
            pass
    assert sent, "no request reached the HTTP layer"
    return sent[0]


async def _asent(agent_call):
    sent, (p1, p2) = _intercept()
    with p1, p2:
        try:
            await agent_call()
        except Exception:
            pass
    assert sent, "no request reached the HTTP layer"
    return sent[0]


def _agent(model, **kwargs):
    kwargs.setdefault("temperature", 0.3)
    return Agent(model=model, system_prompt="test", api_key="sk-test", **kwargs)


@pytest.mark.parametrize(
    "model,requested,sent_effort,route",
    [
        ("claude-sonnet-5-5", "minimal", "low", "/v1/messages"),
        ("claude-opus-5-5", "medium", "medium", "/v1/messages"),
        ("claude-fable-5-1", "none", "low", "/v1/messages"),
        ("gpt-6-astra", "minimal", "low", "/v1/responses"),
        ("gpt-5.6", "high", "high", "/v1/responses"),
    ],
)
def test_run_drops_temperature_and_normalizes_effort(
    model, requested, sent_effort, route
):
    url, body = _sent(lambda: _agent(model, reasoning_effort=requested).run("hi"))

    assert url.endswith(route)
    assert "temperature" not in body
    effort = (body.get("output_config") or {}).get("effort") or (
        body.get("reasoning") or {}
    ).get("effort")
    assert effort == sent_effort


def test_run_on_openai_chat_completions_drops_a_rejected_temperature():
    """No effort set: gpt-5.6 goes to chat completions (through the OpenAI
    SDK), at a default effort that is not 'none', so the temperature goes."""
    url, body = _sent(lambda: _agent("gpt-5.6").run("hi"))

    assert url.endswith("/chat/completions")
    assert "temperature" not in body


def test_run_on_openai_chat_completions_keeps_an_accepted_temperature():
    url, body = _sent(lambda: _agent("gpt-4o").run("hi"))

    assert url.endswith("/chat/completions")
    assert body["temperature"] == 0.3


def test_run_keeps_temperature_on_a_reasoning_model_at_effort_none():
    """Effort 'none' stays on chat completions: there is no reasoning to
    bring back, and on the Responses route litellm can't see the effort."""
    url, body = _sent(lambda: _agent("gpt-6-luna", reasoning_effort="none").run("hi"))

    assert url.endswith("/chat/completions")
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.3


def test_run_drops_temperature_next_to_budget_thinking():
    """Anthropic rejects a temperature alongside thinking.type=enabled."""
    url, body = _sent(
        lambda: _agent("claude-haiku-4-5", reasoning_effort="medium").run("hi")
    )

    assert body["thinking"]["type"] == "enabled"
    assert "temperature" not in body


def test_run_keeps_temperature_on_budget_claude_without_reasoning():
    url, body = _sent(lambda: _agent("claude-haiku-4-5").run("hi"))

    assert body["temperature"] == 0.3


def test_run_normalizes_kimi_k3_effort_and_keeps_its_temperature():
    """Kimi K3 takes low/high/max only, but does take a temperature."""
    url, body = _sent(
        lambda: _agent("openrouter/moonshotai/kimi-k3", reasoning_effort="medium").run(
            "hi"
        )
    )

    assert url.endswith("/chat/completions")
    assert body["temperature"] == 0.3
    assert body["reasoning_effort"] == "high"


def test_run_on_gemini_keeps_temperature_and_maps_effort():
    url, body = _sent(
        lambda: _agent("gemini/gemini-3.8-flash", reasoning_effort="xhigh").run("hi")
    )

    assert ":generateContent" in url
    config = body["generationConfig"]
    assert config["temperature"] == 0.3
    assert config["thinkingConfig"]["thinkingLevel"] == "high"


def test_tools_are_never_dropped_to_make_a_call_go_through():
    """Only the temperature is ever left out. On a model litellm says takes
    no tools, the run fails before anything is sent -- a blanket drop_params
    would have stripped the tools and run the agent toolless."""
    tool = BuiltinTool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: "ok",
    )
    agent = _agent("perplexity/sonar")
    sent, (p1, p2) = _intercept()
    with p1, p2:
        output = agent.run("hi", tools={"echo": tool})

    assert sent == []
    assert output.status == ExecutionStatus.FAILED
    assert "tools" in (output.error or "")


def test_stream_drops_temperature_on_claude_5_5():
    def consume():
        for _ in _agent("claude-sonnet-5-5", reasoning_effort="minimal").stream("hi"):
            pass

    url, body = _sent(consume)

    assert url.endswith("/v1/messages")
    assert "temperature" not in body
    assert body["output_config"]["effort"] == "low"


@pytest.mark.asyncio
async def test_arun_drops_temperature_on_claude_5_5():
    url, body = await _asent(lambda: _agent("claude-sonnet-5-5").arun("hi"))

    assert url.endswith("/v1/messages")
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_arun_keeps_temperature_where_accepted():
    url, body = await _asent(lambda: _agent("claude-haiku-4-5").arun("hi"))

    assert body["temperature"] == 0.3


@pytest.mark.asyncio
async def test_astream_drops_temperature_on_claude_5_5():
    async def consume():
        async for _ in _agent("claude-sonnet-5-5").astream("hi"):
            pass

    url, body = await _asent(consume)

    assert url.endswith("/v1/messages")
    assert "temperature" not in body


def _consume(gen):
    for _ in gen:
        pass


def test_stream_keeps_an_accepted_temperature():
    url, body = _sent(lambda: _consume(_agent("claude-haiku-4-5").stream("hi")))

    assert body["temperature"] == 0.3


def test_stream_drops_temperature_next_to_budget_thinking():
    url, body = _sent(
        lambda: _consume(
            _agent("claude-haiku-4-5", reasoning_effort="medium").stream("hi")
        )
    )

    assert body["thinking"]["type"] == "enabled"
    assert "temperature" not in body


def test_stream_on_openai_keeps_temperature_at_effort_none():
    url, body = _sent(
        lambda: _consume(_agent("gpt-6-luna", reasoning_effort="none").stream("hi"))
    )

    assert url.endswith("/chat/completions")
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.3


def test_stream_on_responses_sends_the_normalized_effort():
    url, body = _sent(
        lambda: _consume(_agent("gpt-6-astra", reasoning_effort="minimal").stream("hi"))
    )

    assert url.endswith("/v1/responses")
    assert body["reasoning"]["effort"] == "low"
    assert "temperature" not in body


def test_stream_on_gemini_keeps_temperature():
    url, body = _sent(
        lambda: _consume(
            _agent("gemini/gemini-3.8-flash", reasoning_effort="xhigh").stream("hi")
        )
    )

    assert ":streamGenerateContent" in url
    assert body["generationConfig"]["temperature"] == 0.3
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


@pytest.mark.asyncio
async def test_astream_keeps_an_accepted_temperature():
    async def consume():
        async for _ in _agent("claude-haiku-4-5").astream("hi"):
            pass

    url, body = await _asent(consume)

    assert body["temperature"] == 0.3
