"""OpenRouter returns a model's reasoning as `reasoning_details` and accepts it
back on an assistant message. Within a run, every step replays the previous
steps' details, so a reasoning model keeps its own earlier thinking between
tool calls. Across runs it is not replayed."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from agentic.agent.agent import Agent
from agentic.agent.output import AgentOutput
from agentic.agent.session import AgentSession
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus

_MODEL = "openrouter/moonshotai/kimi-k3"
_DETAILS = [
    {
        "type": "reasoning.text",
        "text": "I should probe.",
        "format": "unknown",
        "index": 0,
    }
]
_EVENT = "reasoning_dropped_at_provider_switch"


def _probe_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: "ok",
    )


# --- The real LiteLLM OpenRouter stream, over a mocked HTTP transport -------


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "moonshotai/kimi-k3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _reasoning_delta(text: str) -> dict:
    return {
        "content": "",
        "reasoning": text,
        "reasoning_details": [
            {"type": "reasoning.text", "text": text, "format": "unknown", "index": 0}
        ],
    }


_TOOL_STEP = _sse(
    _chunk({"role": "assistant", **_reasoning_delta("I should ")}),
    _chunk(_reasoning_delta("probe.")),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ]
        },
        "tool_calls",
    ),
)
_ANSWER_STEP = _sse(_chunk({"role": "assistant", "content": "done"}, "stop"))


def _completion_json(message: dict, finish_reason: str) -> bytes:
    return json.dumps(
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": "moonshotai/kimi-k3",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    ).encode()


_TOOL_STEP_JSON = _completion_json(
    {
        "role": "assistant",
        "content": "",
        "reasoning": "I should probe.",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "I should probe.",
                "format": "unknown",
                "index": 0,
            }
        ],
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "probe", "arguments": "{}"},
            }
        ],
    },
    "tool_calls",
)
_ANSWER_STEP_JSON = _completion_json({"role": "assistant", "content": "done"}, "stop")


def _run_streamed(responses: list[bytes], stream: bool = True) -> list[dict]:
    """Run a two-step tool loop through LiteLLM's own OpenRouter
    transformation; returns the JSON bodies it posted."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream" if stream else "application/json"
            },
            content=responses[len(bodies) - 1],
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))
    real_completion = litellm.completion

    def completion(**kwargs):
        return real_completion(**kwargs, client=client, api_key="test-key")

    with (
        patch("agentic.agent.agent.litellm.completion", side_effect=completion),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": str(stream).lower()}),
    ):
        agent = Agent(model=_MODEL, reasoning_effort="high")
        output = agent.run(
            "hi", context=ExecutionContext(), tools={"probe": _probe_tool()}
        )
    assert output.status.is_success()
    return bodies


def _prior_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m.get("role") == "assistant"][-1]


def test_streamed_reasoning_details_are_posted_back_on_the_next_step():
    bodies = _run_streamed([_TOOL_STEP, _ANSWER_STEP])
    assert len(bodies) == 2
    assert _prior_assistant(bodies[1]["messages"])["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "text": "I should probe.",
            "format": "unknown",
            "index": 0,
        }
    ]


def test_non_streamed_reasoning_details_are_posted_back_on_the_next_step():
    """LiteLLM's non-streaming OpenRouter message carries the details in
    `provider_specific_fields`, not as an attribute of their own."""
    bodies = _run_streamed([_TOOL_STEP_JSON, _ANSWER_STEP_JSON], stream=False)
    assert len(bodies) == 2
    assert _prior_assistant(bodies[1]["messages"])["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "text": "I should probe.",
            "format": "unknown",
            "index": 0,
        }
    ]


# --- Mocked litellm.completion (non-streaming shapes) -----------------------


def _tool_step(details=_DETAILS, summary="I should probe."):
    msg = SimpleNamespace(
        content=None,
        role="assistant",
        reasoning_content=summary,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="probe", arguments="{}"),
            )
        ],
        thinking_blocks=None,
        provider_specific_fields=None,
    )
    if details is not None:
        msg.reasoning_details = details
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        id="resp_1",
    )


def _answer_step():
    msg = SimpleNamespace(
        content="done",
        role="assistant",
        reasoning_content=None,
        tool_calls=None,
        thinking_blocks=None,
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="resp_2",
    )


def _run_two_steps(first_step):
    with (
        patch(
            "agentic.agent.agent.litellm.completion",
            side_effect=[first_step, _answer_step()],
        ) as completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model=_MODEL, reasoning_effort="high")
        output = agent.run(
            "hi", context=ExecutionContext(), tools={"probe": _probe_tool()}
        )
    assert output.status.is_success()
    assert completion.call_count == 2
    return output, completion.call_args_list


def test_reasoning_details_replayed_on_the_next_step():
    output, calls = _run_two_steps(_tool_step())
    assert (
        _prior_assistant(calls[1].kwargs["messages"])["reasoning_details"] == _DETAILS
    )
    step_one = [m for m in output.messages if m.get("role") == "assistant"][0]
    assert step_one["reasoning"]["provider"] == "openrouter"
    assert step_one["reasoning"]["reasoning_details"] == _DETAILS
    assert step_one["reasoning"]["summary_text"] == "I should probe."


def test_reasoning_text_alone_is_recorded_but_not_replayed():
    """A response with reasoning text but no `reasoning_details` anywhere: the
    step is recorded, nothing is replayed."""
    output, calls = _run_two_steps(_tool_step(details=None))
    assert "reasoning_details" not in _prior_assistant(calls[1].kwargs["messages"])
    step_one = [m for m in output.messages if m.get("role") == "assistant"][0]
    assert step_one["reasoning"]["provider"] == "openrouter"
    assert step_one["reasoning"]["summary_text"] == "I should probe."


# --- Fallback to another OpenRouter model -----------------------------------


class _RateLimited(Exception):
    status_code = 429


def _run_with_fallback(fallback_model):
    events: list[dict] = []
    responses = [_tool_step(), _RateLimited("provider error"), _answer_step()]

    def fake_completion(**kwargs):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with (
        patch(
            "agentic.agent.agent.litellm.completion", side_effect=fake_completion
        ) as completion,
        patch("agentic.agent.agent.classify_error", return_value="rate_limit"),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model=_MODEL, reasoning_effort="high")
        output = agent.run(
            "hi",
            context=ExecutionContext(on_event=events.append),
            tools={"probe": _probe_tool()},
            fallback_model=fallback_model,
        )
    assert output.status.is_success()
    assert completion.call_count == 3
    dropped = [
        {k: v for k, v in e.items() if k not in ("seq", "ts")}
        for e in events
        if e.get("type") == _EVENT
    ]
    return completion.call_args_list[2], dropped


def test_fallback_to_another_vendor_on_openrouter_drops_the_details():
    """Both models resolve to the `openrouter` LiteLLM provider, but the
    reasoning came from another vendor's model."""
    fallback_call, dropped = _run_with_fallback("openrouter/anthropic/claude-sonnet-5")
    for message in fallback_call.kwargs["messages"]:
        assert "reasoning_details" not in message
    assert dropped == [
        {
            "type": _EVENT,
            "step": 2,
            "from_provider": "openrouter/moonshotai",
            "to_provider": "openrouter/anthropic",
        }
    ]


def test_fallback_to_the_same_vendor_on_openrouter_keeps_the_details():
    fallback_call, dropped = _run_with_fallback("openrouter/moonshotai/kimi-k2.5")
    assert (
        _prior_assistant(fallback_call.kwargs["messages"])["reasoning_details"]
        == _DETAILS
    )
    assert dropped == []


# --- Across runs -------------------------------------------------------------


def _earlier_answer():
    return {
        "role": "assistant",
        "content": "earlier answer",
        "reasoning_details": [dict(d) for d in _DETAILS],
        "reasoning": {"provider": "openrouter", "reasoning_details": _DETAILS},
    }


def _first_request_messages(run_input, session=None):
    with (
        patch(
            "agentic.agent.agent.litellm.completion", return_value=_answer_step()
        ) as completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model=_MODEL, reasoning_effort="high")
        output = agent.run(run_input, context=ExecutionContext(), session=session)
    assert output.status.is_success()
    return completion.call_args_list[0].kwargs["messages"]


def test_session_history_never_replays_reasoning_details():
    session = AgentSession()
    session.add_output(
        AgentOutput(
            execution_id="earlier",
            status=ExecutionStatus.COMPLETED,
            content="earlier answer",
            messages=[
                {"role": "user", "content": "earlier question"},
                _earlier_answer(),
            ],
        )
    )
    messages = _first_request_messages("next question", session=session)
    assert any(m.get("content") == "earlier answer" for m in messages)
    for message in messages:
        assert "reasoning_details" not in message


def test_list_input_never_replays_reasoning_details():
    messages = _first_request_messages(
        [
            {"role": "user", "content": "earlier question"},
            _earlier_answer(),
            {"role": "user", "content": "next question"},
        ]
    )
    assert any(m.get("content") == "earlier answer" for m in messages)
    for message in messages:
        assert "reasoning_details" not in message
