"""Tests that the assistant msg_dict gets a reasoning field attached when an
artifact extracts (intra-run replay — the Anthropic 400 fix)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext


def _make_response_with_thinking():
    msg = MagicMock()
    msg.content = "answer"
    msg.tool_calls = None
    msg.reasoning_content = "I considered X."
    msg.thinking_blocks = [
        {"type": "thinking", "thinking": "I considered X.", "signature": "sig1"}
    ]
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=200, total_tokens=210
    )
    response.id = "resp1"
    return response


def test_anthropic_msg_dict_carries_reasoning_field():
    """After a Claude run with thinking, state.messages assistant entry has
    a `reasoning` field with the discriminated artifact dump."""
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion",
            return_value=_make_response_with_thinking(),
        ),
        patch(
            "agentic.llm.reasoning_extractor.litellm.get_llm_provider",
            return_value=("claude-opus-4-7", "anthropic", None, None),
        ),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7", reasoning_effort="medium")
        output = agent.run("hi", context=ExecutionContext())

    # state.messages is exposed via output.messages
    assistant_msgs = [m for m in output.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    msg = assistant_msgs[0]
    assert "reasoning" in msg
    assert msg["reasoning"]["provider"] == "anthropic"
    assert msg["reasoning"]["thinking_blocks"] == [
        {"type": "thinking", "thinking": "I considered X.", "signature": "sig1"}
    ]


def test_no_artifact_means_no_reasoning_key():
    """Run with no reasoning_effort → no reasoning field on msg_dict."""
    msg = MagicMock()
    msg.content = "answer"
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.thinking_blocks = None
    msg.provider_specific_fields = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.id = None

    with (
        patch("agentic.agent.agent.litellm.completion", return_value=response),
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        agent = Agent(model="anthropic/claude-opus-4-7")  # no reasoning_effort
        output = agent.run("hi", context=ExecutionContext())

    assistant_msgs = [m for m in output.messages if m.get("role") == "assistant"]
    assert "reasoning" not in assistant_msgs[0]


_BLOCKS = [{"type": "thinking", "thinking": "I should probe.", "signature": "sig1"}]
_ITEM = {"id": "rs_1", "type": "reasoning", "encrypted_content": "ENC", "summary": []}


def _probe_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: "ok",
    )


def _tool_step(*, thinking_blocks=None, reasoning_items=None, psf=None, summary=None):
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
        thinking_blocks=thinking_blocks,
        provider_specific_fields=psf,
    )
    if reasoning_items is not None:
        msg.reasoning_items = reasoning_items
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


def _run_two_steps(model, first_step):
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion",
            side_effect=[first_step, _answer_step()],
        ) as completion,
        patch.dict(
            "os.environ",
            {"AGENT_LLM_STREAMING_ENABLED": "false", "OPENAI_REASONING_SUMMARY": ""},
        ),
    ):
        agent = Agent(model=model, reasoning_effort="high")
        output = agent.run(
            "hi", context=ExecutionContext(), tools={"probe": _probe_tool()}
        )
    assert output.status.is_success()
    assert completion.call_count == 2
    return completion.call_args_list


def _prior_assistant(call):
    return [m for m in call.kwargs["messages"] if m.get("role") == "assistant"][-1]


def test_anthropic_thinking_blocks_replayed_on_the_next_step():
    calls = _run_two_steps(
        "anthropic/claude-opus-4-8",
        _tool_step(thinking_blocks=_BLOCKS, summary="I should probe."),
    )
    assert _prior_assistant(calls[1])["thinking_blocks"] == _BLOCKS


def test_replayed_blocks_keep_litellm_from_dropping_thinking(monkeypatch):
    """The regression itself: with no thinking blocks on any assistant turn,
    LiteLLM (modify_params) pops `thinking` from the request."""
    import litellm
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    monkeypatch.setattr(litellm, "modify_params", True)
    calls = _run_two_steps(
        "anthropic/claude-opus-4-8",
        _tool_step(thinking_blocks=_BLOCKS, summary="I should probe."),
    )
    second = calls[1].kwargs
    body = AnthropicConfig().transform_request(
        model="claude-opus-4-8",
        messages=second["messages"],
        optional_params={"thinking": second["thinking"], "max_tokens": 4000},
        litellm_params={},
        headers={},
    )
    assert body.get("thinking") == second["thinking"]


def test_openai_reasoning_items_replayed_and_encrypted_content_requested():
    calls = _run_two_steps("openai/gpt-5.4", _tool_step(reasoning_items=[_ITEM]))
    for call in calls:
        assert call.kwargs["model"] == "openai/responses/gpt-5.4"
        assert call.kwargs["extra_body"]["include"] == ["reasoning.encrypted_content"]
    assert _prior_assistant(calls[1])["reasoning_items"] == [_ITEM]


def test_replayed_items_reach_the_responses_input_before_their_call():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    calls = _run_two_steps("openai/gpt-5.4", _tool_step(reasoning_items=[_ITEM]))
    items, _ = (
        LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(
            calls[1].kwargs["messages"]
        )
    )
    kinds = [i["type"] for i in items]
    reasoning = items[kinds.index("reasoning")]
    assert kinds.index("reasoning") < kinds.index("function_call")
    assert reasoning["id"] == "rs_1"
    assert reasoning["encrypted_content"] == "ENC"


def test_gemini_thought_signatures_replayed():
    calls = _run_two_steps(
        "gemini/gemini-2.5-pro", _tool_step(psf={"thought_signatures": ["gsig"]})
    )
    assert _prior_assistant(calls[1])["provider_specific_fields"] == {
        "thought_signatures": ["gsig"]
    }
