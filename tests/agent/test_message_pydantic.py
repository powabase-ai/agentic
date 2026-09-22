"""Tests for Pydantic Message and ReasoningArtifact discriminated union."""

import pytest

from agentic.agent.message import (
    AnthropicReasoning,
    GeminiReasoning,
    Message,
    OpenAIReasoning,
    drop_reasoning_replay_fields,
    reasoning_replay_fields,
)


def test_anthropic_reasoning_serialization_round_trip():
    artifact = AnthropicReasoning(
        thinking_blocks=[
            {"type": "thinking", "thinking": "step 1", "signature": "abc"}
        ],
        summary_text="summary",
        requested_effort="medium",
        output_tokens=1234,
    )
    dumped = artifact.model_dump(exclude_none=True)
    assert dumped["provider"] == "anthropic"
    parsed = AnthropicReasoning.model_validate(dumped)
    assert parsed == artifact


def test_openai_reasoning_serialization_round_trip():
    artifact = OpenAIReasoning(
        response_id="resp_xyz",
        reasoning_items=[
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "...",
                "summary": [],
            }
        ],
        summary_text="summary",
        requested_effort="high",
        reasoning_token_count=500,
    )
    dumped = artifact.model_dump(exclude_none=True)
    parsed = OpenAIReasoning.model_validate(dumped)
    assert parsed == artifact


def test_gemini_reasoning_serialization_round_trip():
    artifact = GeminiReasoning(
        thought_signatures=["sig1", "sig2"],
        summary_text="summary",
        requested_effort="low",
        thoughts_token_count=300,
    )
    dumped = artifact.model_dump(exclude_none=True)
    parsed = GeminiReasoning.model_validate(dumped)
    assert parsed == artifact


def test_message_with_anthropic_reasoning_round_trip():
    msg = Message(
        role="assistant",
        content="The answer is 42.",
        tool_calls=None,
        reasoning=AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "let me think"}],
            summary_text="Considered options and chose 42.",
            requested_effort="medium",
            output_tokens=100,
        ),
        reasoning_requested=True,
    )
    dumped = msg.model_dump(exclude_none=True)
    assert dumped["reasoning"]["provider"] == "anthropic"
    parsed = Message.model_validate(dumped)
    assert isinstance(parsed.reasoning, AnthropicReasoning)
    assert parsed == msg


def test_message_discriminator_routes_correctly():
    """Pydantic v2 discriminator picks the right class by `provider` field."""
    raw = {
        "role": "assistant",
        "content": "x",
        "reasoning": {
            "provider": "openai",
            "encrypted_content_items": [],
            "summary_text": "y",
        },
        "reasoning_requested": True,
    }
    msg = Message.model_validate(raw)
    assert isinstance(msg.reasoning, OpenAIReasoning)


def test_message_without_reasoning_defaults_correctly():
    msg = Message(role="user", content="hello")
    assert msg.reasoning is None
    assert msg.reasoning_requested is False


def test_message_to_litellm_input_anthropic_emits_thinking_blocks():
    msg = Message(
        role="assistant",
        content="answer",
        reasoning=AnthropicReasoning(
            thinking_blocks=[{"type": "thinking", "thinking": "x", "signature": "s"}],
            summary_text="s",
        ),
    )
    out = msg.to_litellm_input()
    assert out["thinking_blocks"] == [
        {"type": "thinking", "thinking": "x", "signature": "s"}
    ]


def test_message_to_litellm_input_omits_openai_reasoning_items():
    """Host session history is rebuilt through this, and an OpenAI reasoning
    item from an earlier turn can only cost a rejected request."""
    item = {"id": "rs_1", "type": "reasoning", "encrypted_content": "e", "summary": []}
    msg = Message(
        role="assistant",
        content="answer",
        reasoning=OpenAIReasoning(reasoning_items=[item]),
    )
    out = msg.to_litellm_input()
    assert "reasoning_items" not in out
    assert "provider_specific_fields" not in out
    assert out == {"role": "assistant", "content": "answer"}


def test_message_to_litellm_input_gemini_emits_thought_signatures():
    msg = Message(
        role="assistant",
        content="answer",
        reasoning=GeminiReasoning(thought_signatures=["sig1"]),
    )
    out = msg.to_litellm_input()
    assert out["provider_specific_fields"] == {"thought_signatures": ["sig1"]}


def test_message_to_litellm_input_no_reasoning_omits_replay_fields():
    msg = Message(role="assistant", content="answer")
    out = msg.to_litellm_input()
    assert "thinking_blocks" not in out
    assert "provider_specific_fields" not in out
    assert out == {"role": "assistant", "content": "answer"}


def test_message_to_litellm_input_preserves_tool_calls():
    msg = Message(
        role="assistant",
        content="thinking out loud",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }
        ],
    )
    out = msg.to_litellm_input()
    assert out["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "x", "arguments": "{}"},
        }
    ]


def test_message_to_litellm_input_preserves_tool_call_id():
    msg = Message(role="tool", content="result", tool_call_id="call_1")
    out = msg.to_litellm_input()
    assert out["tool_call_id"] == "call_1"


def test_legacy_openai_row_still_validates():
    """Rows persisted before the rename carry `encrypted_content_items` (always
    empty); they must still load, with nothing to replay."""
    raw = {
        "role": "assistant",
        "content": "x",
        "reasoning": {
            "provider": "openai",
            "response_id": "resp_1",
            "encrypted_content_items": [],
        },
    }
    msg = Message.model_validate(raw)
    assert isinstance(msg.reasoning, OpenAIReasoning)
    assert msg.reasoning.reasoning_items == []
    assert "reasoning_items" not in msg.to_litellm_input()


def test_replay_fields_anthropic():
    blocks = [{"type": "thinking", "thinking": "x", "signature": "s"}]
    assert reasoning_replay_fields(AnthropicReasoning(thinking_blocks=blocks)) == {
        "thinking_blocks": blocks
    }


def test_replay_fields_openai():
    item = {"id": "rs_1", "type": "reasoning", "encrypted_content": "e", "summary": []}
    assert reasoning_replay_fields(OpenAIReasoning(reasoning_items=[item])) == {
        "reasoning_items": [item]
    }


def test_replay_fields_gemini():
    assert reasoning_replay_fields(GeminiReasoning(thought_signatures=["sig"])) == {
        "provider_specific_fields": {"thought_signatures": ["sig"]}
    }


@pytest.mark.parametrize(
    "artifact",
    [
        AnthropicReasoning(summary_text="s"),
        OpenAIReasoning(summary_text="s"),
        GeminiReasoning(summary_text="s"),
    ],
)
def test_replay_fields_empty_when_nothing_to_replay(artifact):
    assert reasoning_replay_fields(artifact) == {}


def test_replay_fields_are_copies():
    """The loop hands these lists to LiteLLM; an edit there must not reach the
    artifact the host persists."""
    blocks = [{"type": "thinking", "thinking": "x", "signature": "s"}]
    artifact = AnthropicReasoning(thinking_blocks=blocks)
    out = reasoning_replay_fields(artifact)
    out["thinking_blocks"][0]["thinking"] = "edited"
    out["thinking_blocks"].append({"type": "thinking"})
    assert artifact.thinking_blocks == [
        {"type": "thinking", "thinking": "x", "signature": "s"}
    ]


def test_drop_replay_fields_removes_every_provider_key():
    msg = {
        "role": "assistant",
        "content": "a",
        "reasoning": {"provider": "anthropic"},
        "thinking_blocks": [{"type": "thinking"}],
        "reasoning_items": [{"id": "rs_1"}],
        "provider_specific_fields": {
            "thought_signatures": ["s"],
            "encrypted_content_items": [],
            "other": 1,
        },
    }
    out = drop_reasoning_replay_fields(msg)
    assert out == {
        "role": "assistant",
        "content": "a",
        "reasoning": {"provider": "anthropic"},
        "provider_specific_fields": {"other": 1},
    }
    assert "thinking_blocks" in msg  # input untouched


def test_drop_replay_fields_removes_an_emptied_provider_dict():
    out = drop_reasoning_replay_fields(
        {"role": "assistant", "provider_specific_fields": {"thought_signatures": ["s"]}}
    )
    assert out == {"role": "assistant"}
