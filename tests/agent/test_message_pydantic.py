"""Tests for Pydantic Message and ReasoningArtifact discriminated union."""

from agentic.agent.message import (
    AnthropicReasoning,
    GeminiReasoning,
    Message,
    OpenAIReasoning,
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
        encrypted_content_items=[{"type": "reasoning", "encrypted_content": "..."}],
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


def test_message_to_litellm_input_openai_emits_provider_specific_fields():
    msg = Message(
        role="assistant",
        content="answer",
        reasoning=OpenAIReasoning(
            encrypted_content_items=[{"type": "reasoning", "encrypted_content": "e"}],
        ),
    )
    out = msg.to_litellm_input()
    assert out["provider_specific_fields"] == {
        "encrypted_content_items": [{"type": "reasoning", "encrypted_content": "e"}]
    }


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
