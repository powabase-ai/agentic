"""Pydantic Message + ReasoningArtifact for the persistence boundary.

Used at four boundary categories (spec §5.6.1): output_messages writes,
load_session_history read, and the route-layer build_messages_for_llm
conversion. In-memory in agent.py the messages stay list[dict] — Pydantic is
the validation/serialization layer at the boundary, not a replacement for the
in-memory shape.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AnthropicReasoning(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    thinking_blocks: list[dict] = Field(default_factory=list)
    summary_text: str | None = None
    requested_effort: str | None = None
    output_tokens: int | None = None


class OpenAIReasoning(BaseModel):
    provider: Literal["openai"] = "openai"
    response_id: str | None = None
    encrypted_content_items: list[dict] = Field(default_factory=list)
    summary_text: str | None = None
    requested_effort: str | None = None
    reasoning_token_count: int | None = None


class GeminiReasoning(BaseModel):
    provider: Literal["gemini"] = "gemini"
    thought_signatures: list[str | None] = Field(default_factory=list)
    summary_text: str | None = None
    requested_effort: str | None = None
    thoughts_token_count: int | None = None


ReasoningArtifact = Annotated[
    AnthropicReasoning | OpenAIReasoning | GeminiReasoning,
    Field(discriminator="provider"),
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    reasoning: ReasoningArtifact | None = None
    reasoning_requested: bool = False

    def to_litellm_input(self) -> dict:
        """Produce the LiteLLM-bound shape for replay.

        Provider-specific replay fields are emitted unconditionally per the
        artifact's provider. The decision to *use* them (i.e., not strip on
        cross-provider boundary) belongs to the caller (build_messages_for_llm
        in services/session.py — see spec §5.5).
        """
        base: dict = {"role": self.role}
        if self.content is not None:
            base["content"] = self.content
        if self.tool_calls:
            base["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            base["tool_call_id"] = self.tool_call_id
        if self.role == "assistant" and self.reasoning is not None:
            r = self.reasoning
            if isinstance(r, AnthropicReasoning):
                if r.thinking_blocks:
                    base["thinking_blocks"] = r.thinking_blocks
            elif isinstance(r, OpenAIReasoning):
                if r.encrypted_content_items:
                    base["provider_specific_fields"] = {
                        "encrypted_content_items": r.encrypted_content_items
                    }
            elif isinstance(r, GeminiReasoning):
                if r.thought_signatures:
                    base["provider_specific_fields"] = {
                        "thought_signatures": r.thought_signatures
                    }
        return base
