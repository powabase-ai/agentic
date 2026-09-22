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
    # Responses API reasoning items ({id, type, encrypted_content, summary}).
    # Rows persisted before this field existed carry `encrypted_content_items`
    # instead (always empty); the default `extra="ignore"` drops it on load.
    reasoning_items: list[dict] = Field(default_factory=list)
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

# provider_specific_fields entries that carry reasoning replay. The second is
# the pre-rename OpenAI key, which old session history may still hold.
_REPLAY_PSF_KEYS = ("thought_signatures", "encrypted_content_items")


def reasoning_replay_fields(reasoning: ReasoningArtifact) -> dict:
    """The LiteLLM message keys that hand ``reasoning`` back to its provider.

    Anthropic reads ``thinking_blocks``; LiteLLM's OpenAI Responses bridge
    reads a top-level ``reasoning_items``; Gemini reads
    ``provider_specific_fields.thought_signatures``. Empty when the artifact
    carries nothing to replay. The lists are copies, so a request built from
    them cannot reach back into the artifact the host persists.
    """
    if isinstance(reasoning, AnthropicReasoning):
        if reasoning.thinking_blocks:
            return {"thinking_blocks": [dict(b) for b in reasoning.thinking_blocks]}
    elif isinstance(reasoning, OpenAIReasoning):
        if reasoning.reasoning_items:
            return {"reasoning_items": [dict(i) for i in reasoning.reasoning_items]}
    elif isinstance(reasoning, GeminiReasoning):
        if reasoning.thought_signatures:
            return {
                "provider_specific_fields": {
                    "thought_signatures": list(reasoning.thought_signatures)
                }
            }
    return {}


def drop_reasoning_replay_fields(message: dict) -> dict:
    """Copy of ``message`` without any provider's reasoning replay keys.

    For a request going to a different provider than the one that reasoned:
    it cannot verify the reasoning, and LiteLLM converts some of it into
    malformed input (Claude thinking blocks become Gemini thought parts).
    ``reasoning``, the host-facing record, is kept.
    """
    out = {
        k: v
        for k, v in message.items()
        if k not in ("thinking_blocks", "reasoning_items")
    }
    psf = out.get("provider_specific_fields")
    if isinstance(psf, dict):
        kept = {k: v for k, v in psf.items() if k not in _REPLAY_PSF_KEYS}
        if kept:
            out["provider_specific_fields"] = kept
        else:
            del out["provider_specific_fields"]
    return out


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    reasoning: ReasoningArtifact | None = None
    reasoning_requested: bool = False

    def to_litellm_input(self) -> dict:
        """Produce the LiteLLM-bound shape for replay across turns.

        Anthropic thinking blocks and Gemini thought signatures are emitted
        per the artifact's provider; the decision to *use* them (i.e., not
        strip them at a cross-provider boundary) belongs to the caller.
        OpenAI reasoning items are not emitted: the Responses API discards
        reasoning from turns before the latest user message and binds
        encrypted content to the organization that produced it, so replaying
        one across turns can only cost a rejected request.
        """
        base: dict = {"role": self.role}
        if self.content is not None:
            base["content"] = self.content
        if self.tool_calls:
            base["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            base["tool_call_id"] = self.tool_call_id
        if self.role == "assistant" and self.reasoning is not None:
            replay = reasoning_replay_fields(self.reasoning)
            replay.pop("reasoning_items", None)
            base.update(replay)
        return base
