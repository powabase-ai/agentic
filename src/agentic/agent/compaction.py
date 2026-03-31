"""Context compaction — summarize conversation history when context is full."""

from __future__ import annotations

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_COMPACTION_MODEL = "gpt-4.1-nano"

COMPACTION_PROMPT = """Summarize the following conversation history concisely.
Preserve key facts, decisions, and context that would be needed to continue the conversation.
Do not include pleasantries or filler. Be factual and brief."""


def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """Rough estimate of token count for a message list."""
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return total_chars // CHARS_PER_TOKEN


def compact_messages(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_COMPACTION_MODEL,
    keep_last_n: int = 2,
) -> list[dict[str, Any]]:
    """Compact message history by summarizing older messages.

    Preserves: system message (first), last N user/assistant turns.
    Summarizes: everything in between.
    """
    if len(messages) <= keep_last_n + 1:
        return messages

    system_msg = None
    rest = messages
    if messages and messages[0]["role"] == "system":
        system_msg = messages[0]
        rest = messages[1:]

    if len(rest) <= keep_last_n:
        return messages

    to_summarize = rest[:-keep_last_n]
    to_keep = rest[-keep_last_n:]

    conversation_text = "\n".join(
        f"{m['role']}: {m.get('content', '')}" for m in to_summarize
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": COMPACTION_PROMPT},
                {"role": "user", "content": conversation_text},
            ],
            stream=False,
        )
        summary = response.choices[0].message.content
    except Exception as e:
        logger.warning("Compaction failed, keeping original messages: %s", e)
        return messages

    result = []
    if system_msg:
        result.append(system_msg)
    result.append(
        {
            "role": "assistant",
            "content": f"[Conversation summary: {summary}]",
        }
    )
    result.extend(to_keep)
    return result
