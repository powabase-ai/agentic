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


def _content_char_len(content: Any) -> int:
    """Return approximate character length of a message content field.

    Handles both ``str`` content and multimodal ``list[dict]`` content blocks
    (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``).
    Image blocks are estimated at 1000 chars each as a rough proxy.
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    total += len(item.get("text", ""))
                elif item.get("type") == "image_url":
                    total += 1000  # rough proxy for an image block
        return total
    return len(str(content)) if content else 0


def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """Rough estimate of token count for a message list."""
    total_chars = sum(_content_char_len(m.get("content", "") or "") for m in messages)
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

    def _content_as_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content) if content else ""

    conversation_text = "\n".join(
        f"{m['role']}: {_content_as_text(m.get('content', ''))}"
        for m in to_summarize
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


_MODEL_CONTEXT_WINDOWS = {
    "claude-sonnet-4-6": 200000,
    "claude-opus-4-6": 200000,
    "claude-haiku-4-5": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1000000,
    "gpt-4.1-mini": 1000000,
    "gpt-4.1-nano": 1000000,
}
_DEFAULT_CONTEXT_WINDOW = 128000
_MAX_OUTPUT_TOKENS = 8000
_COMPACT_BUFFER = 13000


def get_context_threshold(model: str) -> int:
    for prefix, window in _MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return window - _MAX_OUTPUT_TOKENS - _COMPACT_BUFFER
    return _DEFAULT_CONTEXT_WINDOW - _MAX_OUTPUT_TOKENS - _COMPACT_BUFFER


def prune_messages(
    messages: list[dict[str, Any]], keep_last_n_turns: int = 3
) -> list[dict[str, Any]]:
    if not messages:
        return messages
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if keep_last_n_turns > 0 and len(user_indices) > keep_last_n_turns:
        protect_from = user_indices[-keep_last_n_turns]
    else:
        protect_from = len(messages)
    result = []
    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and i < protect_from:
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": "[Previous tool result removed to save context]",
                }
            )
        else:
            result.append(msg)
    return result
