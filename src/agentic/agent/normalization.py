"""Message normalization before LLM API calls."""

from __future__ import annotations

from typing import Any

_STANDARD_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def normalize_messages(
    messages: list[dict[str, Any]],
    available_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare messages for LLM API call."""
    valid_tool_call_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for msg in messages:
        clean: dict[str, Any] = {}
        is_injected = msg.get("_injected", False)

        for key in _STANDARD_FIELDS:
            if key in msg:
                clean[key] = msg[key]

        if is_injected and clean.get("role") == "system" and clean.get("content"):
            clean["content"] = (
                f"<system-context>\n{clean['content']}\n</system-context>"
            )

        if "tool_calls" in clean and available_tool_names is not None:
            clean["tool_calls"] = [
                tc
                for tc in clean["tool_calls"]
                if tc.get("function", {}).get("name") in available_tool_names
            ]
            if not clean["tool_calls"]:
                del clean["tool_calls"]

        if "tool_calls" in clean:
            for tc in clean["tool_calls"]:
                valid_tool_call_ids.add(tc["id"])

        normalized.append(clean)

    result: list[dict[str, Any]] = []
    for msg in normalized:
        if msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in valid_tool_call_ids:
                continue
        result.append(msg)

    return result
