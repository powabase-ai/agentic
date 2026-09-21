"""Prompt-cache helpers: deterministic tool ordering and explicit breakpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from agentic.agent.tools import ToolDefinition

_TYPE_PRIORITY = {
    "BuiltinTool": 0,
    "KnowledgeSearchTool": 1,
    "CustomTool": 2,
    "McpTool": 3,
    "DelegateTool": 4,
}

# Providers that serve Claude and turn litellm's `cache_control` into their
# own breakpoint syntax (Bedrock Converse emits `cachePoint` blocks).
_EXPLICIT_BREAKPOINT_PROVIDERS = frozenset({"anthropic", "vertex_ai", "bedrock"})
_EPHEMERAL = {"type": "ephemeral"}


def sort_tools_for_cache(tools: dict[str, ToolDefinition]) -> list[ToolDefinition]:
    """Sort tools deterministically for prompt cache stability."""

    def sort_key(tool: ToolDefinition) -> tuple[int, str]:
        type_name = type(tool).__name__
        priority = _TYPE_PRIORITY.get(type_name, 99)
        return (priority, tool.name)

    return sorted(tools.values(), key=sort_key)


def uses_explicit_cache_breakpoints(model: str) -> bool:
    """True when ``model`` is Claude on a provider that caches only up to
    explicit ``cache_control`` breakpoints.

    OpenAI, DeepSeek, Gemini and the OpenRouter routes cache automatically;
    the key buys nothing there and some endpoints reject unknown fields.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return False
    return provider in _EXPLICIT_BREAKPOINT_PROVIDERS and "claude" in model.lower()


def add_cache_breakpoints(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return copies of ``messages`` and ``tools`` carrying cache breakpoints.

    The provider caches the request prefix — tools, then system, then
    messages — up to each breakpoint, and allows at most four. This spends
    three:

    1. the last tool definition — tools open the prefix;
    2. the leading system message — large and fixed for the whole run;
    3. the last message — it moves every step, so each call reads the
       history the previous call wrote.

    The inputs are never mutated. A breakpoint belongs to one request: left in
    stored history it would still be there next step, and the count would
    grow past the limit. Models without explicit breakpoints get the inputs
    back unchanged.
    """
    if not uses_explicit_cache_breakpoints(model):
        return messages, tools

    if tools:
        tools = [*tools[:-1], {**tools[-1], "cache_control": dict(_EPHEMERAL)}]

    marked = list(messages)
    targets = set()
    if marked and marked[0].get("role") == "system":
        targets.add(0)
    if marked:
        targets.add(len(marked) - 1)
    for i in targets:
        marked[i] = _with_breakpoint(marked[i])
    return marked, tools


def _with_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``message`` with a breakpoint where litellm picks it up.

    litellm moves a message-level ``cache_control`` onto the provider block
    for string content and for tool results; list content needs it on the
    last block.
    """
    content = message.get("content")
    if message.get("role") == "tool" or not isinstance(content, list) or not content:
        return {**message, "cache_control": dict(_EPHEMERAL)}
    last = {**content[-1], "cache_control": dict(_EPHEMERAL)}
    return {**message, "content": [*content[:-1], last]}
