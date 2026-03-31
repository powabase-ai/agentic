"""Prompt cache-aware tool ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.agent.tools import ToolDefinition

_TYPE_PRIORITY = {
    "BuiltinTool": 0,
    "KnowledgeSearchTool": 1,
    "CustomTool": 2,
    "McpTool": 3,
    "DelegateTool": 4,
}


def sort_tools_for_cache(tools: dict[str, ToolDefinition]) -> list[ToolDefinition]:
    """Sort tools deterministically for prompt cache stability."""

    def sort_key(tool: ToolDefinition) -> tuple[int, str]:
        type_name = type(tool).__name__
        priority = _TYPE_PRIORITY.get(type_name, 99)
        return (priority, tool.name)

    return sorted(tools.values(), key=sort_key)
