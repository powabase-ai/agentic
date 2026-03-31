from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class McpToolInfo:
    """Information about a tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only_hint: bool = False
    destructive_hint: bool = False
    open_world_hint: bool = False
