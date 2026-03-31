"""Tool definitions for agent ReAct loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from agentic.execution.context import ExecutionContext

MAX_TOOL_OUTPUT_LENGTH = 10000


@dataclass
class ToolDefinition:
    """Base class for all tool types."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        raise NotImplementedError(f"Tool {self.name} does not implement execute()")

    def to_function_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format for LiteLLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class BuiltinTool(ToolDefinition):
    """Platform-provided tool. Execution logic is in-process."""

    handler: Callable[[dict[str, Any], ExecutionContext | None], str] = field(
        default=None, repr=False
    )

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        return self.handler(arguments, context)


@dataclass
class CustomTool(ToolDefinition):
    """Client-defined tool. Calls an HTTP endpoint."""

    endpoint: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        response = requests.request(
            self.method,
            self.endpoint,
            json=arguments,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text[:MAX_TOOL_OUTPUT_LENGTH]
