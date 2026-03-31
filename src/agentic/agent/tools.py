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


@dataclass
class KnowledgeSearchTool(ToolDefinition):
    """Tool that searches knowledge bases via the platform's retrieval pipeline.

    The search_handler is injected by the project-service at execution time
    (it wraps create_and_execute()). The core library doesn't depend on
    Flask/SQLAlchemy — the handler bridges that gap.
    """

    # Override parent's required input_schema with a default so callers can omit it.
    # __post_init__ auto-generates the schema when an empty dict is passed.
    input_schema: dict[str, Any] = field(default_factory=dict)
    knowledge_base_configs: list[dict[str, Any]] = field(default_factory=list)
    max_context_tokens: int = 8000
    include_kb_filter: bool = False
    search_handler: Callable[..., str] | None = field(default=None, repr=False)

    def __post_init__(self):
        # Auto-generate input_schema if not provided (empty dict)
        if not self.input_schema:
            props: dict[str, Any] = {
                "query": {"type": "string", "description": "Search query"},
            }
            required = ["query"]
            if self.include_kb_filter:
                kb_names = [
                    c.get("name", c.get("id", "")) for c in self.knowledge_base_configs
                ]
                props["knowledge_base_names"] = {
                    "type": "array",
                    "items": {"type": "string", "enum": kb_names},
                    "description": "Which knowledge bases to search. Omit to search all.",
                }
            self.input_schema = {
                "type": "object",
                "properties": props,
                "required": required,
            }

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        if self.search_handler is None:
            raise NotImplementedError(
                f"KnowledgeSearchTool '{self.name}' has no search_handler. "
                "The project-service must inject one at execution time."
            )
        session_history = context.session_history if context else None
        return self.search_handler(
            query=arguments["query"],
            kb_configs=self.knowledge_base_configs,
            max_tokens=self.max_context_tokens,
            session_history=session_history,
        )
