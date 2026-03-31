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

    # Execution metadata (fail-closed defaults)
    is_concurrency_safe: bool = False  # Safe to run in parallel?
    is_read_only: bool = False  # Does not modify state?
    is_destructive: bool = False  # Irreversible action?
    max_result_chars: int | None = 50000  # None = unlimited

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

    # Metadata overrides: knowledge search is safe to parallelise and read-only
    is_concurrency_safe: bool = True
    is_read_only: bool = True
    max_result_chars: int | None = None  # Unlimited

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


@dataclass
class DelegateTool(ToolDefinition):
    """Delegates to a sub-agent, running a nested ReAct loop.

    Used by the orchestration supervisor to invoke specialist agents.
    The delegate creates a child ExecutionContext and calls agent.run().
    """

    # Override parent's input_schema with auto-generated default
    input_schema: dict[str, Any] = field(default_factory=dict)

    # Metadata override: delegate results can be arbitrarily large
    max_result_chars: int | None = None  # Unlimited

    agent: Any = field(
        default=None, repr=False
    )  # Agent instance (Any to avoid circular import)
    agent_tools: dict[str, ToolDefinition] = field(default_factory=dict)
    max_steps: int = 10

    def __post_init__(self):
        if not self.input_schema:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What you need this agent to do",
                    },
                },
                "required": ["task"],
            }

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        if context is None:
            from agentic.execution.context import ExecutionContext as EC

            context = EC()

        # Create child context (raises MaxDepthExceeded if at limit)
        child_ctx = context.child_context()

        context.emit_event(
            {
                "type": "delegation_started",
                "agent": self.agent.name if self.agent else self.name,
                "task": arguments.get("task", ""),
                "child_execution_id": child_ctx.execution_id,
            }
        )

        output = self.agent.run(
            input=arguments["task"],
            context=child_ctx,
            tools=self.agent_tools if self.agent_tools else None,
            max_steps=self.max_steps,
        )

        context.emit_event(
            {
                "type": "delegation_completed",
                "agent": self.agent.name if self.agent else self.name,
                "child_execution_id": child_ctx.execution_id,
                "status": output.status.value,
                "steps": output.steps,
                "usage": output.usage,
            }
        )

        return output.content or output.error or ""
