"""Tool definitions for agent ReAct loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import requests

from agentic.agent.tool_config import (
    CUSTOM_TOOL_TIMEOUT,
    DEFAULT_MAX_RESULT_CHARS,
    DELEGATE_MAX_STEPS,
    MAX_TOOL_OUTPUT_LENGTH,
    MCP_TOOL_TIMEOUT,
)
from agentic.knowledge.model_config import (
    DEFAULT_MAX_CONTEXT_TOKENS as _DEFAULT_MAX_CONTEXT_TOKENS,
)

if TYPE_CHECKING:
    from agentic.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


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
    max_result_chars: int | None = DEFAULT_MAX_RESULT_CHARS  # None = unlimited

    def validate_input(self, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate input before execution. Override for custom validation."""
        return True, None

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str | list[dict[str, Any]]:
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
    timeout: int = CUSTOM_TOOL_TIMEOUT
    max_output_length: int = MAX_TOOL_OUTPUT_LENGTH

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        from agentic.agent.url_validation import SSRFError, validate_url

        try:
            validate_url(self.endpoint)
        except SSRFError as e:
            return f"Error: URL blocked — {e}"

        response = requests.request(
            self.method,
            self.endpoint,
            json=arguments,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text[: self.max_output_length]


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
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS
    include_kb_filter: bool = False
    search_handler: Callable[..., str | list[dict[str, Any]]] | None = field(
        default=None, repr=False
    )

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
    ) -> str | list[dict[str, Any]]:
        if self.search_handler is None:
            raise NotImplementedError(
                f"KnowledgeSearchTool '{self.name}' has no search_handler. "
                "The project-service must inject one at execution time."
            )

        kb_configs = self.knowledge_base_configs

        # Filter by requested KB names if provided
        requested_names = arguments.get("knowledge_base_names")
        if requested_names and self.include_kb_filter:
            requested_set = set(requested_names)
            kb_configs = [c for c in kb_configs if c.get("name") in requested_set]
            if not kb_configs:
                return "No matching knowledge bases found for the requested names."

        session_history = context.session_history if context else None
        result = self.search_handler(
            query=arguments["query"],
            kb_configs=kb_configs,
            max_tokens=self.max_context_tokens,
            session_history=session_history,
        )
        # Handler may return (content, metadata) tuple with context_handler_id
        if isinstance(result, tuple) and len(result) == 2:
            content, metadata = result
            if (
                context
                and isinstance(metadata, dict)
                and metadata.get("context_handler_id")
            ):
                context.emit_event(
                    {
                        "type": "context_handler_created",
                        "context_handler_id": metadata["context_handler_id"],
                        "tool_name": self.name,
                        "query": arguments.get("query", ""),
                    }
                )
            return content
        return result


@dataclass
class McpTool(ToolDefinition):
    """Tool backed by an MCP server."""

    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""
    server_url: str = ""
    server_headers: dict[str, str] = field(default_factory=dict)
    transport: str = "http"
    mcp_tool_name: str = ""

    def execute(
        self, arguments: dict[str, Any], context: ExecutionContext | None
    ) -> str:
        from agentic.mcp.client import call_mcp_tool

        return call_mcp_tool(
            server_url=self.server_url,
            server_headers=self.server_headers,
            tool_name=self.mcp_tool_name,
            arguments=arguments,
            timeout=MCP_TOOL_TIMEOUT,
        )


@dataclass
class DelegateTool(ToolDefinition):
    """Delegates to a sub-agent, running a nested ReAct loop.

    Used by the orchestration supervisor to invoke specialist agents.
    The delegate creates a child ExecutionContext and calls agent.run().
    An optional on_run_complete hook is invoked after the sub-agent returns —
    the platform layer uses this to persist the delegated run.
    """

    # Override parent's input_schema with auto-generated default
    input_schema: dict[str, Any] = field(default_factory=dict)

    # Metadata override: delegate results can be arbitrarily large
    max_result_chars: int | None = None  # Unlimited

    agent: Any = field(
        default=None, repr=False
    )  # Agent instance (Any to avoid circular import)
    agent_tools: dict[str, ToolDefinition] = field(default_factory=dict)
    max_steps: int = DELEGATE_MAX_STEPS
    on_run_complete: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )

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
        agent_name = self.agent.name if self.agent else self.name

        context.emit_event(
            {
                "type": "delegation_started",
                "agent": agent_name,
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
                "agent": agent_name,
                "child_execution_id": child_ctx.execution_id,
                "status": output.status.value,
                "steps": output.steps,
                "usage": output.usage,
            }
        )

        if self.on_run_complete is not None:
            try:
                self.on_run_complete(
                    {
                        "agent_name": agent_name,
                        "model": self.agent.model if self.agent else None,
                        "child_execution_id": child_ctx.execution_id,
                        "orchestration_run_id": context.orchestration_run_id,
                        "task": arguments.get("task", ""),
                        "content": output.content,
                        "status": output.status,
                        "error": output.error,
                        "usage": output.usage or {},
                        "steps": output.steps,
                        "events": output.events,
                        "tool_calls": [
                            tc.to_dict() for tc in (output.tool_calls or [])
                        ],
                        "messages": output.messages or [],
                        "started_at": output.started_at,
                        "completed_at": output.completed_at,
                    }
                )
            except Exception:
                logger.exception(
                    "DelegateTool.on_run_complete hook failed for %s; continuing",
                    agent_name,
                )

        return output.content or output.error or ""
