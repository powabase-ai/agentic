"""
Agent output - result of a single agent execution.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentic.execution.base import BaseOutput
from agentic.execution.status import ExecutionStatus

if TYPE_CHECKING:
    from agentic.agent.message import ReasoningArtifact


@dataclass
class ToolCallRecord:
    """Record of a single tool call within a ReAct loop step."""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    result: str | list[dict[str, Any]]
    duration_ms: int
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "step": self.step,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": self.duration_ms,
        }
        if self.usage is not None:
            d["usage"] = self.usage
        return d


@dataclass
class AgentOutput(BaseOutput):
    """
    Result of a single Agent.run() call.

    AgentOutput captures everything that happened during an agent execution,
    including the response content, all messages exchanged, and token usage.

    Attributes:
        execution_id: Unique identifier for this execution (inherited)
        status: Execution status (inherited)
        started_at: When execution started (inherited)
        completed_at: When execution finished (inherited)
        error: Error message if failed (inherited)
        content: The agent's response content (text or structured)
        messages: All messages exchanged during this execution
        usage: Token usage statistics from the LLM

    Example:
        >>> output = agent.run("What is 2+2?")
        >>> print(output.content)
        "2 + 2 = 4"
        >>> print(output.usage)
        {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
    """

    content: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    # ReAct loop fields
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Reasoning artifact (Anthropic thinking_blocks, OpenAI encrypted_content,
    # Gemini thought_signatures) extracted from the final assistant message,
    # plus a flag for whether reasoning was actually requested for this run.
    reasoning_artifact: "ReasoningArtifact | None" = None
    reasoning_requested: bool = False

    def get_content(self) -> str:
        """
        Get the content, returning empty string if None.

        Returns:
            The response content or empty string
        """
        return self.content or ""

    def total_tokens(self) -> int | None:
        """
        Get total tokens used in this execution.

        Returns:
            Total token count, or None if usage not available
        """
        if self.usage is None:
            return None
        return self.usage.get("total_tokens")

    @classmethod
    def from_error(cls, execution_id: str, error: str) -> "AgentOutput":
        """
        Create an AgentOutput representing a failed execution.

        Args:
            execution_id: The execution ID
            error: Error message

        Returns:
            AgentOutput with FAILED status
        """
        from datetime import datetime

        return cls(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            error=error,
        )
