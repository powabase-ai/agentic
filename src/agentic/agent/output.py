"""
Agent output - result of a single agent execution.
"""

from dataclasses import dataclass, field
from typing import Any

from agentic.execution.base import BaseOutput
from agentic.execution.status import ExecutionStatus


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
