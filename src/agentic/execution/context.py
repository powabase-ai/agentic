"""
Execution context - runtime context passed during execution.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ExecutionContext:
    """
    Runtime context for an execution.

    ExecutionContext carries information about the current execution that may
    be needed by various components during processing. It is created at the
    start of each execution and passed through the execution pipeline.

    Attributes:
        execution_id: Unique identifier for this execution
        session_id: Optional session this execution belongs to
        user_id: Optional user who initiated this execution
        metadata: Additional key-value metadata for this execution

    Example:
        >>> ctx = ExecutionContext(
        ...     execution_id="exec-123",
        ...     session_id="sess-456",
        ...     user_id="user-789",
        ... )
        >>> ctx.execution_id
        'exec-123'
    """

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kwargs: Any) -> "ExecutionContext":
        """
        Create a new context with additional metadata.

        Args:
            **kwargs: Key-value pairs to add to metadata

        Returns:
            New ExecutionContext with merged metadata
        """
        new_metadata = {**self.metadata, **kwargs}
        return ExecutionContext(
            execution_id=self.execution_id,
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=new_metadata,
        )
