"""
Base output class - shared base for all execution outputs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from agentic.execution.status import ExecutionStatus


@dataclass
class BaseOutput:
    """
    Base class for all execution outputs.

    Provides common fields shared across AgentOutput, OrchestrationOutput,
    and WorkflowOutput. All output types inherit from this class.

    Attributes:
        execution_id: Unique identifier for this execution
        status: Current status of the execution
        started_at: When the execution started
        completed_at: When the execution finished (None if still running)
        error: Error message if status is FAILED

    Example:
        >>> output = BaseOutput(
        ...     execution_id="exec-123",
        ...     status=ExecutionStatus.COMPLETED,
        ...     started_at=datetime.now(),
        ... )
        >>> output.is_success()
        True
    """

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None

    def is_success(self) -> bool:
        """Check if this execution completed successfully."""
        return self.status == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if this execution failed."""
        return self.status == ExecutionStatus.FAILED

    def is_done(self) -> bool:
        """Check if this execution is in a terminal state."""
        return self.status.is_terminal()

    def duration_seconds(self) -> float | None:
        """
        Get the execution duration in seconds.

        Returns:
            Duration in seconds, or None if not yet completed
        """
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of this output
        """
        data = asdict(self)
        # Convert datetime to ISO format strings
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        # Convert enum to string
        data["status"] = self.status.value
        return data
