"""
Execution status enum for tracking the state of any execution.
"""

from enum import Enum


class ExecutionStatus(str, Enum):
    """
    Status of an execution (Agent, Orchestration, or Workflow).

    Inherits from str for JSON serialization compatibility.

    States:
        PENDING: Execution created but not yet started
        RUNNING: Execution is currently in progress
        COMPLETED: Execution finished successfully
        FAILED: Execution encountered an error
        CANCELLED: Execution was cancelled before completion

    Example:
        >>> status = ExecutionStatus.COMPLETED
        >>> status.value
        'completed'
        >>> str(status)
        'completed'
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this status represents a terminal state (execution is done)."""
        return self in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        )

    def is_success(self) -> bool:
        """Check if this status represents successful completion."""
        return self == ExecutionStatus.COMPLETED
