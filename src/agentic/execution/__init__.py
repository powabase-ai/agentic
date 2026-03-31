"""
Execution module - shared infrastructure for all agentic executions.

This module provides the foundational types used across Agent, Orchestration,
and Workflow executions:
- ExecutionStatus: Enum for tracking execution state
- ExecutionContext: Runtime context passed during execution
- BaseOutput: Base class for all execution outputs
"""

from agentic.execution.base import BaseOutput
from agentic.execution.context import ExecutionContext, MaxDepthExceeded, TokenBudget
from agentic.execution.status import ExecutionStatus

__all__ = [
    "ExecutionStatus",
    "ExecutionContext",
    "TokenBudget",
    "MaxDepthExceeded",
    "BaseOutput",
]
