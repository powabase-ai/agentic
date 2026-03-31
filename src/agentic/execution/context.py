"""Execution context for agent and orchestration runs."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class MaxDepthExceeded(Exception):
    """Raised when orchestration nesting depth exceeds the limit."""


@dataclass
class TokenBudget:
    """Token budget for a run."""

    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def exceeded(self) -> bool:
        return self.used_tokens >= self.max_tokens

    def consume(self, tokens: int) -> None:
        self.used_tokens += tokens

    def allocate_child(self) -> TokenBudget:
        """Give a child the remaining budget."""
        return TokenBudget(max_tokens=self.remaining)


@dataclass
class ExecutionContext:
    """Runtime context for agent and orchestration executions."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Orchestration hierarchy
    parent_run_id: str | None = None
    orchestration_run_id: str | None = None

    # Budget
    budget: TokenBudget | None = None

    # Depth control
    depth: int = 0
    max_depth: int = 3

    # Event persistence callback
    on_event: Callable[[dict[str, Any]], None] | None = None

    # Session history for tools that need conversation context (e.g., KB search)
    session_history: list[dict[str, Any]] | None = None

    # Internal event counter
    _event_seq: int = field(default=0, repr=False)

    def child_context(self) -> ExecutionContext:
        """Create a context for a sub-agent execution."""
        if self.depth >= self.max_depth:
            raise MaxDepthExceeded(f"Max orchestration depth {self.max_depth} reached")
        return ExecutionContext(
            execution_id=str(uuid.uuid4()),
            session_id=self.session_id,
            user_id=self.user_id,
            parent_run_id=self.execution_id,
            orchestration_run_id=self.orchestration_run_id,
            budget=self.budget.allocate_child() if self.budget else None,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            on_event=self.on_event,
            session_history=self.session_history,
            metadata=self.metadata,
        )

    def emit_event(self, event: dict[str, Any]) -> None:
        """Emit an event via the on_event callback if set."""
        self._event_seq += 1
        event["seq"] = self._event_seq
        event.setdefault("ts", datetime.now(UTC).isoformat())
        if self.on_event is not None:
            self.on_event(event)

    def with_metadata(self, **kwargs: Any) -> ExecutionContext:
        """Create a new context with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return ExecutionContext(
            execution_id=self.execution_id,
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=new_metadata,
            parent_run_id=self.parent_run_id,
            orchestration_run_id=self.orchestration_run_id,
            budget=self.budget,
            depth=self.depth,
            max_depth=self.max_depth,
            on_event=self.on_event,
            session_history=self.session_history,
        )
