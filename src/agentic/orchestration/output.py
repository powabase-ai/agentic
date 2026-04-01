"""Output types for orchestration runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic.execution.base import BaseOutput


@dataclass
class OrchestrationOutput(BaseOutput):
    """Output from an orchestration run."""

    content: str | None = None
    agent_outputs: list[Any] = field(default_factory=list)
    coordination_metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
