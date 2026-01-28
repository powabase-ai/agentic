"""
Orchestration session - state container for orchestration execution.

Status: Not yet implemented.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentic.orchestration.output import OrchestrationOutput


@dataclass
class OrchestrationSession:
    """
    Container for orchestration execution history and state.

    OrchestrationSession maintains the history of all orchestration executions,
    coordination state, and individual agent contexts within the orchestration.

    Status: Not yet implemented.

    Planned Attributes:
        session_id: Unique identifier for this session
        orchestration_name: Name of the orchestration
        outputs: List of all OrchestrationOutputs from this session
        coordination_state: State for coordination (e.g., supervisor plans)
        agent_contexts: Per-agent state within this orchestration

    See Also:
        - docs/CONCEPTS.md for detailed documentation
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    orchestration_name: str | None = None
    outputs: list[OrchestrationOutput] = field(default_factory=list)
    coordination_state: dict[str, Any] = field(default_factory=dict)
    agent_contexts: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
