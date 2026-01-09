"""
Orchestration output - result of an orchestration execution.

Status: Not yet implemented.
"""

from dataclasses import dataclass
from typing import Any

from agentic.execution.base import BaseOutput


@dataclass
class OrchestrationOutput(BaseOutput):
    """
    Result of a single Orchestration.run() call.
    
    OrchestrationOutput captures the combined result of coordinating
    multiple agents, including individual agent outputs and coordination
    metadata.
    
    Status: Not yet implemented.
    
    Planned Attributes:
        content: The final combined output content
        agent_outputs: Individual outputs from each agent
        coordination_metadata: Information about how agents were coordinated
    
    See Also:
        - docs/CONCEPTS.md for detailed documentation
    """
    
    content: str | None = None
    agent_outputs: list[Any] | None = None
    coordination_metadata: dict[str, Any] | None = None

