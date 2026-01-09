"""
Orchestration module - multi-agent coordination.

This module provides orchestration patterns for coordinating multiple agents:
- Sequential: Agents run one after another
- Supervisor: A coordinator agent delegates to specialists
- Router: Route to a single best agent based on input
- Parallel: Multiple agents run simultaneously

Status: Not yet implemented. See docs/CONCEPTS.md for planned features.
"""

from agentic.orchestration.orchestration import Orchestration
from agentic.orchestration.output import OrchestrationOutput
from agentic.orchestration.session import OrchestrationSession

__all__ = [
    "Orchestration",
    "OrchestrationOutput",
    "OrchestrationSession",
]

