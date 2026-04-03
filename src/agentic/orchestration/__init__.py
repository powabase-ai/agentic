"""
Orchestration module - multi-agent coordination.

This module provides orchestration patterns for coordinating multiple agents:
- Supervisor: A coordinator agent delegates to specialists
- Sequential: Agents run in order, each receiving the previous output
- Parallel: All agents run simultaneously, results merged by LLM
"""

from agentic.orchestration.engine import StrategyEngine, get_strategy_engine
from agentic.orchestration.orchestration import Orchestration, OrchestrationEntity
from agentic.orchestration.output import OrchestrationOutput
from agentic.orchestration.session import OrchestrationSession
from agentic.orchestration.strategies import (
    ParallelEngine,
    SequentialEngine,
    SupervisorEngine,
)

__all__ = [
    "Orchestration",
    "OrchestrationEntity",
    "OrchestrationOutput",
    "OrchestrationSession",
    "ParallelEngine",
    "SequentialEngine",
    "StrategyEngine",
    "SupervisorEngine",
    "get_strategy_engine",
]
