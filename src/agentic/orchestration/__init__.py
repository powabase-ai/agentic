"""
Orchestration module - multi-agent coordination.

This module provides orchestration patterns for coordinating multiple agents:
- Supervisor: A coordinator agent delegates to specialists

More strategies (Sequential, Router, Parallel) planned for future releases.
"""

from agentic.orchestration.engine import StrategyEngine, get_strategy_engine
from agentic.orchestration.orchestration import Orchestration, OrchestrationEntity
from agentic.orchestration.output import OrchestrationOutput
from agentic.orchestration.session import OrchestrationSession
from agentic.orchestration.strategies import SupervisorEngine

__all__ = [
    "Orchestration",
    "OrchestrationEntity",
    "OrchestrationOutput",
    "OrchestrationSession",
    "StrategyEngine",
    "get_strategy_engine",
    "SupervisorEngine",
]
