"""Strategy engine base and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic.execution.context import ExecutionContext
    from agentic.orchestration.orchestration import Orchestration
    from agentic.orchestration.output import OrchestrationOutput


class StrategyEngine:
    """Base class for orchestration strategies."""

    def execute(
        self,
        orchestration: Orchestration,
        input: str,
        session: Any,
        context: ExecutionContext | None,
        *,
        history: list[dict] | None = None,
    ) -> OrchestrationOutput:
        raise NotImplementedError


def get_strategy_engine(strategy: str) -> StrategyEngine:
    from agentic.orchestration.strategies import (
        ParallelEngine,
        SequentialEngine,
        SupervisorEngine,
    )

    engines: dict[str, StrategyEngine] = {
        "supervisor": SupervisorEngine(),
        "sequential": SequentialEngine(),
        "parallel": ParallelEngine(),
    }
    if strategy not in engines:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(engines.keys())}"
        )
    return engines[strategy]
