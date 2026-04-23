"""Orchestration - multi-agent coordination patterns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic.agent.agent import Agent
    from agentic.agent.tools import ToolDefinition
    from agentic.execution.context import ExecutionContext
    from agentic.orchestration.output import OrchestrationOutput
    from agentic.orchestration.session import OrchestrationSession


@dataclass
class OrchestrationEntity:
    """An entity (agent or tool) participating in an orchestration."""

    entity_type: str  # "agent" or "tool"
    agent: Agent | None = None
    tool: ToolDefinition | None = None
    role_description: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    position: int | None = None
    agent_tools: dict[str, ToolDefinition] | None = None


class Orchestration:
    """
    Multi-agent coordination.

    Orchestration coordinates multiple agents using a strategy engine:
    - Supervisor: A coordinator agent delegates tasks to specialists

    Example:
        >>> from agentic import Agent, Orchestration
        >>>
        >>> researcher = Agent(name="researcher", ...)
        >>> writer = Agent(name="writer", ...)
        >>>
        >>> orch = Orchestration(
        ...     name="research_write",
        ...     description="Research then write",
        ...     strategy="supervisor",
        ... )
        >>> orch.add_entity(entity_type="agent", agent=researcher, role_description="...")
        >>> orch.add_entity(entity_type="agent", agent=writer, role_description="...")
        >>> output = orch.run("Write an article about AI")
    """

    def __init__(
        self,
        name: str,
        description: str,
        strategy: str = "supervisor",
        orchestrator_config: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.strategy = strategy
        self.orchestrator_config: dict[str, Any] = orchestrator_config or {}
        self.settings: dict[str, Any] = settings or {}
        self.entities: list[OrchestrationEntity] = []

    def add_entity(
        self,
        entity_type: str,
        agent: Agent | None = None,
        tool: ToolDefinition | None = None,
        role_description: str | None = None,
        config: dict[str, Any] | None = None,
        position: int | None = None,
        agent_tools: dict[str, ToolDefinition] | None = None,
    ) -> None:
        """Add an agent or tool entity to this orchestration."""
        entity = OrchestrationEntity(
            entity_type=entity_type,
            agent=agent,
            tool=tool,
            role_description=role_description,
            config=config or {},
            position=position,
            agent_tools=agent_tools,
        )
        self.entities.append(entity)

    def run(
        self,
        input: str,
        session: OrchestrationSession | None = None,
        context: ExecutionContext | None = None,
        history: list[dict] | None = None,
        on_delegate_complete: Callable[[dict], None] | None = None,
    ) -> OrchestrationOutput:
        """Execute the orchestration with the given input."""
        from agentic.orchestration.engine import get_strategy_engine

        engine = get_strategy_engine(self.strategy)
        return engine.execute(
            self,
            input,
            session,
            context,
            history=history,
            on_delegate_complete=on_delegate_complete,
        )

    async def arun(
        self,
        input: str,
        session: OrchestrationSession | None = None,
        context: ExecutionContext | None = None,
    ) -> OrchestrationOutput:
        """Execute the orchestration asynchronously. Not yet implemented."""
        raise NotImplementedError("Orchestration.arun() is not yet implemented.")

    def __repr__(self) -> str:
        return (
            f"Orchestration(name={self.name!r}, strategy={self.strategy!r}, "
            f"entities={len(self.entities)})"
        )
