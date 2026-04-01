"""Orchestration strategy implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agentic.agent.agent import Agent
from agentic.agent.tools import DelegateTool, ToolDefinition
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus
from agentic.orchestration.engine import StrategyEngine
from agentic.orchestration.output import OrchestrationOutput


def _build_orchestrator_prompt(orchestration, entities) -> str:
    """Build the supervisor agent's system prompt."""
    agent_descriptions = []
    for entity in entities:
        if entity.entity_type == "agent" and entity.agent:
            name = entity.agent.name or "unnamed"
            desc = entity.role_description or (entity.agent.system_prompt or "")[:200]
            agent_descriptions.append(f"- delegate_to_{name}: {desc}")

    agent_list = (
        "\n".join(agent_descriptions) if agent_descriptions else "(no agents available)"
    )
    additional = orchestration.orchestrator_config.get("additional_instructions", "")

    return f"""You are an orchestrator for: {orchestration.description}

You have the following specialist agents available as tools:
{agent_list}

Your job is to:
1. Understand the user's request
2. Delegate to the appropriate specialist(s)
3. Synthesize their responses into a coherent answer

{additional}""".strip()


def _build_delegate_tools(entities) -> dict[str, DelegateTool]:
    """Build delegate tools from agent entities."""
    tools: dict[str, DelegateTool] = {}
    for entity in entities:
        if entity.entity_type == "agent" and entity.agent:
            name = entity.agent.name or "unnamed"
            tool_name = f"delegate_to_{name}"
            tools[tool_name] = DelegateTool(
                name=tool_name,
                description=entity.role_description or f"Delegate to {name}",
                agent=entity.agent,
                agent_tools=entity.agent_tools or {},
                max_steps=entity.config.get("max_steps", 10),
            )
    return tools


class SupervisorEngine(StrategyEngine):
    """LLM-driven supervisor that delegates to specialist agents."""

    def execute(
        self,
        orchestration,
        input: str,
        session: Any,
        context: ExecutionContext | None,
    ) -> OrchestrationOutput:
        if context is None:
            context = ExecutionContext()

        output = OrchestrationOutput(execution_id=context.execution_id)
        output.status = ExecutionStatus.RUNNING
        output.started_at = datetime.now(UTC)

        entities = orchestration.entities
        agent_entities = [e for e in entities if e.entity_type == "agent"]

        if not agent_entities:
            output.status = ExecutionStatus.FAILED
            output.error = "No agent entities in orchestration"
            output.completed_at = datetime.now(UTC)
            return output

        context.emit_event(
            {
                "type": "orchestration_started",
                "strategy": "supervisor",
                "name": orchestration.name,
            }
        )

        try:
            # Build delegate tools (one per agent entity)
            delegate_tools: dict[str, ToolDefinition] = _build_delegate_tools(entities)

            # Also include any tool entities directly
            for entity in entities:
                if entity.entity_type == "tool" and entity.tool:
                    delegate_tools[entity.tool.name] = entity.tool

            # Create the orchestrator agent
            orchestrator = Agent(
                model=orchestration.settings.get("model", "gpt-4.1-mini"),
                system_prompt=_build_orchestrator_prompt(orchestration, entities),
                name=f"{orchestration.name}_orchestrator",
            )

            # Run the orchestrator's ReAct loop
            agent_output = orchestrator.run(
                input=input,
                context=context,
                tools=delegate_tools,
                max_steps=orchestration.settings.get("max_steps", 25),
            )

            output.content = agent_output.content
            output.status = agent_output.status
            output.steps = agent_output.steps
            output.usage = agent_output.usage or {}
            output.events = agent_output.events
            output.coordination_metadata = {
                "strategy": "supervisor",
                "model": orchestration.settings.get("model", "gpt-4.1-mini"),
                "tool_calls": [tc.to_dict() for tc in agent_output.tool_calls],
            }
            output.error = agent_output.error

        except Exception as e:
            output.status = ExecutionStatus.FAILED
            output.error = str(e)

        output.completed_at = datetime.now(UTC)

        context.emit_event(
            {
                "type": "orchestration_completed",
                "status": output.status.value,
                "steps": output.steps,
                "usage": output.usage,
            }
        )

        return output
