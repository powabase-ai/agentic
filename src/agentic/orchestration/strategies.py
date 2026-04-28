"""Orchestration strategy implementations."""

from __future__ import annotations

import concurrent.futures
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agentic.agent.agent import Agent
from agentic.agent.tools import DelegateTool, ToolDefinition
from agentic.execution.context import ExecutionContext
from agentic.execution.status import ExecutionStatus
from agentic.orchestration.engine import StrategyEngine
from agentic.orchestration.output import OrchestrationOutput

logger = logging.getLogger(__name__)


def _build_orchestrator_prompt(orchestration, entities) -> str:
    """Build the supervisor agent's system prompt."""
    agent_descriptions = []
    for entity in entities:
        if entity.entity_type == "agent" and entity.agent:
            name = entity.agent.name or "unnamed"
            desc = entity.role_description or (entity.agent.system_prompt or "")[:200]
            agent_descriptions.append(
                f"- delegate_to_{_sanitize_tool_name(name)}: {desc}"
            )

    agent_list = (
        "\n".join(agent_descriptions) if agent_descriptions else "(no agents available)"
    )
    additional = orchestration.orchestrator_config.get("additional_instructions", "")

    return f"""You are an orchestrator for: {orchestration.description}

You have the following specialist agents available as tools:
{agent_list}

Your job is to:
1. Understand the user's request
2. Plan which specialists to use and in what order
3. Delegate to each required specialist in sequence — do NOT stop after just one delegation if the task requires multiple
4. Use the output from earlier delegations as context when delegating to later ones
5. After ALL necessary delegations are complete, synthesize their responses into a coherent answer for the user

IMPORTANT: If your instructions or the task require multiple agents, you MUST call each one. Do not skip agents or stop early. Complete the full workflow before responding to the user.

{additional}""".strip()


def _sanitize_tool_name(name: str) -> str:
    """Sanitize a name to be a valid LLM tool name (^[a-zA-Z0-9_-]+$)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _build_delegate_tools(
    entities, on_run_complete: Callable[[dict], None] | None = None
) -> dict[str, DelegateTool]:
    """Build delegate tools from agent entities."""
    tools: dict[str, DelegateTool] = {}
    for entity in entities:
        if entity.entity_type == "agent" and entity.agent:
            name = entity.agent.name or "unnamed"
            tool_name = f"delegate_to_{_sanitize_tool_name(name)}"
            tools[tool_name] = DelegateTool(
                name=tool_name,
                description=entity.role_description or f"Delegate to {name}",
                agent=entity.agent,
                agent_tools=entity.agent_tools or {},
                max_steps=entity.config.get("max_steps", 10),
                on_run_complete=on_run_complete,
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
        *,
        history: list[dict] | None = None,
        on_delegate_complete: Callable[[dict], None] | None = None,
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
            delegate_tools: dict[str, ToolDefinition] = _build_delegate_tools(
                entities, on_run_complete=on_delegate_complete
            )

            # Also include any tool entities directly
            for entity in entities:
                if entity.entity_type == "tool" and entity.tool:
                    delegate_tools[entity.tool.name] = entity.tool

            # Create the orchestrator agent
            orchestrator = Agent(
                model=orchestration.settings.get("model", "gpt-5.4"),
                system_prompt=_build_orchestrator_prompt(orchestration, entities),
                name=f"{orchestration.name}_orchestrator",
            )

            # Build input with history for multi-turn conversations
            if history:
                agent_input: str | list[dict] = list(history) + [
                    {"role": "user", "content": input}
                ]
            else:
                agent_input = input

            # Run the orchestrator's ReAct loop
            agent_output = orchestrator.run(
                input=agent_input,
                context=context,
                tools=delegate_tools,
                max_steps=orchestration.settings.get("max_steps", 25),
            )

            # output.usage holds the ORCHESTRATOR's own LLM tokens only — the
            # children land in their respective agent_run rows via the
            # on_delegate_complete persistence hook. Keeping the orchestration
            # row leaf-only avoids double-counting in the platform's
            # "tokens by all sources" dashboard rollup. Per-run drill-down UI
            # sums orch_run + child_runs explicitly.
            output.content = agent_output.content
            output.status = agent_output.status
            output.steps = agent_output.steps
            output.usage = agent_output.usage or {}
            output.events = agent_output.events
            output.coordination_metadata = {
                "strategy": "supervisor",
                "model": orchestration.settings.get("model", "gpt-5.4"),
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


class SequentialEngine(StrategyEngine):
    """Deterministic pipeline: agents run in position order."""

    def execute(
        self,
        orchestration,
        input: str,
        session: Any,
        context: ExecutionContext | None,
        *,
        history: list[dict] | None = None,
        on_delegate_complete: Callable[[dict], None] | None = None,
    ) -> OrchestrationOutput:
        if context is None:
            context = ExecutionContext()

        output = OrchestrationOutput(execution_id=context.execution_id)
        output.status = ExecutionStatus.RUNNING
        output.started_at = datetime.now(UTC)

        agent_entities = sorted(
            [e for e in orchestration.entities if e.entity_type == "agent"],
            key=lambda e: e.position or 0,
        )

        if not agent_entities:
            output.status = ExecutionStatus.FAILED
            output.error = "No agent entities in orchestration"
            output.completed_at = datetime.now(UTC)
            return output

        context.emit_event(
            {
                "type": "orchestration_started",
                "strategy": "sequential",
                "name": orchestration.name,
                "agent_count": len(agent_entities),
            }
        )

        # Prepend history to input for the first agent in the pipeline
        if history:
            current_input: str | list[dict] = list(history) + [
                {"role": "user", "content": input}
            ]
        else:
            current_input: str | list[dict] = input
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            for i, entity in enumerate(agent_entities):
                agent_name = entity.agent.name or f"agent_{i}"
                context.emit_event(
                    {"type": "sequential_step", "step": i, "agent": agent_name}
                )

                child_ctx = context.child_context()
                agent_output = entity.agent.run(
                    input=current_input,
                    context=child_ctx,
                    tools=entity.agent_tools if entity.agent_tools else None,
                    max_steps=entity.config.get("max_steps", 10),
                )

                if on_delegate_complete is not None:
                    try:
                        on_delegate_complete(
                            {
                                "agent_name": agent_name,
                                "model": entity.agent.model if entity.agent else None,
                                "child_execution_id": child_ctx.execution_id,
                                "orchestration_run_id": context.orchestration_run_id,
                                "task": current_input
                                if isinstance(current_input, str)
                                else str(current_input),
                                "content": agent_output.content,
                                "status": agent_output.status,
                                "error": agent_output.error,
                                "usage": agent_output.usage or {},
                                "steps": agent_output.steps,
                                "events": agent_output.events,
                                "messages": agent_output.messages or [],
                                "tool_calls": [
                                    tc.to_dict()
                                    for tc in (agent_output.tool_calls or [])
                                ],
                                "started_at": agent_output.started_at,
                                "completed_at": agent_output.completed_at,
                            }
                        )
                    except Exception:
                        logger.exception(
                            "SequentialEngine.on_delegate_complete hook failed for %s; continuing",
                            agent_name,
                        )

                if agent_output.usage:
                    for k in total_usage:
                        total_usage[k] += (agent_output.usage or {}).get(k, 0)

                if not agent_output.status.is_success():
                    output.status = ExecutionStatus.FAILED
                    output.error = f"Agent '{agent_name}' failed: {agent_output.error}"
                    output.completed_at = datetime.now(UTC)
                    context.emit_event(
                        {"type": "orchestration_completed", "status": "failed"}
                    )
                    return output

                current_input = agent_output.content or ""

            output.content = current_input
            output.status = ExecutionStatus.COMPLETED
            output.usage = total_usage
            output.steps = len(agent_entities)
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


class ParallelEngine(StrategyEngine):
    """Concurrent execution: all agents simultaneously, then merge."""

    def execute(
        self,
        orchestration,
        input: str,
        session: Any,
        context: ExecutionContext | None,
        *,
        history: list[dict] | None = None,
        on_delegate_complete: Callable[[dict], None] | None = None,
    ) -> OrchestrationOutput:
        if context is None:
            context = ExecutionContext()

        output = OrchestrationOutput(execution_id=context.execution_id)
        output.status = ExecutionStatus.RUNNING
        output.started_at = datetime.now(UTC)

        agent_entities = [e for e in orchestration.entities if e.entity_type == "agent"]

        if not agent_entities:
            output.status = ExecutionStatus.FAILED
            output.error = "No agent entities in orchestration"
            output.completed_at = datetime.now(UTC)
            return output

        context.emit_event(
            {
                "type": "orchestration_started",
                "strategy": "parallel",
                "name": orchestration.name,
                "agent_count": len(agent_entities),
            }
        )

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            # Build input with history for multi-turn conversations
            if history:
                parallel_input: str | list[dict] = list(history) + [
                    {"role": "user", "content": input}
                ]
            else:
                parallel_input: str | list[dict] = input

            def run_agent(entity):
                child_ctx = context.child_context()
                agent_out = entity.agent.run(
                    input=parallel_input,
                    context=child_ctx,
                    tools=entity.agent_tools if entity.agent_tools else None,
                    max_steps=entity.config.get("max_steps", 10),
                )
                return agent_out, child_ctx

            agent_outputs: dict[str, Any] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(agent_entities)
            ) as pool:
                future_to_entity = {
                    pool.submit(run_agent, e): e for e in agent_entities
                }
                for future in concurrent.futures.as_completed(future_to_entity):
                    entity = future_to_entity[future]
                    agent_name = entity.agent.name or "unnamed"
                    agent_out, child_ctx = future.result()
                    agent_outputs[agent_name] = agent_out
                    if agent_out.usage:
                        for k in total_usage:
                            total_usage[k] += (agent_out.usage or {}).get(k, 0)
                    if on_delegate_complete is not None:
                        try:
                            on_delegate_complete(
                                {
                                    "agent_name": agent_name,
                                    "model": entity.agent.model if entity.agent else None,
                                    "child_execution_id": child_ctx.execution_id,
                                    "orchestration_run_id": context.orchestration_run_id,
                                    "task": parallel_input
                                    if isinstance(parallel_input, str)
                                    else str(parallel_input),
                                    "content": agent_out.content,
                                    "status": agent_out.status,
                                    "error": agent_out.error,
                                    "usage": agent_out.usage or {},
                                    "steps": agent_out.steps,
                                    "events": agent_out.events,
                                    "messages": agent_out.messages or [],
                                    "tool_calls": [
                                        tc.to_dict()
                                        for tc in (agent_out.tool_calls or [])
                                    ],
                                    "started_at": agent_out.started_at,
                                    "completed_at": agent_out.completed_at,
                                }
                            )
                        except Exception:
                            logger.exception(
                                "ParallelEngine.on_delegate_complete hook failed for %s; continuing",
                                agent_name,
                            )

            # Check for failed agents
            failed = {
                name: out
                for name, out in agent_outputs.items()
                if not out.status.is_success()
            }
            if failed:
                output.status = ExecutionStatus.FAILED
                output.error = f"Agents failed: {', '.join(failed.keys())}"
                output.usage = total_usage
                output.completed_at = datetime.now(UTC)
                context.emit_event(
                    {"type": "orchestration_completed", "status": "failed"}
                )
                return output

            if len(agent_outputs) == 1:
                only_output = next(iter(agent_outputs.values()))
                output.content = only_output.content
                output.status = only_output.status
                output.usage = total_usage
                output.steps = 1
                output.completed_at = datetime.now(UTC)
                context.emit_event(
                    {"type": "orchestration_completed", "status": output.status.value}
                )
                return output

            # Abort check before expensive merge step
            if context.is_aborted:
                output.status = ExecutionStatus.CANCELLED
                output.error = "Aborted before merge"
                output.completed_at = datetime.now(UTC)
                return output

            merge_parts = ["Multiple agents analyzed the same input:\n"]
            for name, agent_out in agent_outputs.items():
                merge_parts.append(
                    f"--- {name} ---\n{agent_out.content or '(no output)'}\n"
                )
            merge_parts.append("\nSynthesize these into a single coherent response.")

            merge_agent = Agent(
                model=orchestration.settings.get("model", "gpt-5.4"),
                system_prompt="You combine multiple analysis results into a single coherent response.",
                name=f"{orchestration.name}_merger",
            )
            merge_output = merge_agent.run(
                input="\n".join(merge_parts), context=context
            )

            if merge_output.usage:
                for k in total_usage:
                    total_usage[k] += (merge_output.usage or {}).get(k, 0)

            output.content = merge_output.content
            output.status = ExecutionStatus.COMPLETED
            output.usage = total_usage
            output.steps = len(agent_entities) + 1
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
