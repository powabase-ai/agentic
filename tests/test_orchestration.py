from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext
from agentic.orchestration.engine import get_strategy_engine
from agentic.orchestration.orchestration import Orchestration


def _mock_completion_response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, role="assistant", tool_calls=None
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _mock_tool_call_response(tool_name, arguments_json):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name=tool_name, arguments=arguments_json
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestOrchestration:
    def test_create(self):
        orch = Orchestration(
            name="test", description="test orch", strategy="supervisor"
        )
        assert orch.name == "test"
        assert orch.strategy == "supervisor"
        assert orch.entities == []

    def test_add_agent_entity(self):
        agent = Agent(model="gpt-4o-mini", system_prompt="specialist")
        orch = Orchestration(name="test", description="test")
        orch.add_entity(
            entity_type="agent",
            agent=agent,
            role_description="Handles claims",
        )
        assert len(orch.entities) == 1
        assert orch.entities[0].entity_type == "agent"
        assert orch.entities[0].role_description == "Handles claims"

    def test_add_tool_entity(self):
        tool = BuiltinTool(
            name="db_query",
            description="Query DB",
            input_schema={"type": "object"},
            handler=lambda a, c: "ok",
        )
        orch = Orchestration(name="test", description="test")
        orch.add_entity(entity_type="tool", tool=tool)
        assert len(orch.entities) == 1
        assert orch.entities[0].entity_type == "tool"


class TestGetStrategyEngine:
    def test_supervisor(self):
        engine = get_strategy_engine("supervisor")
        assert engine is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy_engine("nonexistent")


class TestSupervisorEngine:
    @patch("agentic.agent.agent.litellm")
    def test_single_delegation(self, mock_litellm):
        """Supervisor delegates to one agent, gets result, synthesizes."""
        # Step 1: Supervisor calls delegate_to_specialist
        delegate_call = _mock_tool_call_response(
            "delegate_to_specialist",
            '{"task": "Analyze the claim"}',
        )
        # Step 2 (inside specialist): Specialist responds
        specialist_response = _mock_completion_response(
            "Claim is valid, $5000 coverage."
        )
        # Step 3: Supervisor synthesizes
        final_response = _mock_completion_response(
            "Based on analysis: claim approved for $5000."
        )

        mock_litellm.completion.side_effect = [
            delegate_call,  # Supervisor step 1
            specialist_response,  # Specialist step 1 (inside DelegateTool)
            final_response,  # Supervisor step 2
        ]

        specialist = Agent(
            model="gpt-4o-mini", system_prompt="You analyze claims.", name="specialist"
        )

        orch = Orchestration(name="claims_processor", description="Processes claims")
        orch.add_entity(
            entity_type="agent",
            agent=specialist,
            role_description="Analyzes insurance claims and determines coverage",
        )

        output = orch.run("I had a car accident, what's my coverage?")

        assert output.content == "Based on analysis: claim approved for $5000."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 3

    @patch("agentic.agent.agent.litellm")
    def test_multiple_delegations(self, mock_litellm):
        """Supervisor delegates to two agents sequentially."""
        # Supervisor calls agent_a
        delegate_a = _mock_tool_call_response("delegate_to_agent_a", '{"task": "Do A"}')
        agent_a_resp = _mock_completion_response("A is done.")
        # Supervisor calls agent_b
        delegate_b = _mock_tool_call_response("delegate_to_agent_b", '{"task": "Do B"}')
        agent_b_resp = _mock_completion_response("B is done.")
        # Supervisor synthesizes
        final = _mock_completion_response("Both A and B are done.")

        mock_litellm.completion.side_effect = [
            delegate_a,
            agent_a_resp,
            delegate_b,
            agent_b_resp,
            final,
        ]

        agent_a = Agent(model="gpt-4o-mini", system_prompt="Agent A", name="agent_a")
        agent_b = Agent(model="gpt-4o-mini", system_prompt="Agent B", name="agent_b")

        orch = Orchestration(name="multi", description="multi delegation")
        orch.add_entity(entity_type="agent", agent=agent_a, role_description="Does A")
        orch.add_entity(entity_type="agent", agent=agent_b, role_description="Does B")

        output = orch.run("Do both A and B")

        assert output.content == "Both A and B are done."
        assert output.status.is_success()
        assert mock_litellm.completion.call_count == 5

    @patch("agentic.agent.agent.litellm")
    def test_events_emitted(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_completion_response(
            "Direct answer."
        )

        agent = Agent(model="gpt-4o-mini", system_prompt="test", name="agent1")
        orch = Orchestration(name="test", description="test")
        orch.add_entity(entity_type="agent", agent=agent, role_description="test agent")

        events = []
        ctx = ExecutionContext(
            execution_id="orch-1",
            on_event=lambda e: events.append(e),
        )
        orch.run("hello", context=ctx)

        event_types = [e["type"] for e in events]
        assert "orchestration_started" in event_types
        assert "step_started" in event_types
        assert "orchestration_completed" in event_types

    @patch("agentic.agent.agent.litellm")
    def test_no_agents_returns_error(self, mock_litellm):
        orch = Orchestration(name="empty", description="empty")
        output = orch.run("hello")
        assert output.status.value == "failed"
        assert "No agent entities" in output.error

    @patch("agentic.agent.agent.litellm")
    def test_direct_tool_entity(self, mock_litellm):
        """Supervisor can use tools directly (not just delegate)."""
        tool_call = _mock_tool_call_response("db_query", '{"query": "SELECT 1"}')
        final = _mock_completion_response("Query returned 1.")
        mock_litellm.completion.side_effect = [tool_call, final]

        db_tool = BuiltinTool(
            name="db_query",
            description="Query DB",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            handler=lambda args, ctx: "1",
        )
        agent = Agent(model="gpt-4o-mini", system_prompt="test", name="dummy")

        orch = Orchestration(name="test", description="test")
        orch.add_entity(entity_type="agent", agent=agent, role_description="test")
        orch.add_entity(entity_type="tool", tool=db_tool)

        output = orch.run("What does SELECT 1 return?")
        assert output.content == "Query returned 1."
