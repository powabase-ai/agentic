"""Tests for Sequential and Parallel orchestration strategies."""

from types import SimpleNamespace
from unittest.mock import patch

from agentic import Agent
from agentic.execution.context import ExecutionContext
from agentic.orchestration.engine import get_strategy_engine
from agentic.orchestration.orchestration import Orchestration


def _mock_response(content, finish_reason="stop"):
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


class TestGetStrategyEngine:
    def test_sequential(self):
        engine = get_strategy_engine("sequential")
        assert engine is not None

    def test_parallel(self):
        engine = get_strategy_engine("parallel")
        assert engine is not None


class TestSequentialEngine:
    @patch("agentic.agent.agent.litellm")
    def test_agents_run_in_order(self, mock_litellm):
        """Agents run in position order, each receiving previous output as input."""
        mock_litellm.completion.side_effect = [
            _mock_response("Step 1 done: extracted data."),
            _mock_response("Step 2 done: analyzed extracted data."),
            _mock_response("Step 3 done: generated report."),
        ]

        agent_a = Agent(
            model="gpt-4o-mini", system_prompt="Extract data.", name="extractor"
        )
        agent_b = Agent(
            model="gpt-4o-mini", system_prompt="Analyze data.", name="analyzer"
        )
        agent_c = Agent(
            model="gpt-4o-mini", system_prompt="Generate report.", name="reporter"
        )

        orch = Orchestration(
            name="pipeline", description="ETL pipeline", strategy="sequential"
        )
        orch.add_entity(
            entity_type="agent", agent=agent_a, role_description="Extract", position=0
        )
        orch.add_entity(
            entity_type="agent", agent=agent_b, role_description="Analyze", position=1
        )
        orch.add_entity(
            entity_type="agent", agent=agent_c, role_description="Report", position=2
        )

        output = orch.run("Process this document")

        assert output.status.is_success()
        assert output.content == "Step 3 done: generated report."
        assert mock_litellm.completion.call_count == 3

        # Verify chaining: agent B should receive agent A's output
        second_call_messages = mock_litellm.completion.call_args_list[1].kwargs[
            "messages"
        ]
        assert any(
            "Step 1 done" in str(m.get("content", "")) for m in second_call_messages
        )

        # Agent C should receive agent B's output
        third_call_messages = mock_litellm.completion.call_args_list[2].kwargs[
            "messages"
        ]
        assert any(
            "Step 2 done" in str(m.get("content", "")) for m in third_call_messages
        )

    @patch("agentic.agent.agent.litellm")
    def test_sequential_events(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("Done.")

        agent = Agent(model="gpt-4o-mini", system_prompt="test", name="agent1")
        orch = Orchestration(name="test", description="test", strategy="sequential")
        orch.add_entity(entity_type="agent", agent=agent, position=0)

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))
        orch.run("hello", context=ctx)

        event_types = [e["type"] for e in events]
        assert "orchestration_started" in event_types
        assert "sequential_step" in event_types
        assert "orchestration_completed" in event_types

    @patch("agentic.agent.agent.litellm")
    def test_sequential_empty_fails(self, mock_litellm):
        orch = Orchestration(name="empty", description="empty", strategy="sequential")
        output = orch.run("hello")
        assert output.status.value == "failed"

    @patch("agentic.agent.agent.litellm")
    def test_sequential_agent_failure_stops_pipeline(self, mock_litellm):
        """If an agent fails, the pipeline stops."""
        mock_litellm.completion.side_effect = Exception("LLM error")

        agent = Agent(model="gpt-4o-mini", system_prompt="test", name="failing")
        orch = Orchestration(name="test", description="test", strategy="sequential")
        orch.add_entity(entity_type="agent", agent=agent, position=0)

        output = orch.run("hello")
        assert output.status.value == "failed"


class TestParallelEngine:
    @patch("agentic.agent.agent.litellm")
    def test_agents_run_and_merge(self, mock_litellm):
        """All agents run on same input, results merged by LLM."""
        mock_litellm.completion.side_effect = [
            _mock_response("Analysis A: positive sentiment."),
            _mock_response("Analysis B: revenue up 20%."),
            _mock_response("Combined: positive sentiment, revenue up 20%."),
        ]

        agent_a = Agent(
            model="gpt-4o-mini", system_prompt="Sentiment analysis.", name="sentiment"
        )
        agent_b = Agent(
            model="gpt-4o-mini", system_prompt="Financial analysis.", name="financial"
        )

        orch = Orchestration(
            name="multi-analysis", description="Multi-perspective", strategy="parallel"
        )
        orch.add_entity(
            entity_type="agent", agent=agent_a, role_description="Sentiment"
        )
        orch.add_entity(
            entity_type="agent", agent=agent_b, role_description="Financial"
        )

        output = orch.run("Analyze this quarterly report")

        assert output.status.is_success()
        assert mock_litellm.completion.call_count >= 3  # 2 parallel + 1 merge

    @patch("agentic.agent.agent.litellm")
    def test_parallel_single_agent_no_merge(self, mock_litellm):
        """With one agent, no merge step needed."""
        mock_litellm.completion.return_value = _mock_response("Only result.")

        agent = Agent(model="gpt-4o-mini", system_prompt="test", name="solo")
        orch = Orchestration(name="solo", description="single", strategy="parallel")
        orch.add_entity(entity_type="agent", agent=agent)

        output = orch.run("hello")
        assert output.content == "Only result."
        assert mock_litellm.completion.call_count == 1

    @patch("agentic.agent.agent.litellm")
    def test_parallel_empty_fails(self, mock_litellm):
        orch = Orchestration(name="empty", description="empty", strategy="parallel")
        output = orch.run("hello")
        assert output.status.value == "failed"

    @patch("agentic.agent.agent.litellm")
    def test_parallel_events(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _mock_response("Result A"),
            _mock_response("Result B"),
            _mock_response("Merged"),
        ]

        agent_a = Agent(model="gpt-4o-mini", system_prompt="A", name="a")
        agent_b = Agent(model="gpt-4o-mini", system_prompt="B", name="b")

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        orch = Orchestration(name="test", description="test", strategy="parallel")
        orch.add_entity(entity_type="agent", agent=agent_a)
        orch.add_entity(entity_type="agent", agent=agent_b)

        orch.run("test", context=ctx)

        event_types = [e["type"] for e in events]
        assert "orchestration_started" in event_types

    @patch("agentic.agent.agent.litellm")
    def test_parallel_agent_failure_fails_orchestration(self, mock_litellm):
        """If one parallel agent fails (returns failed status), orchestration should fail."""
        mock_litellm.completion.side_effect = [
            _mock_response("Agent A succeeded."),
            Exception("Agent B LLM error"),
        ]

        agent_a = Agent(model="gpt-4o-mini", system_prompt="A", name="good_agent")
        agent_b = Agent(model="gpt-4o-mini", system_prompt="B", name="bad_agent")

        orch = Orchestration(name="test", description="test", strategy="parallel")
        orch.add_entity(entity_type="agent", agent=agent_a)
        orch.add_entity(entity_type="agent", agent=agent_b)

        output = orch.run("test")
        assert output.status.value == "failed"
