"""
Tests for the Agent module.
"""

import pytest

from agentic import Agent, AgentOutput, AgentSession, ExecutionStatus


class TestAgentCreation:
    """Tests for Agent instantiation."""

    def test_agent_default_values(self):
        """Agent should have sensible defaults."""
        agent = Agent()

        assert agent.model == "gpt-4o-mini"
        assert agent.system_prompt == ""
        assert agent.name is None

    def test_agent_with_custom_values(self):
        """Agent should accept custom configuration."""
        agent = Agent(
            model="gpt-4o",
            system_prompt="You are a coding assistant.",
            name="coder",
        )

        assert agent.model == "gpt-4o"
        assert agent.system_prompt == "You are a coding assistant."
        assert agent.name == "coder"

    def test_agent_repr(self):
        """Agent should have a useful string representation."""
        agent = Agent(model="gpt-4o", name="assistant")

        repr_str = repr(agent)
        assert "gpt-4o" in repr_str
        assert "assistant" in repr_str


class TestAgentRun:
    """Tests for Agent.run() execution."""

    def test_agent_run_returns_output(self, mock_litellm):
        """run() should return an AgentOutput."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert isinstance(output, AgentOutput)

    def test_agent_run_success_status(self, mock_litellm):
        """Successful run should have COMPLETED status."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.status == ExecutionStatus.COMPLETED
        assert output.is_success()
        assert output.error is None

    def test_agent_run_content(self, mock_litellm):
        """run() should return the LLM response content."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.content == "Test response"

    def test_agent_run_usage(self, mock_litellm):
        """run() should include token usage."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.usage is not None
        assert output.usage["prompt_tokens"] == 10
        assert output.usage["completion_tokens"] == 20
        assert output.usage["total_tokens"] == 30

    def test_agent_run_messages_include_system(self, mock_litellm):
        """Messages should include system prompt if provided."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert len(output.messages) >= 2
        assert output.messages[0]["role"] == "system"
        assert output.messages[0]["content"] == "You are helpful"

    def test_agent_run_with_message_list(self, mock_litellm):
        """run() should accept message list as input."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        assert output.status == ExecutionStatus.COMPLETED

    def test_agent_run_error_handling(self, mock_litellm_error):
        """run() should handle errors gracefully."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.status == ExecutionStatus.FAILED
        assert output.is_failed()
        assert output.error is not None
        assert "LLM API Error" in output.error

    def test_agent_run_has_execution_id(self, mock_litellm):
        """run() should generate an execution ID."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.execution_id is not None
        assert len(output.execution_id) > 0

    def test_agent_run_has_timing(self, mock_litellm):
        """run() should track start and completion times."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.started_at is not None
        assert output.completed_at is not None
        assert output.completed_at >= output.started_at


class TestAgentSession:
    """Tests for AgentSession."""

    def test_session_creation(self):
        """Session should be created with default values."""
        session = AgentSession()

        assert session.session_id is not None
        assert session.outputs == []
        assert session.agent_name is None

    def test_session_add_output(self, mock_litellm):
        """Session should store outputs."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        output = agent.run("Hello", session=session)
        session.add_output(output)

        assert len(session.outputs) == 1
        assert session.outputs[0] == output

    def test_session_get_messages(self, mock_litellm):
        """get_messages() should return conversation history."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        output = agent.run("Hello", session=session)
        session.add_output(output)

        messages = session.get_messages()

        # Should have user message and assistant response
        assert len(messages) >= 1

    def test_session_get_messages_excludes_system_by_default(self, mock_litellm):
        """get_messages() should exclude system messages by default."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        output = agent.run("Hello", session=session)
        session.add_output(output)

        messages = session.get_messages(include_system=False)

        for msg in messages:
            assert msg["role"] != "system"

    def test_session_get_last_output(self, mock_litellm):
        """get_last_output() should return most recent output."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        output1 = agent.run("Hello", session=session)
        session.add_output(output1)

        output2 = agent.run("Goodbye", session=session)
        session.add_output(output2)

        last = session.get_last_output()
        assert last == output2

    def test_session_clear(self, mock_litellm):
        """clear() should reset the session."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        output = agent.run("Hello", session=session)
        session.add_output(output)

        session.clear()

        assert len(session.outputs) == 0

    def test_session_multi_turn(self, mock_litellm):
        """Session should maintain context across turns."""
        agent = Agent(system_prompt="You are helpful")
        session = AgentSession()

        # First turn
        output1 = agent.run("My name is Alice", session=session)
        session.add_output(output1)

        # Second turn - verify session is passed
        _ = agent.run("What's my name?", session=session)

        # Verify litellm was called with history
        calls = mock_litellm.completion.call_args_list
        assert len(calls) == 2

        # Second call should include more messages (history)
        second_call_messages = calls[1][1]["messages"]
        assert len(second_call_messages) > 2  # system + user + history


class TestAgentOutput:
    """Tests for AgentOutput."""

    def test_output_from_error(self):
        """from_error() should create a failed output."""
        output = AgentOutput.from_error("exec-123", "Something went wrong")

        assert output.execution_id == "exec-123"
        assert output.status == ExecutionStatus.FAILED
        assert output.error == "Something went wrong"

    def test_output_get_content(self, mock_litellm):
        """get_content() should return content or empty string."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.get_content() == "Test response"

        # Test with None content
        output.content = None
        assert output.get_content() == ""

    def test_output_total_tokens(self, mock_litellm):
        """total_tokens() should return token count."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        assert output.total_tokens() == 30

    def test_output_duration(self, mock_litellm):
        """duration_seconds() should return execution time."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        duration = output.duration_seconds()
        assert duration is not None
        assert duration >= 0

    def test_output_to_dict(self, mock_litellm):
        """to_dict() should serialize output."""
        agent = Agent(system_prompt="You are helpful")
        output = agent.run("Hello")

        data = output.to_dict()

        assert "execution_id" in data
        assert "status" in data
        assert data["status"] == "completed"
        assert "content" in data


def _consume_stream(gen):
    """Helper to consume a stream generator and get its return value."""
    chunks = []
    output = None
    while True:
        try:
            chunk = next(gen)
            chunks.append(chunk)
        except StopIteration as e:
            output = e.value
            break
    return chunks, output


class TestAgentStream:
    """Tests for Agent.stream() method."""

    def test_stream_yields_chunks(self, mock_litellm_streaming):
        """stream() should yield content chunks."""
        agent = Agent(system_prompt="You are helpful")

        chunks, _ = _consume_stream(agent.stream("Hello"))

        assert len(chunks) == 4
        assert chunks[0] == "Hello"
        assert chunks[1] == " "
        assert chunks[2] == "World"
        assert chunks[3] == "!"

    def test_stream_returns_output(self, mock_litellm_streaming):
        """stream() should return AgentOutput after iteration."""
        agent = Agent(system_prompt="You are helpful")

        chunks, output = _consume_stream(agent.stream("Hello"))

        assert isinstance(output, AgentOutput)
        assert output.status == ExecutionStatus.COMPLETED
        assert output.content == "Hello World!"

    def test_stream_content_is_concatenated(self, mock_litellm_streaming):
        """stream() should concatenate chunks into final content."""
        agent = Agent(system_prompt="You are helpful")

        chunks, output = _consume_stream(agent.stream("Hello"))

        # Collected chunks should match
        collected = "".join(chunks)
        assert collected == "Hello World!"

        # Final output content should match
        assert output.content == "Hello World!"

    def test_stream_includes_messages(self, mock_litellm_streaming):
        """stream() output should include all messages."""
        agent = Agent(system_prompt="You are helpful")

        _, output = _consume_stream(agent.stream("Hello"))

        # Should have system, user, and assistant messages
        assert len(output.messages) >= 3
        assert output.messages[0]["role"] == "system"
        assert output.messages[-1]["role"] == "assistant"
        assert output.messages[-1]["content"] == "Hello World!"

    def test_stream_has_execution_id(self, mock_litellm_streaming):
        """stream() output should have execution ID."""
        agent = Agent(system_prompt="You are helpful")

        _, output = _consume_stream(agent.stream("Hello"))

        assert output.execution_id is not None

    def test_stream_error_handling(self, mock_litellm_streaming_error):
        """stream() should handle errors gracefully."""
        agent = Agent(system_prompt="You are helpful")

        _, output = _consume_stream(agent.stream("Hello"))

        assert output.status == ExecutionStatus.FAILED
        assert "LLM Streaming Error" in output.error


class TestAgentAstream:
    """Tests for Agent.astream() async method."""

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self, mock_litellm_streaming):
        """astream() should yield content chunks asynchronously."""
        agent = Agent(system_prompt="You are helpful")

        chunks = []
        async for chunk in agent.astream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert "".join(chunks) == "Hello World!"

    @pytest.mark.asyncio
    async def test_astream_can_be_collected(self, mock_litellm_streaming):
        """astream() chunks can be collected into full content."""
        agent = Agent(system_prompt="You are helpful")

        content_parts = []
        async for chunk in agent.astream("Hello"):
            content_parts.append(chunk)

        full_content = "".join(content_parts)
        assert full_content == "Hello World!"


class TestToolCallRecord:
    def test_create(self):
        from agentic.agent.output import ToolCallRecord

        record = ToolCallRecord(
            step=1,
            tool_name="search",
            arguments={"q": "hello"},
            result="found it",
            duration_ms=150,
        )
        assert record.step == 1
        assert record.tool_name == "search"
        assert record.duration_ms == 150

    def test_to_dict(self):
        from agentic.agent.output import ToolCallRecord

        record = ToolCallRecord(
            step=2,
            tool_name="db",
            arguments={"sql": "SELECT 1"},
            result="1",
            duration_ms=50,
        )
        d = record.to_dict()
        assert d["step"] == 2
        assert d["tool_name"] == "db"
        assert d["arguments"] == {"sql": "SELECT 1"}

    def test_to_dict_with_usage(self):
        from agentic.agent.output import ToolCallRecord

        record = ToolCallRecord(
            step=1,
            tool_name="x",
            arguments={},
            result="ok",
            duration_ms=10,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        d = record.to_dict()
        assert "usage" in d
        assert d["usage"]["prompt_tokens"] == 100

    def test_to_dict_without_usage(self):
        from agentic.agent.output import ToolCallRecord

        record = ToolCallRecord(
            step=1,
            tool_name="x",
            arguments={},
            result="ok",
            duration_ms=10,
        )
        d = record.to_dict()
        assert "usage" not in d


class TestAgentOutputNewFields:
    def test_default_values(self):
        output = AgentOutput(execution_id="test")
        assert output.steps == 0
        assert output.tool_calls == []
        assert output.events == []

    def test_total_usage_aggregation(self):
        output = AgentOutput(
            execution_id="test",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert output.total_tokens() == 150


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_status_is_terminal(self):
        """is_terminal() should identify terminal states."""
        assert ExecutionStatus.COMPLETED.is_terminal()
        assert ExecutionStatus.FAILED.is_terminal()
        assert ExecutionStatus.CANCELLED.is_terminal()

        assert not ExecutionStatus.PENDING.is_terminal()
        assert not ExecutionStatus.RUNNING.is_terminal()

    def test_status_is_success(self):
        """is_success() should only be true for COMPLETED."""
        assert ExecutionStatus.COMPLETED.is_success()

        assert not ExecutionStatus.PENDING.is_success()
        assert not ExecutionStatus.RUNNING.is_success()
        assert not ExecutionStatus.FAILED.is_success()
        assert not ExecutionStatus.CANCELLED.is_success()

    def test_status_str_value(self):
        """Status should serialize to lowercase string."""
        # Use .value for explicit string access
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.FAILED.value == "failed"
