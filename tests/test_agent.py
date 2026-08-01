"""
Tests for the Agent module.
"""

from unittest.mock import patch

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


def _make_streaming_chunks(
    content_fragments: list[str] | None = None,
    reasoning_fragments: list[str] | None = None,
    thinking_blocks: list[dict] | None = None,
    usage: dict | None = None,
):
    """Build a list of LiteLLM-shaped streaming chunks for tests.

    Each chunk has `choices[0].delta` with content/reasoning fields. A trailing
    chunk with `choices=[]` and a `usage` attribute is emitted when `usage` is
    set (mirrors LiteLLM's `stream_options={"include_usage": True}` behavior).
    """
    from types import SimpleNamespace

    def _at(seq: list | None, i: int):
        return seq[i] if seq and i < len(seq) else None

    chunks: list = []
    n = max(
        len(content_fragments or []),
        len(reasoning_fragments or []),
        len(thinking_blocks or []),
    )
    for i in range(n):
        block = _at(thinking_blocks, i)
        delta = SimpleNamespace(
            content=_at(content_fragments, i),
            reasoning_content=_at(reasoning_fragments, i),
            thinking_blocks=[block] if block else None,
            tool_calls=None,
            provider_specific_fields=None,
        )
        chunks.append(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=delta, finish_reason=None)],
                usage=None,
            )
        )
    if usage is not None:
        chunks.append(
            SimpleNamespace(choices=[], usage=SimpleNamespace(**usage)),
        )
    return chunks


class TestAgentStreamCallbacks:
    """Streaming callback contract — callbacks fire per fragment and
    reasoning_artifact surfaces on the returned AgentOutput."""

    def test_stream_invokes_on_content_delta_callback(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(content_fragments=["Hello", " ", "World"])
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            agent = Agent(system_prompt="hi")

            received: list[str] = []
            _consume_stream(
                agent.stream("go", on_content_delta=lambda d: received.append(d))
            )

        assert received == ["Hello", " ", "World"]

    def test_stream_invokes_on_reasoning_delta_callback(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(
            content_fragments=["A", "B"],
            reasoning_fragments=["thinking ", "more"],
        )
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            agent = Agent(system_prompt="hi")

            content_received: list[str] = []
            reasoning_received: list[str] = []
            _consume_stream(
                agent.stream(
                    "go",
                    on_content_delta=lambda d: content_received.append(d),
                    on_reasoning_delta=lambda d: reasoning_received.append(d),
                )
            )

        assert content_received == ["A", "B"]
        assert reasoning_received == ["thinking ", "more"]

    def test_stream_populates_reasoning_artifact_for_anthropic(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(
            content_fragments=["answer"],
            reasoning_fragments=["I'm thinking"],
            thinking_blocks=[
                {
                    "index": 0,
                    "type": "thinking",
                    "thinking": "step 1",
                    "signature": "sig",
                },
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 42,
                "total_tokens": 52,
            },
        )
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            mock_litellm.get_llm_provider.return_value = (
                "claude-sonnet-4-6",
                "anthropic",
                None,
                None,
            )

            agent = Agent(
                model="claude-sonnet-4-6",
                system_prompt="hi",
                reasoning_effort="high",
            )

            _, output = _consume_stream(agent.stream("go"))

        assert output.reasoning_artifact is not None
        assert output.reasoning_requested is True
        assert output.reasoning_artifact.summary_text == "I'm thinking"
        assert len(output.reasoning_artifact.thinking_blocks) == 1
        # N-3: token counts must round-trip through the usage stub.
        assert output.reasoning_artifact.output_tokens == 42

    def test_stream_no_artifact_when_no_reasoning(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(content_fragments=["plain answer"])
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            mock_litellm.get_llm_provider.return_value = (
                "gpt-4o-mini",
                "openai",
                None,
                None,
            )
            agent = Agent(system_prompt="hi")

            _, output = _consume_stream(agent.stream("go"))

        assert output.reasoning_artifact is None
        assert output.reasoning_requested is False

    def test_stream_worker_error_returns_failed_output(self, monkeypatch):
        """C-4: a worker exception must surface as a FAILED AgentOutput, not
        leak. Also covers the worker error branch added for N-2/C-3."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")

        def explode(*args, **kwargs):
            raise RuntimeError("synthetic provider boom")

        with patch("agentic.llm.streaming.accumulate_stream", side_effect=explode):
            with patch("agentic.agent.agent.litellm") as mock_litellm:
                mock_litellm.completion.return_value = iter([])
                agent = Agent(system_prompt="hi")

                _, output = _consume_stream(agent.stream("go"))

        assert output.status == ExecutionStatus.FAILED
        assert "synthetic provider boom" in (output.error or "")
        # reasoning_requested still populates on the FAILED branch
        assert output.reasoning_requested is False

    def test_stream_passes_reasoning_effort_anthropic_top_level(self, monkeypatch):
        """Non-Responses providers (Anthropic direct) get a top-level
        `reasoning_effort` kwarg AND must NOT get `extra_body` (the two
        shapes are mutually exclusive — see routing.py)."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(content_fragments=["hi"])
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            mock_litellm.supports_reasoning.return_value = True
            mock_litellm.get_llm_provider.return_value = (
                "claude-sonnet-4-6",
                "anthropic",
                None,
                None,
            )
            agent = Agent(
                model="claude-sonnet-4-6",
                system_prompt="hi",
                reasoning_effort="high",
            )
            _consume_stream(agent.stream("go"))

            call_kwargs = mock_litellm.completion.call_args.kwargs
            assert call_kwargs.get("reasoning_effort") == "high"
            assert (
                "extra_body" not in call_kwargs
            ), "Responses-bridge shape must not appear on the non-Responses path"

    def test_stream_passes_reasoning_effort_openai_responses_bridge(self, monkeypatch):
        """OpenAI reasoning models route through `openai/responses/<model>`
        and must pack effort under `extra_body.reasoning.effort` — and MUST
        NOT carry a top-level `reasoning_effort` (litellm silently drops it
        on the Responses path)."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(content_fragments=["hi"])
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            mock_litellm.supports_reasoning.return_value = True
            mock_litellm.get_llm_provider.return_value = (
                "gpt-5",
                "openai",
                None,
                None,
            )
            agent = Agent(
                model="gpt-5",
                system_prompt="hi",
                reasoning_effort="high",
            )
            _consume_stream(agent.stream("go"))

            call_kwargs = mock_litellm.completion.call_args.kwargs
            assert call_kwargs.get("model") == "openai/responses/gpt-5", (
                f"OpenAI reasoning model should route through Responses bridge; "
                f"got {call_kwargs.get('model')}"
            )
            assert (
                call_kwargs.get("extra_body", {}).get("reasoning", {}).get("effort")
                == "high"
            )
            assert "reasoning_effort" not in call_kwargs, (
                "litellm silently drops top-level reasoning_effort on the "
                "Responses path — must not be set"
            )

    def test_stream_no_effort_passes_no_reasoning_kwargs(self, monkeypatch):
        """An agent without reasoning_effort must not inject any reasoning
        kwargs into the litellm call."""
        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(content_fragments=["hi"])
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            mock_litellm.get_llm_provider.return_value = (
                "gpt-4o-mini",
                "openai",
                None,
                None,
            )
            agent = Agent(model="gpt-4o-mini", system_prompt="hi")
            _consume_stream(agent.stream("go"))

            call_kwargs = mock_litellm.completion.call_args.kwargs
            assert "reasoning_effort" not in call_kwargs
            assert "extra_body" not in call_kwargs

    def test_stream_aborts_on_generator_close(self, monkeypatch):
        """N-1 + N-2: closing the generator (e.g. SSE client disconnect)
        sets the context's abort_signal so the underlying LLM stream is
        signalled to stop, rather than leaking the worker thread."""
        import threading as _threading

        from agentic.execution.context import ExecutionContext

        monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
        chunks = _make_streaming_chunks(
            content_fragments=["a", "b", "c", "d", "e"],
        )
        with patch("agentic.agent.agent.litellm") as mock_litellm:
            mock_litellm.completion.return_value = iter(chunks)
            agent = Agent(system_prompt="hi")
            abort_event = _threading.Event()
            gen = agent.stream(
                "go",
                context=ExecutionContext(abort_signal=abort_event),
            )
            next(gen)
            gen.close()

        assert (
            abort_event.is_set()
        ), "generator close should propagate to the context's abort_signal"


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
