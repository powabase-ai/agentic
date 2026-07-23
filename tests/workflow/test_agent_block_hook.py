"""Unit test: AgentBlock invokes the on_agent_run_complete services hook."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.execution.status import ExecutionStatus
from agentic.workflow.block import BlockInput
from agentic.workflow.blocks.agent import AgentBlock


def _make_block(config=None):
    return AgentBlock(config=config or {"model": "gpt-4o-mini", "input": "hello"})


def _resolver(model: str) -> str | None:
    """Stub resolver that returns None (BYOK not configured in unit tests).

    IMP-NEW-2: AgentBlock.execute() / .stream() now require 'resolve_agent_api_key'
    in services (fail-closed). All BlockInput() calls must include it.
    """
    return None


def test_agent_block_calls_services_hook_after_run():
    called = {}

    def hook(payload):
        called.update(payload)

    fake_output = MagicMock()
    fake_output.content = "response"
    fake_output.usage = {"total_tokens": 5}

    block = _make_block()
    # Patch Agent.arun on the instance built inside execute.
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "agentic.workflow.blocks.agent.Agent"
    ) as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(
            services={"on_agent_run_complete": hook, "resolve_agent_api_key": _resolver}
        )
        result = asyncio.run(block.execute(bi))

    assert result.data["output"] == "response"
    assert called["content"] == "response"
    assert called["usage"] == {"total_tokens": 5}
    assert called["block_id"] is None or isinstance(
        called["block_id"], str | type(None)
    )
    assert called["system_prompt"] is not None
    assert called["prompt"] == "hello"


def test_agent_block_without_hook_works_as_before():
    fake_output = MagicMock()
    fake_output.content = "response"
    fake_output.usage = {}

    block = _make_block()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "agentic.workflow.blocks.agent.Agent"
    ) as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(
            services={"resolve_agent_api_key": _resolver}
        )  # no hook, but resolver required
        result = asyncio.run(block.execute(bi))

    assert result.data["output"] == "response"


def test_agent_block_stream_calls_services_hook_after_run():
    called = {}

    def hook(payload):
        called.update(payload)

    async def fake_astream(prompt):
        yield "hello "
        yield "world"

    block = _make_block()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "agentic.workflow.blocks.agent.Agent"
    ) as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.astream = fake_astream

        bi = BlockInput(
            services={"on_agent_run_complete": hook, "resolve_agent_api_key": _resolver}
        )

        async def _run():
            outputs = []
            async for item in block.stream(bi):
                outputs.append(item)
            return outputs

        outputs = asyncio.run(_run())

    # Last item should be BlockOutput with full content
    from agentic.workflow.block import BlockOutput

    assert isinstance(outputs[-1], BlockOutput)
    assert outputs[-1].data["output"] == "hello world"

    # Hook must have been called with the same payload keys as execute()
    assert called["content"] == "hello world"
    assert called["block_id"] is None or isinstance(
        called["block_id"], str | type(None)
    )
    assert called["system_prompt"] is not None
    assert called["prompt"] == "hello"
    assert called["model"] == "gpt-4o-mini"


def test_agent_block_hook_exceptions_do_not_break_execute():
    def bad_hook(payload):
        raise RuntimeError("boom")

    fake_output = MagicMock()
    fake_output.content = "response"
    fake_output.usage = {}

    block = _make_block()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "agentic.workflow.blocks.agent.Agent"
    ) as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(
            services={
                "on_agent_run_complete": bad_hook,
                "resolve_agent_api_key": _resolver,
            }
        )
        result = asyncio.run(block.execute(bi))

    assert result.data["output"] == "response"


def test_agent_block_execute_passes_status_and_error_from_output():
    """On a failed arun (status=FAILED), the hook payload must carry status+error
    so the host's recorder can persist with AgentRunStatus.FAILED instead of
    silently labelling it COMPLETED."""
    called = {}

    def hook(payload):
        called.update(payload)

    fake_output = MagicMock()
    fake_output.content = None
    fake_output.usage = {}
    fake_output.status = ExecutionStatus.FAILED
    fake_output.error = "LLM backend unavailable"

    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(
            services={"on_agent_run_complete": hook, "resolve_agent_api_key": _resolver}
        )
        asyncio.run(block.execute(bi))

    assert called["status"] == ExecutionStatus.FAILED
    assert called["error"] == "LLM backend unavailable"
    assert called["content"] is None


def test_agent_block_execute_passes_status_completed_on_success():
    called = {}

    def hook(payload):
        called.update(payload)

    fake_output = MagicMock()
    fake_output.content = "ok"
    fake_output.usage = {"total_tokens": 1}
    fake_output.status = ExecutionStatus.COMPLETED
    fake_output.error = None

    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(
            services={"on_agent_run_complete": hook, "resolve_agent_api_key": _resolver}
        )
        asyncio.run(block.execute(bi))

    assert called["status"] == ExecutionStatus.COMPLETED
    assert called["error"] is None


def test_agent_block_stream_fires_hook_on_astream_error_and_re_raises():
    """If Agent.astream raises mid-stream, AgentBlock.stream must still fire
    the persistence hook (with partial content + error) before re-raising so
    the caller's workflow-block log correctly links to a FAILED agent_run."""
    called = {}

    def hook(payload):
        called.update(payload)

    async def failing_astream(prompt):
        yield "partial "
        raise RuntimeError("stream dropped")

    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.astream = failing_astream

        bi = BlockInput(
            services={"on_agent_run_complete": hook, "resolve_agent_api_key": _resolver}
        )

        async def _consume():
            async for _ in block.stream(bi):
                pass

        with pytest.raises(RuntimeError, match="stream dropped"):
            asyncio.run(_consume())

    assert called["status"] == ExecutionStatus.FAILED
    assert called["error"] == "stream dropped"
    assert called["content"] == "partial "


def test_agent_block_stream_hook_exceptions_do_not_suppress_original_error():
    """If the persistence hook itself raises during the failure path, the
    original astream exception must still propagate (hook error is logged)."""

    def bad_hook(payload):
        raise RuntimeError("hook also broken")

    async def failing_astream(prompt):
        yield "early chunk"
        raise RuntimeError("original failure")

    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.astream = failing_astream

        bi = BlockInput(
            services={
                "on_agent_run_complete": bad_hook,
                "resolve_agent_api_key": _resolver,
            }
        )

        async def _consume():
            async for _ in block.stream(bi):
                pass

        with pytest.raises(RuntimeError, match="original failure"):
            asyncio.run(_consume())


def test_agent_block_fails_closed_without_resolver():
    """IMP-NEW-2: AgentBlock.execute() must raise RuntimeError when
    'resolve_agent_api_key' is absent from services.

    This prevents the silent BYOK bypass where a missing injection would
    use api_key=None (an ambient host key) and bypass the host's cost
    attribution for the call.
    """
    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=MagicMock())

        # No resolver → RuntimeError
        bi = BlockInput()  # empty services
        with pytest.raises(RuntimeError, match="resolve_agent_api_key"):
            asyncio.run(block.execute(bi))

        # With hook but no resolver → same RuntimeError
        bi_with_hook = BlockInput(services={"on_agent_run_complete": lambda _: None})
        with pytest.raises(RuntimeError, match="resolve_agent_api_key"):
            asyncio.run(block.execute(bi_with_hook))


def test_agent_block_stream_fails_closed_without_resolver():
    """IMP-NEW-2: same fail-closed check for AgentBlock.stream()."""
    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value

        async def _noop(prompt):
            yield "chunk"

        instance.astream = _noop

        bi = BlockInput()  # no resolver

        async def _consume():
            async for _ in block.stream(bi):
                pass

        with pytest.raises(RuntimeError, match="resolve_agent_api_key"):
            asyncio.run(_consume())


def test_agent_block_execute_passes_api_key_to_agent():
    """IMP-NEW-6: resolver return value must be forwarded as api_key= to Agent().

    Counterfactual: if AgentBlock called Agent() without passing api_key, the
    Agent would fall back to the host's ambient OPENAI_API_KEY instead of
    the user's BYOK key, silently bypassing the host's cost attribution.
    Removing the `api_key=api_key` argument from the _build_agent call causes
    this test to FAIL because FakeAgentCls would be called with api_key=None
    (or absent).
    """
    fake_output = MagicMock()
    fake_output.content = "result"
    fake_output.usage = {}
    fake_output.status = MagicMock()
    fake_output.error = None

    def _byok_resolver(model: str) -> str | None:
        return "sk-user-byok-key"

    block = _make_block()
    with patch("agentic.workflow.blocks.agent.Agent") as FakeAgentCls:
        instance = FakeAgentCls.return_value
        instance.arun = AsyncMock(return_value=fake_output)

        bi = BlockInput(services={"resolve_agent_api_key": _byok_resolver})
        asyncio.run(block.execute(bi))

    # Agent must have been constructed with the key the resolver returned.
    assert FakeAgentCls.called, "Agent was never constructed"
    call_kwargs = FakeAgentCls.call_args.kwargs
    assert (
        "api_key" in call_kwargs
    ), f"Agent() was not called with api_key kwarg; got kwargs={call_kwargs}"
    assert (
        call_kwargs["api_key"] == "sk-user-byok-key"
    ), f"Expected api_key='sk-user-byok-key'; got {call_kwargs['api_key']!r}"
