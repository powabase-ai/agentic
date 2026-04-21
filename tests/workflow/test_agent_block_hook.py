"""Unit test: AgentBlock invokes the on_agent_run_complete services hook."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentic.workflow.block import BlockInput
from agentic.workflow.blocks.agent import AgentBlock


def _make_block(config=None):
    return AgentBlock(config=config or {"model": "gpt-4o-mini", "input": "hello"})


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

        bi = BlockInput(services={"on_agent_run_complete": hook})
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

        bi = BlockInput()  # no services hook
        result = asyncio.run(block.execute(bi))

    assert result.data["output"] == "response"


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

        bi = BlockInput(services={"on_agent_run_complete": bad_hook})
        result = asyncio.run(block.execute(bi))

    assert result.data["output"] == "response"
