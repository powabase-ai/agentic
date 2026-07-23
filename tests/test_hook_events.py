"""Tests for OnRunStart, OnRunComplete, and OnDelegation hook firing points."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic import Agent
from agentic.agent.hooks import HookConfig
from agentic.execution.context import ExecutionContext


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


class TestOnRunStartFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_run_start_hook_blocks_run(self, mock_litellm):
        """An OnRunStart hook with deny should prevent the run from executing."""
        mock_litellm.completion.return_value = _mock_response("Should not reach here")

        hooks = [
            HookConfig(
                event="OnRunStart",
                matcher=None,
                type="rule",
                config={
                    "condition": "message CONTAINS 'blocked'",
                    "action": "deny",
                    "message": "Run blocked at start",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("blocked input", hooks=hooks)

        # If OnRunStart fires and blocks, the LLM should not be called
        assert output.status.value == "failed"
        assert "Run blocked at start" in output.error
        mock_litellm.completion.assert_not_called()

    @patch("agentic.agent.agent.litellm")
    def test_on_run_start_hook_allows(self, mock_litellm):
        """An OnRunStart hook that doesn't match should let the run proceed."""
        mock_litellm.completion.return_value = _mock_response("Hello!")

        hooks = [
            HookConfig(
                event="OnRunStart",
                matcher=None,
                type="rule",
                config={
                    "condition": "message CONTAINS 'blocked'",
                    "action": "deny",
                    "message": "Blocked",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("normal input", hooks=hooks)

        assert output.status.is_success()
        assert output.content == "Hello!"
        mock_litellm.completion.assert_called_once()


class TestOnRunCompleteFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_run_complete_fires_after_success(self, mock_litellm):
        """OnRunComplete hook should fire after a successful run. Since it's fire-and-forget,
        we verify it fires by checking events."""
        mock_litellm.completion.return_value = _mock_response("Done!")

        events = []
        ctx = ExecutionContext(execution_id="test", on_event=lambda e: events.append(e))

        # Use an HTTP hook that we can detect via events (since rule hooks don't emit events).
        # Actually, let's just check that the agent emits an on_run_complete event.
        hooks = [
            HookConfig(
                event="OnRunComplete",
                matcher=None,
                type="rule",
                config={},  # Empty rule — won't deny anything
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="test")
        output = agent.run("hello", context=ctx, hooks=hooks)

        assert output.status.is_success()
        # Check that an on_run_complete event was emitted
        event_types = [e["type"] for e in events]
        assert "on_run_complete" in event_types


class TestOnDelegationFiring:
    @patch("agentic.agent.agent.litellm")
    def test_on_delegation_hook_blocks_delegation(self, mock_litellm):
        """An OnDelegation hook should fire before DelegateTool executes."""
        # Orchestrator calls delegate_to_specialist
        tool_call_resp = SimpleNamespace(
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
                                    name="delegate_to_specialist",
                                    arguments='{"task": "delete everything"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        final_resp = _mock_response("Delegation was blocked.")
        mock_litellm.completion.side_effect = [tool_call_resp, final_resp]

        from agentic.agent.tools import DelegateTool

        sub_agent = Agent(model="gpt-4o-mini", system_prompt="specialist")
        delegate = DelegateTool(
            name="delegate_to_specialist",
            description="Delegate to specialist",
            agent=sub_agent,
        )

        hooks = [
            HookConfig(
                event="OnDelegation",
                matcher="delegate_to_specialist",
                type="rule",
                config={
                    "condition": "task CONTAINS 'delete'",
                    "action": "deny",
                    "message": "Dangerous delegation blocked",
                },
            )
        ]

        agent = Agent(model="gpt-4o-mini", system_prompt="orchestrator")
        output = agent.run(
            "do it", tools={"delegate_to_specialist": delegate}, hooks=hooks
        )

        assert output.status.is_success()
        # The delegation should have been blocked — sub_agent's LLM should NOT have been called
        # (only 2 calls: orchestrator's tool_call + orchestrator's final response)
        assert mock_litellm.completion.call_count == 2


class TestHookResultEventEmission:
    @patch("agentic.agent.hooks.requests.post")
    def test_preresponse_modify_emits_hook_result_event(self, mock_post, mock_litellm):
        # Webhook rewrites the final answer.
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "modify", "modified_output": "REDACTED"}
            ),
        )
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        agent = Agent(model="gpt-5.4", system_prompt="s", name="a")
        output = agent.run("hello", hooks=hooks)

        assert output.content == "REDACTED"
        hook_results = [e for e in output.events if e.get("type") == "hook_result"]
        assert any(
            e["hook_event"] == "PreResponse" and e["modified"] is True
            for e in hook_results
        )
        # The lifecycle event must live under `hook_event`, never `event`
        # (an `event` key would clobber the SSE envelope label downstream).
        assert all("event" not in e for e in hook_results)

    @patch("agentic.agent.hooks.requests.post")
    def test_onrunstart_allow_emits_hook_result_event(self, mock_post, mock_litellm):
        # Webhook allows the run to proceed (non-blocking).
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"action": "allow"}),
        )
        hooks = [
            HookConfig(
                event="OnRunStart",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h0",
                position=0,
            )
        ]
        agent = Agent(model="gpt-5.4", system_prompt="s", name="a")
        output = agent.run("hello", hooks=hooks)

        hook_results = [e for e in output.events if e.get("type") == "hook_result"]
        assert any(e["hook_event"] == "OnRunStart" for e in hook_results)


class TestFalsyModificationAppliedEndToEnd:
    """R5-C1 at the executor: a PreResponse hook that redacts the answer to the
    empty string must actually suppress it.

    This is the PR's motivating use case — a tenant webhook that finds every
    citation unverifiable and withholds the answer entirely. Truthiness at the
    apply site shipped the ORIGINAL text while the emitted hook_result claimed
    ``modified: true``, so downstream consumers saw an audit trail asserting a
    redaction that never happened.
    """

    @patch("agentic.agent.hooks.requests.post")
    def test_preresponse_empty_output_suppresses_answer(self, mock_post, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "The claimant's SSN is 123-45-6789."
        )
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"action": "modify", "modified_output": ""}),
        )
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        agent = Agent(model="gpt-5.4", system_prompt="s", name="a")
        output = agent.run("hello", hooks=hooks)

        assert output.content == "", (
            "The hook redacted the answer to empty; the original text must not "
            f"survive. Got: {output.content!r}"
        )
        assert "123-45-6789" not in (output.content or "")
        # And the audit record agrees with what actually happened.
        hook_results = [e for e in output.events if e.get("type") == "hook_result"]
        assert any(
            e["hook_event"] == "PreResponse" and e["modified"] is True
            for e in hook_results
        )


def _tool_call_response(tool_name, arguments):
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
                                name=tool_name, arguments=arguments
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestEveryFireSiteEmitsAuditRecords:
    """R5-I7: each hook fire site must emit its executions.

    Only the PreResponse and OnRunStart sites were pinned; deleting the
    _emit_hook_executions call at the other four left every test green. The
    regression that ships is silent and security-relevant: a PreToolUse rule
    hook correctly denies a destructive tool call, but the run's event stream
    carries no hook_result — the audit trail shows a clean run.
    """

    def _hook_results(self, output):
        return [e for e in output.events if e.get("type") == "hook_result"]

    @patch("agentic.agent.agent.litellm")
    def test_pretooluse_block_is_audited(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _tool_call_response("database_query", '{"query": "DROP TABLE users"}'),
            _mock_response("I could not run that."),
        ]
        from agentic.agent.tools import BuiltinTool

        tool = BuiltinTool(
            name="database_query",
            description="Run a query",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args, ctx: "rows",
        )
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "Destructive query blocked",
                },
                id="h-gate",
                position=0,
            )
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
        output = agent.run("drop it", tools={"database_query": tool}, hooks=hooks)

        blocked = [
            e
            for e in self._hook_results(output)
            if e["hook_event"] == "PreToolUse" and e["blocked"] is True
        ]
        assert blocked, (
            "A PreToolUse gate denied a destructive call but emitted no audit "
            "record — the run looks clean to anyone reading its events."
        )
        assert blocked[0]["hook_id"] == "h-gate"

    @patch("agentic.agent.agent.litellm")
    def test_posttooluse_is_audited(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _tool_call_response("lookup", "{}"),
            _mock_response("done"),
        ]
        from agentic.agent.tools import BuiltinTool

        tool = BuiltinTool(
            name="lookup",
            description="Look up",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args, ctx: "raw result",
        )
        hooks = [
            HookConfig(
                event="PostToolUse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h-post",
                position=0,
            )
        ]
        with patch("agentic.agent.hooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200, json=MagicMock(return_value={"action": "allow"})
            )
            agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
            output = agent.run("go", tools={"lookup": tool}, hooks=hooks)

        assert any(
            e["hook_event"] == "PostToolUse" for e in self._hook_results(output)
        ), "PostToolUse fired but emitted no audit record."

    @patch("agentic.agent.agent.litellm")
    def test_onruncomplete_is_audited(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("done")
        hooks = [
            HookConfig(
                event="OnRunComplete",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h-complete",
                position=0,
            )
        ]
        with patch("agentic.agent.hooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200, json=MagicMock(return_value={"action": "allow"})
            )
            agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
            output = agent.run("go", hooks=hooks)

        assert any(
            e["hook_event"] == "OnRunComplete" for e in self._hook_results(output)
        ), "OnRunComplete fired but emitted no audit record."

    @patch("agentic.agent.agent.litellm")
    def test_ondelegation_block_is_audited(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _tool_call_response("delegate_to_specialist", '{"task": "delete all"}'),
            _mock_response("blocked"),
        ]
        from agentic.agent.tools import DelegateTool

        delegate = DelegateTool(
            name="delegate_to_specialist",
            description="Delegate",
            agent=Agent(model="gpt-4o-mini", system_prompt="specialist"),
        )
        hooks = [
            HookConfig(
                event="OnDelegation",
                matcher="delegate_to_specialist",
                type="rule",
                config={
                    "condition": "task CONTAINS 'delete'",
                    "action": "deny",
                    "message": "blocked",
                },
                id="h-deleg",
                position=0,
            )
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
        output = agent.run(
            "go", tools={"delegate_to_specialist": delegate}, hooks=hooks
        )

        assert any(
            e["hook_event"] == "OnDelegation" and e["blocked"] is True
            for e in self._hook_results(output)
        ), "OnDelegation denied a delegation but emitted no audit record."

    @patch("agentic.agent.agent.litellm")
    def test_onrunstart_block_still_returns_events(self, mock_litellm):
        """The blocked-run early return must carry the audit trail with it."""
        mock_litellm.completion.return_value = _mock_response("unreachable")
        hooks = [
            HookConfig(
                event="OnRunStart",
                type="rule",
                config={
                    "condition": "message CONTAINS 'forbidden'",
                    "action": "deny",
                    "message": "nope",
                },
                id="h-start",
                position=0,
            )
        ]
        agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
        output = agent.run("forbidden thing", hooks=hooks)

        assert not output.status.is_success()
        assert any(
            e["hook_event"] == "OnRunStart" and e["blocked"] is True
            for e in self._hook_results(output)
        ), (
            "The OnRunStart early return dropped `events`, so the only record of "
            "why the run was refused is gone."
        )


class TestFalsyModificationAtToolSites:
    """R6-I2: the round-5 `is not None` fix was only pinned at PreResponse.

    Both tool-pipeline apply sites reverted cleanly to truthiness with the whole
    suite green. Each drops a sanitization the tenant asked for while still
    emitting `modified: true` — the same audit-lies-about-a-redaction class as
    R5-C1, at two more sites.
    """

    @patch("agentic.agent.agent.litellm")
    def test_pretooluse_empty_dict_strips_arguments(self, mock_litellm):
        """A webhook stripping every argument off a dangerous call."""
        mock_litellm.completion.side_effect = [
            _tool_call_response("database_query", '{"query": "DROP TABLE users"}'),
            _mock_response("done"),
        ]
        from agentic.agent.tools import BuiltinTool

        seen = {}

        def handler(args, ctx):
            seen["args"] = args
            return "rows"

        tool = BuiltinTool(
            name="database_query",
            description="Run a query",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )
        hooks = [
            HookConfig(
                event="PreToolUse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        with patch("agentic.agent.hooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"action": "modify", "modified_input": {}}),
            )
            agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
            agent.run("go", tools={"database_query": tool}, hooks=hooks)

        assert seen["args"] == {}, (
            "The hook stripped every argument; dropping that sanitization runs "
            f"the tool with the original dangerous input. Got: {seen.get('args')!r}"
        )

    @patch("agentic.agent.agent.litellm")
    def test_posttooluse_empty_string_scrubs_output(self, mock_litellm):
        """A PII scrubber suppressing tool output entirely."""
        mock_litellm.completion.side_effect = [
            _tool_call_response("lookup", "{}"),
            _mock_response("done"),
        ]
        from agentic.agent.tools import BuiltinTool

        tool = BuiltinTool(
            name="lookup",
            description="Look up",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args, ctx: "SSN 123-45-6789",
        )
        hooks = [
            HookConfig(
                event="PostToolUse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        with patch("agentic.agent.hooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={"action": "modify", "modified_output": ""}
                ),
            )
            agent = Agent(model="gpt-4o-mini", system_prompt="s", name="a")
            output = agent.run("go", tools={"lookup": tool}, hooks=hooks)

        tool_messages = [m for m in (output.messages or []) if m.get("role") == "tool"]
        assert tool_messages, "expected a tool result message"
        assert all(
            "123-45-6789" not in str(m.get("content", "")) for m in tool_messages
        ), (
            "The PII scrubber returned an empty replacement; dropping it feeds "
            f"the raw tool output back into the model. Got: {tool_messages}"
        )
