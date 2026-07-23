# agentic/tests/test_hooks.py
from unittest.mock import MagicMock, patch

import pytest
import requests

from agentic.agent.hooks import HookConfig, HookExecution, run_hooks
from agentic.execution.context import ExecutionContext


class TestInlineRuleHook:
    def test_deny_on_match(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            )
        ]
        result = run_hooks(
            "PreToolUse", "database_query", {"query": "DROP TABLE users"}, None, hooks
        )
        assert result.blocked is True
        assert result.message == "No DROP"

    def test_allow_on_no_match(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            )
        ]
        result = run_hooks(
            "PreToolUse", "database_query", {"query": "SELECT 1"}, None, hooks
        )
        assert result.blocked is False

    def test_matcher_filters_by_tool_name(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            )
        ]
        # Different tool name — the hook must not fire. The payload deliberately
        # uses the field the rule reads AND a value it would deny on, so the
        # assertion can only pass because the matcher filtered the hook out.
        result = run_hooks("PreToolUse", "http_request", {"query": "DROP"}, None, hooks)
        assert result.blocked is False
        assert result.executions == [], (
            "A filtered-out hook must not produce an execution record either."
        )

    def test_no_matcher_applies_to_all(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher=None,
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
            )
        ]
        result = run_hooks("PreToolUse", "any_tool", {"query": "DROP"}, None, hooks)
        assert result.blocked is True


class TestHttpHook:
    @patch("agentic.agent.hooks.requests.post")
    def test_http_hook_allow(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"action": "allow"}),
        )
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher=None,
                type="http",
                config={"url": "https://example.com/hook", "timeout_seconds": 5},
            )
        ]
        result = run_hooks("PreToolUse", "tool", {"arg": "val"}, None, hooks)
        assert result.blocked is False
        mock_post.assert_called_once()

    @patch("agentic.agent.hooks.requests.post")
    def test_http_hook_deny(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "deny", "message": "Blocked by webhook"}
            ),
        )
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher=None,
                type="http",
                config={"url": "https://example.com/hook"},
            )
        ]
        result = run_hooks("PreToolUse", "tool", {"arg": "val"}, None, hooks)
        assert result.blocked is True
        assert result.message == "Blocked by webhook"

    @patch("agentic.agent.hooks.requests.post")
    def test_http_hook_modify_input(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "allow", "modified_input": {"arg": "modified"}}
            ),
        )
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher=None,
                type="http",
                config={"url": "https://example.com/hook"},
            )
        ]
        result = run_hooks("PreToolUse", "tool", {"arg": "val"}, None, hooks)
        assert result.blocked is False
        assert result.modified_input == {"arg": "modified"}

    @patch("agentic.agent.hooks.requests.post")
    def test_http_hook_timeout_fails_open(self, mock_post):
        mock_post.side_effect = Exception("Connection timeout")
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher=None,
                type="http",
                config={"url": "https://example.com/hook", "timeout_seconds": 1},
            )
        ]
        result = run_hooks("PreToolUse", "tool", {"arg": "val"}, None, hooks)
        assert result.blocked is False  # Fail-open


class TestEmptyHooks:
    def test_no_hooks(self):
        result = run_hooks("PreToolUse", "tool", {}, None, [])
        assert result.blocked is False

    def test_no_matching_hooks(self):
        hooks = [HookConfig(event="PostToolUse", matcher=None, type="rule", config={})]
        result = run_hooks("PreToolUse", "tool", {}, None, hooks)
        assert result.blocked is False


class TestHookExecutionRecords:
    @patch("agentic.agent.hooks.requests.post")
    def test_http_status_succeeded(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"action": "allow"})
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
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert len(result.executions) == 1
        ex = result.executions[0]
        assert isinstance(ex, HookExecution)
        assert ex.hook_id == "h1"
        assert ex.position == 0
        assert ex.event == "PreResponse"
        assert ex.status == "succeeded"
        assert ex.modified is False
        assert ex.blocked is False

    @patch("agentic.agent.hooks.requests.post")
    def test_http_status_failed_on_non_200(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.blocked is False  # fail-open
        assert result.executions[0].status == "failed"

    @patch("agentic.agent.hooks.requests.post")
    def test_http_status_timed_out(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("slow")
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook", "timeout_seconds": 1},
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.blocked is False  # fail-open
        assert result.executions[0].status == "timed_out"

    @patch("agentic.agent.hooks.requests.post")
    def test_run_block_absent_by_default(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"action": "allow"})
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
        ctx = ExecutionContext(orchestration_run_id="orch-123")
        run_hooks("PreResponse", "", {}, "answer", hooks, context=ctx)
        body = mock_post.call_args.kwargs["json"]
        assert "run" not in body

    @patch("agentic.agent.hooks.requests.post")
    def test_run_block_included_when_opted_in(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"action": "allow"})
        )
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook", "include_run_context": True},
                id="h1",
                position=0,
            )
        ]
        ctx = ExecutionContext(orchestration_run_id="orch-123")
        run_hooks("PreResponse", "", {}, "answer", hooks, context=ctx)
        body = mock_post.call_args.kwargs["json"]
        assert body["run"]["orchestration_run_id"] == "orch-123"
        assert "execution_id" in body["run"]
        # parent_run_id is deliberately absent — hooks are supervisor-only, so it
        # would be null in every payload this feature can produce.
        assert "parent_run_id" not in body["run"]

    def test_executions_preserved_through_block_short_circuit(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "No DROP",
                },
                id="h1",
                position=0,
            ),
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'safe'",
                    "action": "deny",
                    "message": "Blocked safe query",
                },
                id="h2",
                position=1,
            ),
            # A third hook AFTER the blocker: without the short-circuit `return`
            # this one also runs and executions has 3 entries.
            HookConfig(
                event="PreToolUse",
                matcher="database_query",
                type="rule",
                config={
                    "condition": "query CONTAINS 'never'",
                    "action": "deny",
                    "message": "unreachable",
                },
                id="h3",
                position=2,
            ),
        ]
        result = run_hooks(
            "PreToolUse", "database_query", {"query": "safe query"}, None, hooks
        )
        assert result.blocked is True
        assert len(result.executions) == 2, (
            "The chain must stop at the blocking hook; h3 ran anyway. "
            f"Got: {[e.hook_id for e in result.executions]}"
        )
        assert [e.hook_id for e in result.executions] == ["h1", "h2"]
        assert result.executions[0].hook_id == "h1"
        assert result.executions[0].blocked is False
        assert result.executions[1].hook_id == "h2"
        assert result.executions[1].blocked is True


class TestHookExecutionRobustness:
    """Review fixes: C3a (unknown type), Imp1-5 (execution-record accuracy + fail-open)."""

    def test_unknown_type_records_failed_execution(self):
        # C3a: an unknown hook type must not silently vanish.
        hooks = [
            HookConfig(
                event="PreResponse", type="webhook", config={}, id="h1", position=0
            )
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert len(result.executions) == 1
        assert result.executions[0].status == "failed"
        assert "unknown hook type" in (result.executions[0].message or "").lower()

    @patch("agentic.agent.hooks.requests.post")
    def test_modified_reflects_modified_input(self, mock_post):
        # Imp1: a hook that rewrites tool args (modified_input) is not a no-op.
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "allow", "modified_input": {"x": 1}}
            ),
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
        result = run_hooks("PreToolUse", "t", {"x": 0}, None, hooks)
        assert result.executions[0].modified is True

    def test_non_dict_config_fails_open_and_preserves_prior_executions(self):
        # Imp2: a malformed config must not crash the run or discard earlier executions.
        hooks = [
            HookConfig(
                event="PreResponse",
                type="rule",
                config={"condition": "output CONTAINS 'zzz'", "action": "deny"},
                id="ok",
                position=0,
            ),
            HookConfig(
                event="PreResponse", type="http", config=[], id="bad", position=1
            ),
        ]
        # Must not raise despite the non-dict config on the second hook.
        result = run_hooks("PreResponse", "", {}, "clean answer", hooks)
        assert len(result.executions) == 2
        assert result.executions[0].hook_id == "ok"  # earlier execution preserved
        assert result.executions[1].hook_id == "bad"
        assert result.executions[1].status == "failed"

    def test_url_less_http_hook_is_failed_not_succeeded(self):
        # Imp3: a misconfigured http hook (no url) must not report a false green.
        hooks = [
            HookConfig(event="PreResponse", type="http", config={}, id="h1", position=0)
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.executions[0].status == "failed"

    @patch("agentic.agent.hooks.requests.post")
    def test_message_populated_on_non_200(self, mock_post):
        # Imp4: the persisted execution must distinguish failure kinds.
        mock_post.return_value = MagicMock(status_code=503)
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.executions[0].status == "failed"
        assert "503" in (result.executions[0].message or "")

    def test_approval_timeout_status_is_timed_out(self):
        # Imp5: approval timeouts must not be recorded as "succeeded".
        ctx = MagicMock()
        ctx.get_approval_decision.return_value = None  # never decided
        ctx.get_approval_event.return_value = MagicMock()
        hooks = [
            HookConfig(
                event="PreToolUse",
                type="approval",
                config={"timeout": 0},
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreToolUse", "t", {}, None, hooks, context=ctx)
        assert result.blocked is True
        assert result.executions[0].status == "timed_out"


class TestSSRFFailOpen:
    def test_ssrf_blocked_url_fails_open(self):
        # #2: an SSRF-blocked URL is a hook that can't be called — treat it like
        # every other http failure (fail-open + status=failed), not a run-abort.
        from agentic.agent.url_validation import SSRFError

        with patch("agentic.agent.url_validation.validate_url") as mock_validate:
            mock_validate.side_effect = SSRFError("blocked internal address")
            hooks = [
                HookConfig(
                    event="OnRunStart",
                    type="http",
                    config={"url": "http://localhost:9000/hook"},
                    id="h1",
                    position=0,
                )
            ]
            result = run_hooks("OnRunStart", "", {"message": "hi"}, None, hooks)
        assert result.blocked is False
        assert result.executions[0].status == "failed"


class TestGatingHookFailClosed:
    """F2: a rule/approval hook that RAISES must fail CLOSED (block), not allow;
    http (observability/modify) stays fail-open."""

    def test_gating_rule_dispatch_exception_fails_closed(self):
        hooks = [
            HookConfig(
                event="PreToolUse",
                type="rule",
                config={"rules": "not-a-list"},  # raises inside evaluate_rules
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreToolUse", "t", {"x": 1}, None, hooks)
        assert result.executions[0].status == "failed"
        assert result.executions[0].blocked is True
        assert result.blocked is True

    def test_http_dispatch_exception_stays_fail_open(self):
        hooks = [
            HookConfig(
                event="PreResponse",
                type="http",
                config=[],  # non-dict → raises at config.get, caught in run_hooks
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.executions[0].status == "failed"
        assert result.executions[0].blocked is False
        assert result.blocked is False


class TestNonBlockingEventAuditHonesty:
    """N1: PostToolUse/OnRunComplete cannot block per the documented contract —
    the executor discards `blocked`, so the audit record must not claim a block
    and the hook chain must not short-circuit on one."""

    def test_block_on_non_blocking_event_not_recorded_and_chain_continues(self):
        hooks = [
            HookConfig(
                event="PostToolUse",
                type="rule",
                config={
                    "condition": "output CONTAINS 'x'",
                    "action": "deny",
                    "message": "no",
                },
                id="h1",
                position=0,
            ),
            HookConfig(
                event="PostToolUse",
                type="rule",
                config={"condition": "output CONTAINS 'zzz'", "action": "deny"},
                id="h2",
                position=1,
            ),
        ]
        result = run_hooks("PostToolUse", "t", {}, "x marks", hooks)
        assert result.blocked is False  # executor ignores it anyway
        assert result.executions[0].blocked is False  # audit stays honest
        assert len(result.executions) == 2  # chain did not short-circuit

    def test_block_on_blocking_event_still_recorded(self):
        # Counterpart: PreToolUse CAN block — unchanged.
        hooks = [
            HookConfig(
                event="PreToolUse",
                type="rule",
                config={
                    "condition": "q CONTAINS 'DROP'",
                    "action": "deny",
                    "message": "no",
                },
                id="h1",
                position=0,
            )
        ]
        result = run_hooks("PreToolUse", "t", {"q": "DROP TABLE"}, None, hooks)
        assert result.blocked is True
        assert result.executions[0].blocked is True


class TestApprovalDecisionStatuses:
    """Coverage: only the approval *timeout* status was pinned; a completed
    deny must record status="succeeded" (the hook ran) + blocked=True."""

    def test_approval_denied_is_succeeded_and_blocked(self):
        ctx = MagicMock()
        ctx.get_approval_decision.return_value = {"approved": False, "reason": "nope"}
        hooks = [
            HookConfig(
                event="PreToolUse", type="approval", config={}, id="h1", position=0
            )
        ]
        result = run_hooks("PreToolUse", "t", {}, None, hooks, context=ctx)
        assert result.blocked is True
        assert result.executions[0].status == "succeeded"
        assert result.executions[0].blocked is True

    def test_approval_approved_is_succeeded_and_not_blocked(self):
        ctx = MagicMock()
        ctx.get_approval_decision.return_value = {"approved": True}
        hooks = [
            HookConfig(
                event="PreToolUse", type="approval", config={}, id="h1", position=0
            )
        ]
        result = run_hooks("PreToolUse", "t", {}, None, hooks, context=ctx)
        assert result.blocked is False
        assert result.executions[0].status == "succeeded"


class TestFalsyModificationsAreApplied:
    """R5-C1: a hook may legitimately return a *falsy* modification.

    `modified_output: ""` means "replace the answer with nothing" (full
    redaction / suppression) — semantically distinct from omitting the key,
    which means "I changed nothing". Truthiness testing collapses the two and
    silently drops the redaction while the audit record still claims
    ``modified=True``. The audit flag and the apply decision must agree.
    """

    @patch("agentic.agent.hooks.requests.post")
    def test_empty_string_modified_output_is_applied(self, mock_post):
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
        result = run_hooks("PreResponse", "", {}, "secret answer", hooks)
        assert result.modified_output == "", (
            "A full redaction (empty string) must propagate to the caller; "
            "dropping it ships the unredacted answer."
        )
        # The audit record must not claim a modification the executor discarded.
        assert result.executions[0].modified is True

    @patch("agentic.agent.hooks.requests.post")
    def test_empty_dict_modified_input_is_applied(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"action": "modify", "modified_input": {}}),
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
        result = run_hooks("PreToolUse", "search", {"q": "secret"}, None, hooks)
        assert result.modified_input == {}, (
            "Stripping every argument is a valid modification; it must not be "
            "silently ignored."
        )
        assert result.executions[0].modified is True

    @patch("agentic.agent.hooks.requests.post")
    def test_absent_modification_stays_none(self, mock_post):
        """The other half of the contract: omitted keys must NOT be applied."""
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"action": "allow"})
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
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.modified_output is None
        assert result.modified_input is None
        assert result.executions[0].modified is False


class TestNonModifyingEventAuditHonesty:
    """R5-I6: three events discard modifications entirely.

    OnRunStart and OnDelegation consult only ``blocked``; OnRunComplete is pure
    fire-and-forget. A hook returning ``modified_output`` at those events has no
    effect, so the audit record must not claim ``modified=True`` — the same
    honesty rule already applied to ``blocked`` on non-blocking events.
    """

    @patch("agentic.agent.hooks.requests.post")
    @pytest.mark.parametrize("event", ["OnRunStart", "OnDelegation", "OnRunComplete"])
    def test_modification_on_non_modifying_event_not_claimed(self, mock_post, event):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "modify", "modified_output": "ignored"}
            ),
        )
        hooks = [
            HookConfig(
                event=event,
                type="http",
                config={"url": "https://example.com/hook"},
                id="h1",
                position=0,
            )
        ]
        result = run_hooks(event, "", {}, None, hooks)
        assert result.executions[0].modified is False, (
            f"{event} discards modifications; the audit trail must not claim one."
        )
        assert result.modified_output is None
        # The hook still ran and reported its status honestly.
        assert result.executions[0].status == "succeeded"

    @patch("agentic.agent.hooks.requests.post")
    def test_modification_on_modifying_event_still_claimed(self, mock_post):
        """Control: PreResponse DOES apply modifications, so it still reports."""
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
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.executions[0].modified is True
        assert result.modified_output == "REDACTED"


class TestDeadRuleConfigIsHonest:
    """R5-I2: a rule hook with no evaluable rules allows everything.

    ``{"conditon": ...}`` (typo) or ``{"rules": []}`` resolves to an empty rule
    list, which `evaluate_rules` allows. Reporting that as ``succeeded`` makes a
    dead gate indistinguishable from a gate that genuinely passed.
    """

    @pytest.mark.parametrize(
        "config",
        [{}, {"rules": []}, {"conditon": "typo CONTAINS 'x'", "action": "deny"}],
        ids=["empty", "empty-rules", "typo'd-condition"],
    )
    def test_no_evaluable_rules_reports_failed(self, config):
        hooks = [
            HookConfig(
                event="PreToolUse", type="rule", config=config, id="h1", position=0
            )
        ]
        result = run_hooks("PreToolUse", "database_query", {"q": "DROP"}, None, hooks)
        ex = result.executions[0]
        assert ex.status == "failed", (
            "A rule hook that cannot evaluate anything must not report a clean "
            "pass — it is indistinguishable from a working gate in the audit log."
        )
        assert ex.message == "rule hook has no evaluable rules"
        # Deliberately NOT blocked: see the comment in _execute_rule_hook.
        assert result.blocked is False

    def test_real_rule_still_succeeds(self):
        """Control: a well-formed rule that passes still reports succeeded."""
        hooks = [
            HookConfig(
                event="PreToolUse",
                type="rule",
                config={"condition": "q CONTAINS 'DROP'", "action": "deny"},
                id="h1",
                position=0,
            )
        ]
        allowed = run_hooks("PreToolUse", "t", {"q": "SELECT 1"}, None, hooks)
        assert allowed.executions[0].status == "succeeded"
        assert allowed.blocked is False

        denied = run_hooks("PreToolUse", "t", {"q": "DROP TABLE x"}, None, hooks)
        assert denied.blocked is True


class TestModifiedIsFieldSpecific:
    """R6-I1: `modified` must reflect the field the call site actually consumes.

    The PreResponse call site reads only `modified_output`. A hook returning
    just `modified_input` changes nothing there — but a field-agnostic
    `modified` flag trips `_preresponse_edited()`, and both routes then swap the
    persisted answer from the streamed buffer to `output.content`, silently
    dropping every earlier step of a multi-step run.
    """

    @patch("agentic.agent.hooks.requests.post")
    def test_preresponse_ignores_modified_input(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "modify", "modified_input": {"q": "x"}}
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
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.executions[0].modified is False, (
            "PreResponse consumes only modified_output; claiming modified=True "
            "for a modified_input-only hook makes the routes discard the "
            "streamed answer and persist a truncated one."
        )
        assert result.modified_output is None
        assert result.modified_input is None

    @patch("agentic.agent.hooks.requests.post")
    def test_pretooluse_ignores_modified_output(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"action": "modify", "modified_output": "nope"}
            ),
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
        result = run_hooks("PreToolUse", "t", {"q": "a"}, None, hooks)
        assert result.executions[0].modified is False
        assert result.modified_output is None

    @patch("agentic.agent.hooks.requests.post")
    def test_consumed_field_still_reported(self, mock_post):
        """Control: each event still honors the field it does consume."""
        for event, body, attr in [
            ("PreResponse", {"modified_output": "R"}, "modified_output"),
            ("PostToolUse", {"modified_output": "R"}, "modified_output"),
            ("PreToolUse", {"modified_input": {"q": "z"}}, "modified_input"),
        ]:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"action": "modify", **body}),
            )
            hooks = [
                HookConfig(
                    event=event,
                    type="http",
                    config={"url": "https://example.com/hook"},
                    id="h1",
                    position=0,
                )
            ]
            result = run_hooks(event, "t", {"q": "a"}, "answer", hooks)
            assert result.executions[0].modified is True, event
            assert getattr(result, attr) == body[attr], event


class TestDenyCarriesReplacement:
    """R6-I3: a `deny` that supplies a replacement must not lose both.

    On a non-blocking event the deny is inert-ified by design (N1b) — but the
    replacement text the tenant sent alongside it was also discarded, so a
    PostToolUse PII gate wired as `deny` was a complete no-op reported green.
    """

    @patch("agentic.agent.hooks.requests.post")
    def test_deny_with_replacement_degrades_to_redaction(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "action": "deny",
                    "message": "PII detected",
                    "modified_output": "[REDACTED]",
                }
            ),
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
        result = run_hooks("PostToolUse", "lookup", {}, "SSN 123-45-6789", hooks)
        assert result.modified_output == "[REDACTED]", (
            "PostToolUse cannot block, so a deny must degrade to the redaction "
            "the hook supplied rather than silently doing nothing."
        )
        assert result.blocked is False
        assert result.executions[0].modified is True

    @patch("agentic.agent.hooks.requests.post")
    def test_deny_on_blocking_event_does_not_claim_modification(self, mock_post):
        """The block wins at every call site, so the replacement is NOT applied
        — and the audit record must not claim otherwise."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "action": "deny",
                    "message": "nope",
                    "modified_output": "[REDACTED]",
                }
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
        result = run_hooks("PreResponse", "", {}, "answer", hooks)
        assert result.blocked is True
        assert result.executions[0].blocked is True
        assert result.executions[0].modified is False, (
            "A blocked hook's replacement is never applied; claiming "
            "modified=True re-creates the audit-lies-about-a-redaction bug."
        )
        assert result.modified_output is None
