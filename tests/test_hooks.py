# agentic/tests/test_hooks.py
from unittest.mock import MagicMock, patch

from agentic.agent.hooks import HookConfig, run_hooks


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
        # Different tool name — hook shouldn't fire
        result = run_hooks("PreToolUse", "http_request", {"url": "DROP"}, None, hooks)
        assert result.blocked is False

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
