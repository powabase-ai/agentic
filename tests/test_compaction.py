from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic.agent import compaction
from agentic.agent.compaction import (
    _extract_summary,
    _summary_started,
    compact_messages,
    estimate_token_count,
    prune_messages,
    truncate_messages,
)

_BASE = [
    {"role": "system", "content": "You are a bot."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "Sunny."},
    {"role": "user", "content": "And tomorrow?"},
]


def _mock_response(text, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestEstimateTokenCount:
    def test_simple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = estimate_token_count(messages)
        assert count > 0
        assert count < 100

    def test_empty_messages(self):
        assert estimate_token_count([]) == 0

    def test_none_content(self):
        messages = [{"role": "assistant", "content": None}]
        assert estimate_token_count(messages) == 0


class TestExtractSummary:
    def test_extracts_summary_block(self):
        assert (
            _extract_summary("<analysis>x</analysis><summary>the summary</summary>")
            == "the summary"
        )

    def test_untagged_uses_whole_text(self):
        assert _extract_summary("just a plain summary") == "just a plain summary"

    def test_empty(self):
        assert _extract_summary("") == ""

    def test_open_tag_only_truncated(self):
        # Closing </summary> missing (e.g. output truncated by max_tokens) —
        # must not fall back to the whole text (which would leak <analysis>).
        text = "<analysis>some reasoning</analysis><summary>truncated summary content"
        assert _extract_summary(text) == "truncated summary content"

    def test_no_tags_at_all(self):
        assert _extract_summary("plain text, no tags") == "plain text, no tags"


class TestCompactMessages:
    @patch("agentic.agent.compaction.litellm")
    def test_same_model_api_key_and_cache_preserving_prefix(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "<summary>User asked about weather; assistant said sunny.</summary>"
        )
        compact_messages(
            _BASE, model="openrouter/deepseek/deepseek-v4-pro", api_key="sk-x"
        )
        _, kwargs = mock_litellm.completion.call_args
        assert kwargs["model"] == "openrouter/deepseek/deepseek-v4-pro"
        assert kwargs["api_key"] == "sk-x"
        # prefix preserved for cache: sent == original + one trailing instruction
        sent = kwargs["messages"]
        assert sent[: len(_BASE)] == _BASE
        assert len(sent) == len(_BASE) + 1
        assert sent[-1]["role"] == "user"
        assert sent[-1]["content"] == compaction.COMPACTION_INSTRUCTION

    @patch("agentic.agent.compaction.litellm")
    def test_rebuild_is_system_summary_nudge(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "<summary>User asked about weather; assistant said sunny.</summary>"
        )
        result = compact_messages(_BASE, model="gpt-5.4")
        assert result[0] == {"role": "system", "content": "You are a bot."}
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "User asked about weather; assistant said sunny."
        assert result[-1]["content"] == compaction.CONTINUATION_NUDGE
        assert len(result) == 3

    @patch("agentic.agent.compaction.litellm")
    def test_no_api_key_kwarg_when_none(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(_BASE, model="gpt-5.4")
        _, kwargs = mock_litellm.completion.call_args
        assert "api_key" not in kwargs

    @patch("agentic.agent.compaction.litellm")
    def test_failure_returns_original(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("boom")
        assert compact_messages(_BASE, model="gpt-5.4") == _BASE

    @patch("agentic.agent.compaction.litellm")
    def test_short_history_untouched(self, mock_litellm):
        short = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        assert compact_messages(short, model="gpt-5.4") == short
        mock_litellm.completion.assert_not_called()

    @patch("agentic.agent.compaction.litellm")
    def test_caller_list_not_mutated(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        messages = list(_BASE)
        original = list(messages)
        compact_messages(messages, model="gpt-5.4")
        assert len(messages) == len(original)
        assert messages == original

    @patch("agentic.agent.compaction.litellm")
    def test_rebuild_without_system_message(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "<summary>User asked about weather; assistant said sunny.</summary>"
        )
        no_system = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Sunny."},
            {"role": "user", "content": "And tomorrow?"},
        ]
        result = compact_messages(no_system, model="gpt-5.4")
        assert result == [
            {
                "role": "user",
                "content": "User asked about weather; assistant said sunny.",
            },
            {"role": "user", "content": compaction.CONTINUATION_NUDGE},
        ]
        assert len(result) == 2

    @patch("agentic.agent.compaction.litellm")
    def test_unanswered_tool_calls_returns_original(self, mock_litellm):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = compact_messages(messages, model="gpt-5.4")
        # `is`, not `==`: call sites drive the circuit breaker off an identity
        # check, so returning a copy here would silently disable it.
        assert result is messages
        mock_litellm.completion.assert_not_called()

    @patch("agentic.agent.compaction.litellm")
    def test_no_progress_returns_the_same_list_object(self, mock_litellm):
        # Call sites detect "compaction made no progress" with `is` — every
        # no-progress path must return the caller's own list object.
        mock_litellm.completion.side_effect = Exception("boom")
        messages = list(_BASE)
        assert compact_messages(messages, model="gpt-5.4") is messages

        mock_litellm.completion.side_effect = None
        mock_litellm.completion.return_value = _mock_response("")
        assert compact_messages(messages, model="gpt-5.4") is messages

        short = [{"role": "user", "content": "u"}]
        assert compact_messages(short, model="gpt-5.4") is short

    @patch("agentic.agent.compaction.litellm")
    def test_none_summary_content_returns_original(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(None)
        assert compact_messages(_BASE, model="gpt-5.4") == _BASE

    @patch("agentic.agent.compaction.litellm")
    def test_empty_summary_content_returns_original(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("")
        assert compact_messages(_BASE, model="gpt-5.4") == _BASE


class TestCompactMessagesNormalization:
    """Non-standard message keys must never reach the provider (400 otherwise)."""

    _STANDARD = {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "thinking_blocks",
        "provider_specific_fields",
    }

    @patch("agentic.agent.compaction.litellm")
    def test_injected_key_stripped_before_send(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "q"},
            {
                "role": "system",
                "content": "Budget warning: 80% used.",
                "_injected": True,
            },
            {"role": "assistant", "content": "a"},
        ]
        compact_messages(messages, model="gpt-5.4")
        sent = mock_litellm.completion.call_args.kwargs["messages"]
        for m in sent:
            assert not (set(m) - self._STANDARD), f"non-standard key in {m}"
        assert all("_injected" not in m for m in sent)

    @patch("agentic.agent.compaction.litellm")
    def test_orphan_tool_message_dropped_before_send(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            # tool result with no matching assistant tool_calls — providers 400
            {"role": "tool", "tool_call_id": "gone", "content": "orphan result"},
            {"role": "assistant", "content": "a"},
        ]
        compact_messages(messages, model="gpt-5.4")
        sent = mock_litellm.completion.call_args.kwargs["messages"]
        assert all(m["role"] != "tool" for m in sent)

    @patch("agentic.agent.compaction.litellm")
    def test_rebuilt_result_has_no_injected_keys(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        messages = [
            {"role": "system", "content": "You are a bot.", "_injected": True},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
        ]
        result = compact_messages(messages, model="gpt-5.4")
        for m in result:
            assert not (set(m) - self._STANDARD)


class TestPruneMessages:
    def test_prune_removes_old_tool_results(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "x" * 10000},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "y" * 10000},
            {"role": "assistant", "content": "answer 2"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=1)
        tc1_result = next(m for m in pruned if m.get("tool_call_id") == "tc1")
        assert "[Previous tool result removed" in tc1_result["content"]
        assert len(tc1_result["content"]) < 100
        tc2_result = next(m for m in pruned if m.get("tool_call_id") == "tc2")
        assert tc2_result["content"] == "y" * 10000

    def test_prune_preserves_system_message(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "big result"},
            {"role": "assistant", "content": "done"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=0)
        assert pruned[0]["role"] == "system"
        assert pruned[0]["content"] == "System prompt."

    def test_prune_preserves_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "assistant", "content": "world"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=0)
        user_msgs = [m for m in pruned if m["role"] == "user"]
        assistant_msgs = [m for m in pruned if m["role"] == "assistant"]
        assert len(user_msgs) == 1
        assert len(assistant_msgs) == 2

    def test_prune_no_tool_results_unchanged(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pruned = prune_messages(messages, keep_last_n_turns=1)
        assert pruned == messages


class TestGetContextThreshold:
    def test_delegates_to_resolver_minus_buffers(self):
        with patch.object(compaction, "resolve_context_window", return_value=1_000_000):
            # 1_000_000 - _MAX_OUTPUT_TOKENS(8000) - buffer(8% = 80_000)
            assert compaction.get_context_threshold("any-model") == 912_000

    def test_default_window_threshold(self):
        with patch.object(compaction, "resolve_context_window", return_value=128_000):
            assert compaction.get_context_threshold("unknown") == 107_000


class TestTruncateMessages:
    def test_gets_under_target(self):
        messages = [{"role": "user", "content": "x" * 4000} for _ in range(20)]
        # 20 messages * 1000 tokens each = 20_000
        assert estimate_token_count(messages) == 20_000
        result = truncate_messages(messages, 5_000)
        assert estimate_token_count(result) <= 5_000
        assert len(result) == 5

    def test_keeps_most_recent(self):
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "user", "content": "b" * 400},
            {"role": "user", "content": "c" * 400},
        ]
        result = truncate_messages(messages, 200)
        assert [m["content"][0] for m in result] == ["b", "c"]

    def test_preserves_system_message(self):
        messages = [{"role": "system", "content": "SYS"}] + [
            {"role": "user", "content": "x" * 4000} for _ in range(20)
        ]
        result = truncate_messages(messages, 5_000)
        assert result[0] == {"role": "system", "content": "SYS"}
        assert estimate_token_count(result) <= 5_000

    def test_empty_input(self):
        assert truncate_messages([], 1000) == []

    def test_no_orphan_tool_message_at_boundary(self):
        """Truncating between an assistant tool_calls msg and its tool result
        must not leave the orphaned tool result behind."""
        messages = [
            {"role": "user", "content": "q" * 400},
            {
                # Non-empty content so this message has real token cost and
                # the budget genuinely cuts between it and its tool result.
                "role": "assistant",
                "content": "a" * 400,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "r" * 400},
        ]
        # Budget of 100 tokens fits only the last message (the tool result),
        # which would be an orphan without the guard.
        result = truncate_messages(messages, 100)

        tool_call_ids = {tc["id"] for m in result for tc in (m.get("tool_calls") or [])}
        for m in result:
            if m["role"] == "tool":
                assert m["tool_call_id"] in tool_call_ids
        assert not (result and result[0]["role"] == "tool")


class TestAnalysisLeak:
    """<analysis> scratchpad must never become the conversation history."""

    def test_analysis_only_output_is_not_a_summary(self):
        # The model burned its whole output budget on <analysis> and never
        # opened <summary>. Falling back to the whole text would install the
        # raw scratchpad as the entire conversation.
        text = (
            "<analysis>\n1. Chronologically analyzing each message. The user "
            "first asked about X, then I called tool Y... internal scratchpad"
        )
        assert _extract_summary(text) == ""

    def test_unclosed_analysis_with_leading_whitespace(self):
        assert _extract_summary("\n  <analysis>reasoning goes here") == ""

    def test_closed_analysis_without_summary_is_not_a_summary(self):
        assert _extract_summary("<analysis>reasoning</analysis>") == ""

    def test_plain_untagged_text_still_falls_back(self):
        # Intentional existing behavior — only the <analysis> shape is a failure.
        assert _extract_summary("A plain summary with no tags") == (
            "A plain summary with no tags"
        )

    @patch("agentic.agent.compaction.litellm")
    def test_analysis_leak_returns_original_list_object(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "<analysis>1. Chronologically analyzing the conversation..."
        )
        messages = list(_BASE)
        assert compact_messages(messages, model="gpt-5.4") is messages

    @patch("agentic.agent.compaction.litellm")
    def test_length_finish_reason_without_summary_tag_is_failure(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(
            "Some partial prose that got cut off mid-", finish_reason="length"
        )
        messages = list(_BASE)
        assert compact_messages(messages, model="gpt-5.4") is messages

    @patch("agentic.agent.compaction.litellm")
    def test_max_tokens_finish_reason_without_summary_tag_is_failure(
        self, mock_litellm
    ):
        mock_litellm.completion.return_value = _mock_response(
            "partial prose", finish_reason="max_tokens"
        )
        messages = list(_BASE)
        assert compact_messages(messages, model="gpt-5.4") is messages

    @patch("agentic.agent.compaction.litellm")
    def test_length_finish_reason_with_open_summary_tag_still_used(self, mock_litellm):
        # Truncated, but the summary DID open — keep the partial summary.
        mock_litellm.completion.return_value = _mock_response(
            "<analysis>r</analysis><summary>partial but real summary",
            finish_reason="length",
        )
        result = compact_messages(_BASE, model="gpt-5.4")
        assert result is not _BASE
        assert result[1]["content"] == "partial but real summary"


class TestCompactMessagesTools:
    """Tool schemas are part of the cached prefix — they must be sent."""

    _TOOLS = [
        {
            "type": "function",
            "function": {"name": "search", "description": "s", "parameters": {}},
        }
    ]

    @patch("agentic.agent.compaction.litellm")
    def test_tools_forwarded_to_completion(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(_BASE, model="gpt-5.4", tools=self._TOOLS)
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["tools"] == self._TOOLS

    @patch("agentic.agent.compaction.litellm")
    def test_tool_choice_none_so_model_summarizes(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(_BASE, model="gpt-5.4", tools=self._TOOLS)
        assert mock_litellm.completion.call_args.kwargs["tool_choice"] == "none"

    @patch("agentic.agent.compaction.litellm")
    def test_no_tools_kwarg_when_absent(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(_BASE, model="gpt-5.4")
        kwargs = mock_litellm.completion.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


class TestCompactionInstructionContent:
    """Pin the load-bearing requirements — deleting them passes everything else."""

    def test_demands_verbatim_quotes(self):
        assert "verbatim quotes" in compaction.COMPACTION_INSTRUCTION

    def test_has_materials_sources_and_findings_section(self):
        assert "Materials, Sources, and Findings" in compaction.COMPACTION_INSTRUCTION

    def test_requires_identifiers_and_citations(self):
        text = compaction.COMPACTION_INSTRUCTION
        assert "citations" in text
        assert "identifiers" in text

    def test_asks_for_detail_not_brevity(self):
        text = compaction.COMPACTION_INSTRUCTION
        assert "detailed summary" in text
        # Regressions historically reintroduced brevity instructions, which is
        # what made compaction lossy in the first place.
        for banned in ("be brief", "be concise", "keep it short", "briefly"):
            assert banned not in text.lower()

    def test_requires_analysis_then_summary_tags(self):
        text = compaction.COMPACTION_INSTRUCTION
        assert "<analysis>" in text
        assert "<summary>" in text


class TestTruncateMessagesPostconditions:
    def test_never_collapses_to_system_only(self):
        # The exact production shape: state.messages ends with a huge tool
        # result whose parent assistant tool_calls message the budget drops.
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 400_000},
        ]
        result = truncate_messages(messages, 1000)
        assert [m["role"] for m in result] != ["system"]
        assert len(result) > 1
        # The assistant tool_calls parent came along, so the tool result is
        # not an orphan and survives normalization.
        assert any(m["role"] == "tool" for m in result)
        assert any(m.get("tool_calls") for m in result)

    def test_honors_target_for_single_oversized_message(self):
        messages = [{"role": "user", "content": "x" * 40_000}]
        result = truncate_messages(messages, 100)
        assert estimate_token_count(result) <= 100
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_honors_target_for_oversized_tool_result(self):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 400_000},
        ]
        result = truncate_messages(messages, 1000)
        assert estimate_token_count(result) <= 1000

    def test_retains_some_original_content_after_truncation(self):
        messages = [{"role": "user", "content": "IMPORTANT" + "x" * 40_000}]
        result = truncate_messages(messages, 100)
        assert result[0]["content"].startswith("IMPORTANT")

    def test_multimodal_content_truncated_under_target(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "y" * 40_000},
                    {"type": "image_url", "image_url": {"url": "http://x/i.png"}},
                ],
            }
        ]
        result = truncate_messages(messages, 100)
        assert estimate_token_count(result) <= 100


class TestTruncateMessagesNeverRaises:
    """The reactive path persists this result and cannot retry — it must not
    raise, and its postconditions must be enforced without ``assert`` (which
    ``python -O`` strips)."""

    @pytest.mark.parametrize(
        "messages",
        [
            [],
            [{"role": "user", "content": "x" * 4_000_000}],
            [{"role": "system", "content": "SYS"}],
            [{"role": "tool", "tool_call_id": "orphan", "content": "r" * 40_000}],
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "q" * 40_000},
                {"role": "tool", "tool_call_id": "orphan", "content": "r" * 40_000},
            ],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": []}],
        ],
        ids=[
            "empty",
            "single-enormous",
            "system-only",
            "orphan-tool-only",
            "orphan-tool-tail",
            "none-content",
            "empty-list-content",
        ],
    )
    @pytest.mark.parametrize("target", [0, 1, 100])
    def test_no_exception_escapes(self, messages, target):
        truncate_messages(list(messages), target)

    def test_returns_input_object_when_postcondition_violated(self):
        """A guard violation must fail safe: log, and hand back the input list
        object so callers detect no-progress with ``is`` (same contract as
        ``compact_messages``)."""
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "x" * 4000},
        ]
        # Force the "overshot target" branch: _fit_to_budget hands back
        # something still over budget.
        with patch.object(
            compaction, "_fit_to_budget", side_effect=lambda msgs, _t: msgs
        ):
            result = truncate_messages(messages, 10)
        assert result is messages

    def test_returns_input_object_when_result_is_system_only(self):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "x" * 4000},
        ]
        with patch.object(
            compaction,
            "_fit_to_budget",
            side_effect=lambda msgs, _t: [m for m in msgs if m["role"] == "system"],
        ):
            result = truncate_messages(messages, 10_000)
        assert result is messages

    def test_logs_a_warning_on_violation(self, caplog):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "x" * 4000},
        ]
        with (
            patch.object(
                compaction, "_fit_to_budget", side_effect=lambda msgs, _t: msgs
            ),
            caplog.at_level("WARNING"),
        ):
            truncate_messages(messages, 10)
        assert any("postcondition violated" in r.getMessage() for r in caplog.records)


class TestGetContextThresholdOutputReservation:
    def test_reserves_configured_max_tokens_when_larger(self):
        with patch.object(compaction, "resolve_context_window", return_value=200_000):
            # buffer is 8% of 200_000 = 16_000; reserve max(32_000, 8_000)
            assert compaction.get_context_threshold("m", 32_000) == 152_000

    def test_threshold_stays_under_real_input_ceiling(self):
        # The incident shape: with a hardcoded 8k reservation the threshold
        # (176_000) sat ABOVE the true input ceiling (200_000 - 32_000).
        with patch.object(compaction, "resolve_context_window", return_value=200_000):
            threshold = compaction.get_context_threshold("m", 32_000)
        assert threshold < 200_000 - 32_000

    def test_floor_is_max_output_tokens_when_max_tokens_smaller(self):
        with patch.object(compaction, "resolve_context_window", return_value=128_000):
            assert compaction.get_context_threshold("m", 1_000) == 107_000

    def test_none_max_tokens_matches_legacy_behavior(self):
        with patch.object(compaction, "resolve_context_window", return_value=128_000):
            assert compaction.get_context_threshold("m", None) == 107_000
            assert compaction.get_context_threshold("m") == 107_000


class TestGetContextThresholdSmallWindows:
    """A window small enough that reserve + buffer exceeds it must still yield
    a positive, usable threshold. A non-positive threshold makes
    `token_estimate > threshold` unconditionally true, so compaction fires
    before the first request and replaces the user's question with a summary
    of nothing — and neither the circuit breaker nor truncation can recover."""

    @pytest.mark.parametrize(
        "model,window",
        [("gpt-4", 8192), ("gpt-3.5-turbo", 16385)],
    )
    def test_threshold_is_positive(self, model, window):
        with patch.object(compaction, "resolve_context_window", return_value=window):
            threshold = compaction.get_context_threshold(model)
        assert threshold > 0

    @pytest.mark.parametrize(
        "model,window",
        [("gpt-4", 8192), ("gpt-3.5-turbo", 16385)],
    )
    def test_threshold_is_sensible_fraction_of_window(self, model, window):
        # Requirement changed in review round 3: the old `>= 0.4 * window`
        # floor was the very over-commitment being fixed — on a small window
        # it can only be met by shipping a threshold whose own output
        # reservation no longer fits. The real invariant is that the threshold
        # leaves room for the output the caller will actually be served.
        with patch.object(compaction, "resolve_context_window", return_value=window):
            threshold = compaction.get_context_threshold(model)
            output = compaction.effective_max_output_tokens(model)
        assert threshold >= window * compaction._MIN_INPUT_FRACTION
        assert threshold + output + compaction._compact_buffer(window) <= window

    def test_buffer_never_exceeds_a_quarter_of_a_small_window(self):
        assert compaction._compact_buffer(8192) <= 8192 // 4
        assert compaction._compact_buffer(16385) <= 16385 // 4

    def test_warns_when_reservation_exceeds_window(self, caplog):
        # Requirement changed in review round 3: the warning now fires exactly
        # when the output request is capped — i.e. when the caller will not get
        # the output length it asked for. The old predicate was silent on every
        # case where the threshold floor caused harm.
        with (
            patch.object(compaction, "resolve_context_window", return_value=8192),
            caplog.at_level("WARNING"),
        ):
            compaction.get_context_threshold("gpt-4")
        assert "context_output_request_capped" in caplog.text
        assert "gpt-4" in caplog.text
        assert "window=8192" in caplog.text

    def test_no_warning_for_a_healthy_window(self, caplog):
        with (
            patch.object(compaction, "resolve_context_window", return_value=128_000),
            caplog.at_level("WARNING"),
        ):
            compaction.get_context_threshold("big")
        assert "context_output_request_capped" not in caplog.text

    def test_no_warning_when_a_large_request_still_fits(self, caplog):
        # 200k window / 128k request: served in full, so no warning.
        with (
            patch.object(compaction, "resolve_context_window", return_value=200_000),
            caplog.at_level("WARNING"),
        ):
            compaction.get_context_threshold("big", 128_000)
        assert "context_output_request_capped" not in caplog.text

    def test_large_window_behavior_unchanged(self):
        """The bounds must not disturb the pinned large-window thresholds."""
        with patch.object(compaction, "resolve_context_window", return_value=1_048_576):
            assert compaction.get_context_threshold("deepseek") == 956_690
        with patch.object(compaction, "resolve_context_window", return_value=128_000):
            assert compaction.get_context_threshold("unknown") == 107_000


class TestExtractSummaryAnalysisGuard:
    """The <analysis> guard must not be defeated by a preamble. Models
    routinely emit a conversational lead-in before the tag; an anchored match
    let that shape install the model's scratchpad as the whole conversation."""

    def test_bare_analysis_block_rejected(self):
        assert _extract_summary("<analysis>scratch") == ""

    def test_analysis_block_after_preamble_rejected(self):
        assert _extract_summary("Okay, let me analyze.\n<analysis>scratch") == ""

    def test_analysis_block_after_whitespace_rejected(self):
        assert _extract_summary("\n\n   <analysis>scratch") == ""

    def test_genuine_plaintext_fallback_still_works(self):
        assert _extract_summary("just a plain summary") == "just a plain summary"

    def test_summary_tag_wins_over_analysis_anywhere(self):
        text = "preamble <analysis>x</analysis> then <summary>real</summary>"
        assert _extract_summary(text) == "real"

    def test_open_summary_after_analysis_still_extracted(self):
        text = "preamble <analysis>x</analysis><summary>cut off here"
        assert _extract_summary(text) == "cut off here"

    def test_attributed_analysis_without_summary_does_not_leak(self):
        # An <analysis> tag with any attribute defeats a bare-tag regex, so the
        # scratchpad falls through to the plain-text fallback and installs the
        # model's private reasoning as the whole conversation. Models emit
        # attributes routinely; the summary instruction does not forbid them.
        leak = (
            '<analysis confidence="low">PRIVATE: client likely lying about '
            "the dates; my confidence in OR 60 is very low.</analysis>"
        )
        assert _extract_summary(leak) == ""

    def test_attributed_analysis_truncated_does_not_leak(self):
        # Unclosed attributed tag (finish_reason=length) — the whole scratchpad
        # is what follows and must be dropped, not returned.
        assert _extract_summary('<analysis reason="x">secret reasoning') == ""

    def test_attributed_analysis_before_real_summary_is_stripped(self):
        text = (
            '<analysis lvl="2">PRIVATE hedge <summary> mentioned mid-analysis'
            "</analysis>\n<summary>the clean summary</summary>"
        )
        assert _extract_summary(text) == "the clean summary"

    def test_summary_guard_agrees_on_attributed_analysis(self):
        # The finish_reason guard and the extraction path must not disagree:
        # neither should see a summary in an attributed analysis-only output.
        leak = '<analysis confidence="low">no summary here</analysis>'
        assert _summary_started(leak) is False


def _tool_call_msg(args, name="search", call_id="t1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


class TestEstimateTokenCountToolCalls:
    """An assistant message can carry its entire payload in
    tool_calls[].function.arguments with content=None. Measuring that at 0
    tokens made truncate_messages' 'did we get under target' postcondition
    unenforceable: it reported success having shrunk nothing, and the
    identical oversized request was re-sent."""

    def test_tool_call_arguments_are_counted(self):
        messages = [_tool_call_msg("x" * 400_000)]
        assert estimate_token_count(messages) > 99_000

    def test_tool_call_name_is_counted(self):
        assert estimate_token_count([_tool_call_msg("", name="a" * 400)]) == 100

    def test_content_and_tool_calls_are_summed(self):
        msg = _tool_call_msg("a" * 400)
        msg["content"] = "b" * 400
        # 400 args + 400 content + len("search")=6 -> 806 // 4
        assert estimate_token_count([msg]) == 201

    def test_malformed_tool_calls_do_not_raise(self):
        for calls in ("not-a-list", [None], [{"function": "nope"}], [{}], []):
            estimate_token_count([{"role": "assistant", "tool_calls": calls}])


class TestTruncateMessagesShedsToolCallPayloads:
    """Correct measurement is what makes the drop-oldest budget walk able to
    shed these messages: it can only drop what it can see is expensive."""

    def test_oldest_oversized_tool_call_messages_are_dropped(self):
        """The tool_calls payload is the ONLY expensive message here, so
        shedding it depends entirely on the walk being able to see its cost.
        Measured at 0 tokens it was retained and the result falsely reported
        as 5 tokens; measured correctly the walk stops before it."""
        messages = [
            {"role": "user", "content": "q"},
            _tool_call_msg("x" * 400_000),
            {"role": "user", "content": "the actual question"},
        ]
        result = truncate_messages(messages, 1_000)
        assert estimate_token_count(result) <= 1_000
        assert not any(m.get("tool_calls") for m in result)
        assert result[-1]["content"] == "the actual question"

    def test_answered_oversized_tool_call_pair_is_shed(self):
        """An oversized tool_calls message answered by its tool result must be
        shed as a PAIR, not pinned by the orphan repair.

        `_fit_to_budget` cannot shrink `tool_calls` arguments, so walking back
        onto an unaffordable parent to save its tool result guaranteed an
        overshoot — the postcondition fired and truncation bailed entirely,
        leaving the reactive path to re-send the same oversized request with no
        recovery left. Dropping the orphaned result instead lets the walk land
        under target."""
        messages = [
            {"role": "user", "content": "q"},
            _tool_call_msg("x" * 400_000, call_id="t1"),
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
            {"role": "user", "content": "the actual question"},
        ]
        result = truncate_messages(messages, 1_000)

        assert result is not messages, "must not bail to the no-progress path"
        # Postcondition 2: really under target.
        assert estimate_token_count(result) <= 1_000
        # The pair is gone — neither the oversized parent nor its now-orphaned
        # result may survive.
        assert not any(m.get("tool_calls") for m in result)
        assert not any(m["role"] == "tool" for m in result)
        # Postcondition 1: real content survives.
        assert result[-1]["content"] == "the actual question"

    def test_affordable_tool_call_parent_is_still_pulled_back_in(self):
        """The affordability guard must not over-fire: a parent whose
        tool_calls payload fits is still worth walking back onto, even when the
        tool RESULT is huge — result content is truncatable, so keeping the
        pair costs nothing the fitter cannot recover."""
        messages = [
            {"role": "user", "content": "q"},
            _tool_call_msg("{}", call_id="t1"),  # tiny, affordable payload
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 400_000},
        ]
        result = truncate_messages(messages, 1_000)

        assert estimate_token_count(result) <= 1_000
        assert any(m.get("tool_calls") for m in result)
        assert any(m["role"] == "tool" for m in result)

    def test_result_measured_under_target_with_tool_calls_retained(self):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "q" * 40_000},
            _tool_call_msg("a" * 400, call_id="t1"),
            {"role": "tool", "tool_call_id": "t1", "content": "r" * 40_000},
        ]
        result = truncate_messages(messages, 1_000)
        assert estimate_token_count(result) <= 1_000

    def test_unshrinkable_tool_call_payload_is_reported_not_hidden(self, caplog):
        """When the only retainable message is an oversized tool_calls payload
        there is nothing to shed — but that must surface as the no-progress
        contract plus a warning, not as a false success."""
        messages = [
            {"role": "user", "content": "q"},
            _tool_call_msg("x" * 400_000),
        ]
        with caplog.at_level("WARNING"):
            result = truncate_messages(messages, 1_000)
        assert result is messages
        assert "postcondition violated" in caplog.text


class TestTruncateMessagesPreservesSystemMessage:
    """The system message carries the agent's operating and safety rules. The
    budget walk reserves it whole, so water-filling must not then cap it."""

    def test_system_message_is_never_truncated(self):
        system = "IMPORTANT SAFETY RULES. " * 2_000
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "x" * 40_000},
        ]
        result = truncate_messages(messages, 20_000)
        assert result[0]["content"] == system
        assert "[... truncated ...]" not in result[0]["content"]

    def test_system_only_input_goes_through_postconditions(self, caplog):
        """This used to return early, before _fit_to_budget and both
        postconditions — yielding 25_000 tokens against a target of 50."""
        messages = [{"role": "system", "content": "s" * 100_000}]
        with caplog.at_level("WARNING"):
            result = truncate_messages(messages, 50)
        assert result is messages
        assert "postcondition violated" in caplog.text

    def test_small_system_only_input_still_passes_through(self):
        messages = [{"role": "system", "content": "SYS"}]
        result = truncate_messages(messages, 1_000)
        assert [m["role"] for m in result] == ["system"]
        assert result[0]["content"] == "SYS"


class TestTruncateMessagesNeverRaisesOnMalformedInput:
    def test_message_without_role_returns_input_unchanged(self, caplog):
        """normalize_messages indexes msg["role"] unguarded, so a role-less
        message raises KeyError. truncate_messages promises it cannot raise."""
        messages = [
            {"role": "user", "content": "q" * 40_000},
            {"content": "no role here"},
        ]
        with caplog.at_level("WARNING"):
            result = truncate_messages(messages, 1_000)
        assert result is messages
        assert "failed unexpectedly" in caplog.text
        assert "KeyError" in caplog.text


class TestCompactMessagesLogsAllNoProgressPaths:
    def test_too_few_messages_is_logged(self, caplog):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        with caplog.at_level("WARNING"):
            result = compact_messages(messages, "gpt-4")
        assert result is messages
        assert "too few messages" in caplog.text


@pytest.fixture(autouse=True)
def _reset_compaction_warn_dedupe():
    """`effective_max_output_tokens` dedupes its cap warning by key, so a test
    asserting the warning must not be silenced by an earlier test that already
    tripped the same key."""
    compaction._warned.clear()
    yield
    compaction._warned.clear()


class TestEffectiveMaxOutputTokens:
    """The output request the window can actually serve.

    The reserve must be capped so that `threshold + reserve + buffer <= window`
    holds unconditionally. The previous shape floored the *threshold* instead,
    which over-committed the window exactly when `max_tokens` was large.
    """

    @pytest.mark.parametrize(
        "window,max_tok,expected",
        [
            (200_000, 128_000, 128_000),  # fits: served in full
            (1_048_576, 600_000, 600_000),  # fits: served in full
            (131_072, 65_536, 65_536),  # fits exactly
            (8_192, None, 4_096),  # capped: 8192 - 2048 buffer - 2048 input
        ],
    )
    def test_capped_to_what_the_window_can_serve(self, window, max_tok, expected):
        with patch.object(compaction, "resolve_context_window", return_value=window):
            assert compaction.effective_max_output_tokens("m", max_tok) == expected

    def test_default_arg_is_optional(self):
        with patch.object(compaction, "resolve_context_window", return_value=200_000):
            assert compaction.effective_max_output_tokens("m") == 8_000

    def test_never_below_the_hardcoded_minimum_when_window_allows(self):
        with patch.object(compaction, "resolve_context_window", return_value=200_000):
            assert compaction.effective_max_output_tokens("m", 100) == 8_000

    def test_warns_when_the_request_is_capped(self, caplog):
        with (
            patch.object(compaction, "resolve_context_window", return_value=8_192),
            caplog.at_level("WARNING"),
        ):
            compaction.effective_max_output_tokens("gpt-4", None)
        assert "context_output_request_capped" in caplog.text
        assert "gpt-4" in caplog.text
        assert "window=8192" in caplog.text

    def test_does_not_warn_when_the_request_is_served_in_full(self, caplog):
        with (
            patch.object(compaction, "resolve_context_window", return_value=200_000),
            caplog.at_level("WARNING"),
        ):
            compaction.effective_max_output_tokens("big", 128_000)
        assert "context_output_request_capped" not in caplog.text

    def test_warning_is_deduped(self, caplog):
        with (
            patch.object(compaction, "resolve_context_window", return_value=8_192),
            caplog.at_level("WARNING"),
        ):
            compaction.effective_max_output_tokens("gpt-4", None)
            compaction.effective_max_output_tokens("gpt-4", None)
        assert caplog.text.count("context_output_request_capped") == 1


class TestGetContextThresholdFitsTheWindow:
    """Regression: the unconditional threshold floor fired whenever the
    computed threshold was merely *small*, not only when it was non-positive.
    A large `max_tokens` makes it legitimately small, so correct thresholds got
    raised to unsafe ones and `threshold + max_tokens` overshot the window.
    Because `estimate_token_count` is a `chars // 4` undercount, a zero-or-
    negative margin means compaction declines to fire and the provider returns
    the `prompt_too_long` this function exists to prevent."""

    @pytest.mark.parametrize(
        "window,max_tok,expected",
        [
            (200_000, 128_000, 56_000),
            (1_048_576, 600_000, 364_690),
            (131_072, 65_536, 52_536),
        ],
    )
    def test_computed_threshold_is_shipped_unfloored(self, window, max_tok, expected):
        with patch.object(compaction, "resolve_context_window", return_value=window):
            assert compaction.get_context_threshold("m", max_tok) == expected

    @pytest.mark.parametrize(
        "window,max_tok",
        [
            (200_000, 128_000),
            (1_048_576, 600_000),
            (131_072, 65_536),
            (8_192, None),
            (8_192, 128_000),
            (16_385, None),
            (128_000, None),
            (32_000, 30_000),
        ],
    )
    def test_threshold_plus_output_plus_buffer_never_exceeds_window(
        self, window, max_tok
    ):
        with patch.object(compaction, "resolve_context_window", return_value=window):
            threshold = compaction.get_context_threshold("m", max_tok)
            output = compaction.effective_max_output_tokens("m", max_tok)
        assert threshold > 0
        assert threshold + output + compaction._compact_buffer(window) <= window

    def test_small_window_threshold_is_positive_and_safe(self):
        with patch.object(compaction, "resolve_context_window", return_value=8_192):
            assert compaction.get_context_threshold("gpt-4") == 2_048


class TestSummaryMentionInsideAnalysisDoesNotLeak:
    """The COMPACTION_INSTRUCTION tells the model to wrap its output in
    <summary> tags, so a model restating that plan mid-<analysis> is ordinary.
    Treating any occurrence of the literal tag as "the summary started" drags
    the scratchpad — which carries confidence hedges the user must not see —
    into the conversation history."""

    _LEAKY = (
        "<analysis>\n1. PRIVATE: client may be lying about the dates; my "
        "confidence in OR 60 is low.\n2. I will now wrap the result in "
        "<summary> tags as instructed.\n</analysis>\n<summary>\n"
        "Primary Request and Intent: the clean summary.\n</summary>"
    )

    def test_only_the_real_summary_is_extracted(self):
        assert _extract_summary(self._LEAKY) == (
            "Primary Request and Intent: the clean summary."
        )

    def test_no_scratchpad_content_survives(self):
        out = _extract_summary(self._LEAKY)
        assert "PRIVATE" not in out
        assert "analysis>" not in out
        assert "wrap the result" not in out

    def test_truncated_analysis_mentioning_summary_is_rejected(self):
        # finish_reason="length" shape: the analysis never closed, so the
        # <summary> mention inside it is not a summary opening.
        text = (
            "<analysis>\n1. PRIVATE: my confidence in OR 60 is low.\n"
            "2. I will now wrap the result in <summary> tags as instructed."
        )
        assert _extract_summary(text) == ""

    @patch("agentic.agent.compaction.litellm")
    def test_finish_reason_guard_agrees_with_extraction(self, mock_litellm, caplog):
        # The guard must recognize a <summary> mention buried in an unclosed
        # <analysis> as "summary never opened" and bail on the finish_reason
        # path — same verdict the extraction path reaches. The guard's own
        # "truncated before <summary> opened" warning isolates it: the old
        # guard matched the literal mention, thought the summary had started,
        # and fell through to extraction instead of firing here.
        mock_litellm.completion.return_value = _mock_response(
            "<analysis>\n1. PRIVATE hedge.\n2. I will now wrap the result in "
            "<summary> tags as instructed.",
            finish_reason="length",
        )
        messages = list(_BASE)
        with caplog.at_level("WARNING"):
            assert compact_messages(messages, model="gpt-5.4") is messages
        assert "truncated" in caplog.text
        assert "before <summary> opened" in caplog.text

    @patch("agentic.agent.compaction.litellm")
    def test_leaky_output_compacts_to_the_clean_summary(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response(self._LEAKY)
        result = compact_messages(list(_BASE), model="gpt-5.4")
        assert result[1]["content"] == (
            "Primary Request and Intent: the clean summary."
        )


class TestTagWhitespaceTolerance:
    def test_summary_tag_with_trailing_space(self):
        assert _extract_summary("<summary >real summary</summary >") == ("real summary")

    def test_open_only_summary_tag_with_space(self):
        assert _extract_summary("<summary >cut off") == "cut off"

    def test_spaced_analysis_tag_is_still_a_scratchpad(self):
        assert _extract_summary("< analysis >scratch") == ""

    def test_spaced_closed_analysis_block_is_stripped(self):
        text = "< analysis >scratch< / analysis ><summary>clean</summary>"
        assert _extract_summary(text) == "clean"


class TestContentCharLenMeasuresRealPayload:
    """A `total += 1000` proxy under-measured a 6MB base64 data URL by ~2000x,
    so `truncate_messages`' postcondition passed on a megabyte history and the
    whole compaction subsystem was inoperative for image-mode knowledge bases.
    The platform's context handler resolves image storage refs to inline
    base64 data URLs, so this is production-reachable."""

    def test_base64_image_block_measured_by_its_payload(self):
        url = "data:image/png;base64," + "A" * 6_000_000
        content = [{"type": "image_url", "image_url": {"url": url}}]
        assert compaction._content_char_len(content) >= 6_000_000

    def test_image_estimate_is_not_the_old_proxy(self):
        url = "data:image/png;base64," + "A" * 6_000_000
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ]
        assert estimate_token_count(msgs) > 1_000_000

    def test_short_remote_image_url_keeps_a_nonzero_floor(self):
        content = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
        assert compaction._content_char_len(content) >= 1_000

    def test_unrecognized_block_type_is_not_zero(self):
        content = [{"type": "tool_result", "content": "z" * 4_000_000}]
        assert compaction._content_char_len(content) >= 4_000_000

    def test_truncate_gets_a_huge_image_history_under_target(self):
        url = "data:image/png;base64," + "A" * 6_000_000
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "look at this"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "and this"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ]
        result = truncate_messages(messages, 5_000)
        assert result is not messages
        assert estimate_token_count(result) <= 5_000

    def test_truncate_gets_a_huge_unknown_block_under_target(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "z" * 4_000_000}],
            },
        ]
        result = truncate_messages(messages, 5_000)
        assert result is not messages
        assert estimate_token_count(result) <= 5_000


class TestCompactionCallResilience:
    """The compaction call is a normal provider call on the same model as the
    real one: a transient 529 must be retried rather than being a hard
    compaction failure, and it must not inherit litellm's 6000s default
    request timeout (a hung compaction blocking the agent for 100 minutes)."""

    @patch("agentic.agent.compaction.litellm")
    def test_num_retries_matches_the_real_agent_call(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(list(_BASE), model="gpt-5.4")
        assert mock_litellm.completion.call_args.kwargs["num_retries"] == 3

    @patch("agentic.agent.compaction.litellm")
    def test_timeout_is_explicit_and_bounded(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response("<summary>s</summary>")
        compact_messages(list(_BASE), model="gpt-5.4")
        timeout = mock_litellm.completion.call_args.kwargs["timeout"]
        assert 0 < timeout <= 900


class TestCompactMessagesDocumentsReasoningKwargs:
    def test_docstring_notes_reasoning_kwargs_are_not_forwarded(self):
        doc = compact_messages.__doc__ or ""
        assert "extra_body" in doc
        assert "reasoning" in doc
