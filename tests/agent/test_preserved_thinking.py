"""A Claude thinking block's signature binds the top-level system prompt, the
tool set and every message before it. The loop must never replay a block
whose prefix it changed, and recovers once if the provider rejects one."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agentic.agent import agent as agent_module
from agentic.agent.agent import Agent
from agentic.agent.tools import BuiltinTool
from agentic.execution.context import ExecutionContext, TokenBudget
from agentic.execution.status import ExecutionStatus

_REJECTED = (
    "litellm.BadRequestError: AnthropicException - messages.3.content.0: "
    "Invalid 'signature' in 'thinking' block. The block is bound to a "
    "different conversation."
)
_PRUNED = "[Previous tool result removed to save context]"


def _block(sig: str) -> dict:
    return {"type": "thinking", "thinking": f"thought {sig}", "signature": sig}


def _probe_tool():
    return BuiltinTool(
        name="probe",
        description="probe",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: "ok",
    )


def _response(*, content, tool_call_id=None, sig=None, total_tokens=15):
    tool_calls = None
    if tool_call_id is not None:
        tool_calls = [
            SimpleNamespace(
                id=tool_call_id,
                type="function",
                function=SimpleNamespace(name="probe", arguments="{}"),
            )
        ]
    msg = SimpleNamespace(
        content=content,
        role="assistant",
        reasoning_content=f"thought {sig}" if sig else None,
        tool_calls=tool_calls,
        thinking_blocks=[_block(sig)] if sig else None,
        provider_specific_fields=None,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=msg, finish_reason="tool_calls" if tool_calls else "stop"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=total_tokens
        ),
        id="resp",
    )


def _run(responses, *, input="question", context=None, max_steps=10, extra=()):
    context = context or ExecutionContext()
    with (
        patch("litellm.supports_reasoning", return_value=True),
        patch(
            "agentic.agent.agent.litellm.completion", side_effect=responses
        ) as completion,
        patch.dict("os.environ", {"AGENT_LLM_STREAMING_ENABLED": "false"}),
    ):
        for p in extra:
            p.start()
        try:
            agent = Agent(
                model="anthropic/claude-opus-4-8",
                system_prompt="You are a bot.",
                reasoning_effort="high",
            )
            output = agent.run(
                input,
                context=context,
                tools={"probe": _probe_tool()},
                max_steps=max_steps,
            )
        finally:
            for p in extra:
                p.stop()
    return output, [c.kwargs for c in completion.call_args_list]


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


# ---------------------------------------------------------------- budget warning


def test_budget_warning_is_a_user_message_after_the_steps_tool_results():
    budget = TokenBudget(max_tokens=1000)
    _, calls = _run(
        [
            _response(content=None, tool_call_id="c1", sig="s1", total_tokens=900),
            _response(content="done", sig="s2"),
        ],
        context=ExecutionContext(budget=budget),
    )
    assert len(calls) == 2
    sent = calls[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]
    assert sent[2]["thinking_blocks"] == [_block("s1")]
    assert _text(sent[-1]).startswith("BUDGET WARNING")
    assert [m["role"] for m in sent[1:]].count("system") == 0


def test_budget_warning_is_not_added_when_the_step_ends_the_run():
    budget = TokenBudget(max_tokens=1000)
    output, calls = _run(
        [_response(content="done", sig="s1", total_tokens=900)],
        context=ExecutionContext(budget=budget),
    )
    assert len(calls) == 1
    assert output.status.is_success()
    assert not any("BUDGET WARNING" in _text(m) for m in output.messages)


# ------------------------------------------------- _without_blocks_after_edit


def _history():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {}}],
            "thinking_blocks": [_block("a")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "big result"},
        {
            "role": "assistant",
            "content": "one",
            "thinking_blocks": [_block("b")],
            "reasoning_items": [{"id": "rs_1"}],
            "provider_specific_fields": {"thought_signatures": ["g"]},
        },
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "two", "thinking_blocks": [_block("c")]},
    ]


def test_no_edit_leaves_the_history_unchanged():
    before = _history()
    assert agent_module._without_blocks_after_edit(before, _history()) == _history()


def test_blocks_before_the_edit_stay_and_blocks_from_it_on_go():
    before = _history()
    after = _history()
    after[3] = {"role": "tool", "tool_call_id": "c1", "content": _PRUNED}
    result = agent_module._without_blocks_after_edit(before, after)
    assert result[2]["thinking_blocks"] == [_block("a")]
    assert "thinking_blocks" not in result[4]
    assert "thinking_blocks" not in result[6]
    assert result[3]["content"] == _PRUNED


def test_blocks_on_the_edited_message_itself_go():
    before = _history()
    after = _history()
    after[4] = dict(after[4], content="rewritten")
    result = agent_module._without_blocks_after_edit(before, after)
    assert result[2]["thinking_blocks"] == [_block("a")]
    assert "thinking_blocks" not in result[4]
    assert "thinking_blocks" not in result[6]


def test_other_replay_keys_are_untouched():
    before = _history()
    after = _history()
    after[3] = {"role": "tool", "tool_call_id": "c1", "content": _PRUNED}
    result = agent_module._without_blocks_after_edit(before, after)
    assert result[4]["reasoning_items"] == [{"id": "rs_1"}]
    assert result[4]["provider_specific_fields"] == {"thought_signatures": ["g"]}


def test_a_removed_message_counts_as_an_edit():
    before = _history()
    after = _history()
    del after[1]
    result = agent_module._without_blocks_after_edit(before, after)
    assert all("thinking_blocks" not in m for m in result)


def test_bookkeeping_keys_do_not_count_as_an_edit():
    before = _history()
    after = [dict(m, _injected=False) for m in _history()]
    result = agent_module._without_blocks_after_edit(before, after)
    assert result[6]["thinking_blocks"] == [_block("c")]


def test_the_inputs_are_not_mutated():
    before = _history()
    after = _history()
    after[3] = {"role": "tool", "tool_call_id": "c1", "content": _PRUNED}
    agent_module._without_blocks_after_edit(before, after)
    assert after[4]["thinking_blocks"] == [_block("b")]


def test_proactive_prune_sends_no_block_after_the_first_pruned_tool_message():
    reasoning = {"provider": "anthropic"}
    history = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ],
            "thinking_blocks": [_block("a")],
            "reasoning": reasoning,
        },
        {"role": "tool", "tool_call_id": "c0", "content": "big result"},
        {
            "role": "assistant",
            "content": "one",
            "thinking_blocks": [_block("b")],
            "reasoning": reasoning,
        },
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "two",
            "thinking_blocks": [_block("c")],
            "reasoning": reasoning,
        },
        {"role": "user", "content": "q3"},
        {
            "role": "assistant",
            "content": "three",
            "thinking_blocks": [_block("d")],
            "reasoning": reasoning,
        },
        {"role": "user", "content": "q4"},
    ]
    output, calls = _run(
        [_response(content="done", sig="s1")],
        input=history,
        extra=(
            patch("agentic.agent.agent.estimate_token_count", return_value=10_000_000),
            patch("agentic.agent.agent.get_context_threshold", return_value=0),
            patch(
                "agentic.agent.agent.compact_messages",
                side_effect=lambda messages, **kwargs: messages,
            ),
        ),
    )
    assert output.status.is_success()
    sent = calls[0]["messages"]
    pruned_at = next(
        i for i, m in enumerate(sent) if m["role"] == "tool" and _text(m) == _PRUNED
    )
    assert sent[pruned_at - 1]["thinking_blocks"] == [_block("a")]
    for message in sent[pruned_at:]:
        assert "thinking_blocks" not in message


# ------------------------------------------------------------------ backstop


def test_rejected_signature_strips_blocks_and_retries_once():
    events: list[dict] = []
    output, calls = _run(
        [
            _response(content=None, tool_call_id="c1", sig="s1"),
            Exception(_REJECTED),
            _response(content="done", sig="s2"),
        ],
        context=ExecutionContext(on_event=events.append),
    )
    assert output.status.is_success()
    assert output.content == "done"
    assert len(calls) == 3
    assert any("thinking_blocks" in m for m in calls[1]["messages"])
    for message in calls[2]["messages"]:
        assert "thinking_blocks" not in message
    resets = [
        e
        for e in events
        if e.get("type") == "step_reset"
        and e.get("reason") == "thinking_signature_rejected"
    ]
    assert [(e["type"], e["step"], e["reason"]) for e in resets] == [
        ("step_reset", 2, "thinking_signature_rejected")
    ]


def test_a_second_rejection_in_the_same_run_fails_as_before():
    events: list[dict] = []
    output, calls = _run(
        [
            _response(content=None, tool_call_id="c1", sig="s1"),
            Exception(_REJECTED),
            Exception(_REJECTED),
        ],
        context=ExecutionContext(on_event=events.append),
    )
    assert output.status == ExecutionStatus.FAILED
    assert "bound to a different conversation" in (output.error or "")
    assert len(calls) == 3
    resets = [
        e
        for e in events
        if e.get("type") == "step_reset"
        and e.get("reason") == "thinking_signature_rejected"
    ]
    assert len(resets) == 1
