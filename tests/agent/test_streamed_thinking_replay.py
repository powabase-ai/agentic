"""End to end: streamed Claude thinking blocks replay on the next step exactly
as they streamed.

A fake Anthropic SSE server sits at ``httpx.Client.send``. Step 1 streams two
thinking blocks in several fragments each, each signed, then a tool call;
step 2's request body must carry both blocks undoubled, with their
signatures. litellm 1.103 repeats a block's whole text on the delta carrying
its signature, so appending it doubled every block -- which the provider
rejects on replay.
"""

import json
from unittest.mock import patch

import httpx

from agentic import Agent, ExecutionStatus
from agentic.agent.tools import BuiltinTool

MODEL = "claude-sonnet-4-5"


def _sse(events):
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def _start(i, block):
    return {"type": "content_block_start", "index": i, "content_block": block}


def _delta(i, delta):
    return {"type": "content_block_delta", "index": i, "delta": delta}


def _stop(i):
    return {"type": "content_block_stop", "index": i}


def _message_start(message_id):
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": MODEL,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
    }


def _thinking_block(i, fragments, signature):
    return [
        _start(i, {"type": "thinking", "thinking": ""}),
        *(_delta(i, {"type": "thinking_delta", "thinking": f}) for f in fragments),
        _delta(i, {"type": "signature_delta", "signature": signature}),
        _stop(i),
    ]


def _step_one():
    return _sse(
        [
            _message_start("m1"),
            *_thinking_block(0, ["Let me ", "think about ", "this."], "SIG-A"),
            *_thinking_block(1, ["Second ", "block."], "SIG-B"),
            _start(
                2, {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {}}
            ),
            _delta(2, {"type": "input_json_delta", "partial_json": '{"x": "1"}'}),
            _stop(2),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]
    )


def _step_two():
    return _sse(
        [
            _message_start("m2"),
            _start(0, {"type": "text", "text": ""}),
            _delta(0, {"type": "text_delta", "text": "done"}),
            _stop(0),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
    )


def test_streamed_thinking_blocks_replay_undoubled_with_signatures(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_STREAMING_ENABLED", "true")
    bodies: list[dict] = []

    def send(self, request, *args, **kwargs):
        if request.method != "POST":
            raise httpx.ConnectError("no network in tests", request=request)
        bodies.append(json.loads(request.content))
        content = _step_one() if len(bodies) == 1 else _step_two()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content,
            request=request,
        )

    tool = BuiltinTool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=lambda args, ctx: "ok",
    )
    agent = Agent(
        model=MODEL, system_prompt="sys", reasoning_effort="high", api_key="sk-test"
    )
    with patch.object(httpx.Client, "send", send):
        output = agent.run("hi", tools={"echo": tool})

    assert output.status == ExecutionStatus.COMPLETED
    assert len(bodies) == 2
    replayed = next(m for m in bodies[1]["messages"] if m["role"] == "assistant")
    thinking = [b for b in replayed["content"] if b["type"] == "thinking"]
    assert thinking == [
        {
            "type": "thinking",
            "thinking": "Let me think about this.",
            "signature": "SIG-A",
        },
        {"type": "thinking", "thinking": "Second block.", "signature": "SIG-B"},
    ]
