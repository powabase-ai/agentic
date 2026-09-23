"""Reasoning items survive LiteLLM's real Responses-bridge streaming path.

The bridge turns raw Responses API stream events into chat-completion chunks,
putting the response's reasoning items on the ``response.completed`` chunk.
``accumulate_stream`` must collect them from there."""

from __future__ import annotations

import json

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)

from agentic.llm.streaming import accumulate_stream

_REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "encrypted_content": "ENCRYPTED",
    "summary": [],
}
_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "probe",
    "arguments": '{"q": "x"}',
    "status": "completed",
}


def _events() -> list[dict]:
    return [
        {"type": "response.created", "response": {"id": "resp_1", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": _REASONING},
        {"type": "response.output_item.done", "output_index": 0, "item": _REASONING},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": dict(_CALL, arguments="", status="in_progress"),
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "fc_1",
            "delta": '{"q": ',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "fc_1",
            "delta": '"x"}',
        },
        {"type": "response.output_item.done", "output_index": 1, "item": _CALL},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [_REASONING, _CALL],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 12},
                },
            },
        },
    ]


def test_reasoning_items_are_collected_from_the_bridge_stream():
    raw = [f"data: {json.dumps(event)}" for event in _events()]
    chunks = OpenAiResponsesToChatCompletionStreamIterator(iter(raw), sync_stream=True)
    msg, finish_reason, usage = accumulate_stream(iter(chunks))

    assert finish_reason == "tool_calls"
    assert [(tc.function.name, tc.function.arguments) for tc in msg.tool_calls] == [
        ("probe", '{"q": "x"}')
    ]
    items = [
        i if isinstance(i, dict) else i.model_dump(exclude_none=True)
        for i in msg.reasoning_items
    ]
    assert [(i["id"], i["encrypted_content"]) for i in items] == [("rs_1", "ENCRYPTED")]
    assert usage["reasoning_tokens"] == 12
