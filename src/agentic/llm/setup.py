"""Global LiteLLM configuration. Imported once at package load.

modify_params=True makes LiteLLM auto-drop unsupported params on retry. The
specific case it covers: when an OpenAI-compatible client sends `thinking={...}`
to Anthropic on a tool-result turn but the prior assistant message is missing
`thinking_blocks`, LiteLLM drops the `thinking` param instead of returning 400.

Two flags from the v2.0 spec (route_all_chat_openai_to_responses,
reasoning_auto_summary) DO NOT EXIST as globals in litellm 1.80.0 (and
were not load-bearing in 1.83.14 either — the per-call responses/ model
prefix is the verified mechanism). Per-call routing via the responses/ model
prefix and extra_body is used instead — see agentic/llm/routing.py (Task 4).
"""

import json
from pathlib import Path

import litellm

litellm.modify_params = True

# Models released after the cost map bundled with the pinned litellm
# (1.103.2) was cut: claude-opus-5-5, claude-sonnet-5-5, gpt-6-luna,
# gpt-6-sol and gpt-6.1-sol. litellm only picks up a brand-new model when its own runtime
# download of the live cost map succeeds, so whether the entry exists is a
# function of network conditions at import time, not of the litellm version
# pinned. Without an entry:
#   - a bare OpenAI id (gpt-6-luna) resolves to no provider at all, so the
#     call fails before it is sent;
#   - litellm.supports_reasoning() is False, so agentic drops the requested
#     reasoning effort and the model silently runs at its provider default;
#   - max_tokens falls back to litellm's generic default of 4096, which on
#     always-on-thinking Claude includes the thinking, so replies truncate;
#   - Claude 5.5 is missing supports_sampling_params=False, so litellm
#     forwards temperature to a model that rejects it with HTTP 400 instead
#     of dropping it;
#   - cost reporting returns 0.
#
# model_cost_additions.json holds litellm's own entries for these models,
# copied verbatim from the cost map on litellm's main branch (commit
# aa601ce4, 2026-10-01) -- capability flags (supports_adaptive_thinking,
# thinking_always_on, supports_sampling_params, the *_reasoning_effort
# flags) as well as prices, so the request shapes and cost figures match
# what the next litellm release will produce.
#
# Each entry is guarded so it never overrides one litellm already has,
# whether that's a future litellm release that ships the model itself or a
# successful runtime download of the live map in this process. Once the
# pinned litellm's bundled map has a model, its entry here is dead weight
# and can be deleted.
_ADDITIONS_PATH = Path(__file__).with_name("model_cost_additions.json")
_ADDITIONS: dict[str, dict] = json.loads(_ADDITIONS_PATH.read_text())

REGISTERED_MODEL_IDS: tuple[str, ...] = tuple(sorted(_ADDITIONS))

_missing = {
    model: entry
    for model, entry in _ADDITIONS.items()
    if model not in litellm.model_cost
}
if _missing:
    # register_model resolves each entry's provider and prints litellm's
    # "Provider List" banner to stdout for the bare OpenAI ids it doesn't
    # know yet -- the very ids being registered.
    _previous_suppress = litellm.suppress_debug_info
    litellm.suppress_debug_info = True
    try:
        litellm.register_model(_missing)
    finally:
        litellm.suppress_debug_info = _previous_suppress
