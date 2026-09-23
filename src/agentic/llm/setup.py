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

import litellm

litellm.modify_params = True

# claude-opus-5-5 is missing from the cost map bundled with litellm 1.90.1
# (and still missing from 1.102.1, the newest available at the time this was
# written). litellm only picks up a brand-new model when its own runtime
# download of the live cost map succeeds, so whether the entry exists is a
# function of network conditions at import time, not of the litellm version
# pinned. Without an entry for this model:
#   - litellm.supports_reasoning() is False, so agentic drops the requested
#     reasoning effort and the model silently runs at its provider default;
#   - max_tokens falls back to litellm's generic default of 4096, which on
#     this model includes the (always-on) thinking, so replies truncate;
#   - cost reporting returns 0.
# supports_adaptive_thinking has to be part of the registered entry, not just
# supports_reasoning: without it, litellm maps a requested reasoning effort to
# thinking={"type": "enabled", "budget_tokens": N} (the pre-adaptive shape),
# which this model rejects with HTTP 400 -- it only accepts
# thinking={"type": "adaptive"}, with no budget to set or disable.
# supports_xhigh_reasoning_effort / supports_max_reasoning_effort /
# supports_output_config cover the model's xhigh and max effort levels, which
# litellm validates against these flags independently of supports_reasoning.
# supports_sampling_params has the same kind of bite as supports_adaptive_
# thinking: Claude 4.7+ dropped temperature/top_p/top_k, and litellm's name-
# based fallback for that (Fable/opus-4-7/opus-4-8 only) doesn't recognize
# this model, so without the flag litellm forwards temperature straight
# through to a model that rejects it with HTTP 400.
#
# The values below track the capability flags on litellm's own same-shaped
# "claude-opus-4-8" entry (inspected via litellm.model_cost under 1.90.1),
# keeping 5.5's own prices/limits: mode, the *_reasoning_effort/
# supports_output_config trio above, supports_sampling_params,
# supports_response_schema, supports_tool_choice, supports_pdf_input, and
# supports_assistant_prefill. cache_creation_input_token_cost_above_1hr is
# derived (2x the base input-cache-write cost — Anthropic's standard 1-hour
# cache multiplier) rather than copied, since it's price data, not a
# capability flag, and 5.5's own price differs from 4-8's.
#
# Only the bare "claude-opus-5-5" key is registered -- mirroring how litellm
# keys its own same-shaped "claude-opus-4-8" entry (litellm_provider
# "anthropic", no separate "anthropic/"-prefixed duplicate) -- because
# litellm strips a matching provider prefix before the cost-map lookup, so
# "anthropic/claude-opus-5-5" resolves through the same bare entry. This
# covers only the first-party Anthropic API route: the Vertex
# (vertex_ai/claude-opus-5-5) and Bedrock (bedrock/anthropic.claude-opus-5-5)
# routes are not registered here and still depend entirely on litellm's own
# map catching up.
#
# Guarded so this never overrides an entry litellm already has, whether
# that's a future litellm release that ships the model itself or a
# successful runtime download of the live map in this process.
if "claude-opus-5-5" not in litellm.model_cost:
    litellm.register_model(
        {
            "claude-opus-5-5": {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "input_cost_per_token": 4e-6,
                "output_cost_per_token": 2e-5,
                "cache_read_input_token_cost": 2e-7,
                "cache_creation_input_token_cost": 5e-6,
                "cache_creation_input_token_cost_above_1hr": 8e-6,
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 128_000,
                "max_tokens": 128_000,
                "supports_reasoning": True,
                "supports_adaptive_thinking": True,
                "supports_function_calling": True,
                "supports_vision": True,
                "supports_prompt_caching": True,
                "supports_xhigh_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
                "supports_output_config": True,
                "supports_sampling_params": False,
                "supports_response_schema": True,
                "supports_tool_choice": True,
                "supports_pdf_input": True,
                "supports_assistant_prefill": False,
            }
        }
    )
