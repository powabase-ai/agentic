"""Per-call routing helpers for the OpenAI Responses API bridge.

The litellm 1.83.14 mechanism for routing through the Responses API (which is
the only OpenAI endpoint that returns reasoning content) is the responses/
model prefix — verified at litellm/main.py (responses_api_bridge_check).
Models matching `(openai|azure)/responses/<model>` are stripped of the prefix
and routed via the Responses bridge.

We transform the model at Agent call time. Other providers (Anthropic, Gemini)
return reasoning_content natively on Chat Completions when reasoning_effort
is passed; no transformation needed.

A0 verification (Task 1) revealed a LiteLLM 1.83.14 quirk: when going through
the Responses bridge, passing `reasoning_effort` at the top level is silently
dropped from the outgoing request. The verified working pattern is to pack
both `effort` and `summary` into `extra_body['reasoning']` and to NOT pass
top-level reasoning_effort on the Responses path. `reasoning_call_kwargs`
returns the right shape per route.
"""

from __future__ import annotations

import litellm


def maybe_route_through_responses(model: str, reasoning_effort: str | None) -> str:
    """For OpenAI/Azure reasoning models with reasoning_effort set, route via
    the Responses bridge by inserting `responses/` after the provider prefix.

    Provider is resolved via ``litellm.get_llm_provider`` so bare model names
    (e.g. ``gpt-5.4``, what the agent UI stores) are handled the same as
    prefixed forms (``openai/gpt-5.4``).

    Examples:
        gpt-5.4                  + medium → openai/responses/gpt-5.4   (bare → resolved to openai)
        openai/gpt-5.4           + medium → openai/responses/gpt-5.4
        azure/my-deploy          + medium → azure/responses/my-deploy
        claude-opus-4-7          + medium → unchanged (Anthropic returns reasoning natively)
        anthropic/...            + medium → unchanged
        gpt-4o                   + medium → unchanged (no reasoning support)
        openai/responses/gpt-5.4 + medium → unchanged (already routed)
    """
    if reasoning_effort is None:
        return model
    if "/responses/" in model:
        return model

    # Resolve provider for bare or prefixed model names.
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return model

    if provider not in ("openai", "azure"):
        return model

    try:
        if not litellm.supports_reasoning(model=model):
            return model
    except Exception:
        return model

    # Insert /responses/ after the provider prefix, or synthesize one for
    # bare model names.
    if "/" in model:
        prefix, rest = model.split("/", 1)
        return f"{prefix}/responses/{rest}"
    return f"{provider}/responses/{model}"


def reasoning_call_kwargs(reasoning_effort: str | None, model: str) -> dict:
    """Return the kwargs dict to merge into litellm.completion(...) for
    requesting reasoning from this model.

    For non-Responses paths: {"reasoning_effort": effort}
    For Responses paths:     {"extra_body": {"reasoning": {"effort": effort, "summary": "detailed"}}}

    The Responses-path packing is required because litellm 1.83.14 silently
    drops top-level `reasoning_effort` when the call routes through the
    Responses bridge. Verified empirically in A0.1.
    """
    if reasoning_effort is None:
        return {}
    if "/responses/" in model:
        return {
            "extra_body": {
                "reasoning": {
                    "effort": reasoning_effort,
                    "summary": "detailed",
                }
            }
        }
    return {"reasoning_effort": reasoning_effort}
