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
