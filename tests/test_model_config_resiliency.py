"""Pins the indexing-reliability knobs in model_config.

These are the actual fixes for the Celery event-loop transport bug and the
slow-reasoning-model timeout, and they had no test. In particular, the aiohttp
flag is set by attribute assignment — if litellm ever renames
``disable_aiohttp_transport``, the assignment would silently create a dead
attribute and the event-loop bug would resurface with no signal. The hasattr
assertion here turns that rename into a loud CI failure.
"""

from __future__ import annotations

import importlib

import litellm

from agentic.knowledge import model_config
from agentic.knowledge.model_config import _int_env


def test_litellm_exposes_disable_aiohttp_transport():
    """If this fails, litellm renamed/removed the flag — the model_config
    assignment is now a no-op and the aiohttp loop-affinity bug is back. Update
    the flag name (and the model_config guard) to the new litellm API."""
    assert hasattr(litellm, "disable_aiohttp_transport")


def test_aiohttp_transport_disabled_after_import():
    # Importing model_config (done at top) must have flipped the flag.
    assert litellm.disable_aiohttp_transport is True


def test_pageindex_llm_resiliency_defaults():
    assert model_config.PAGEINDEX_LLM_TIMEOUT == 300
    assert model_config.PAGEINDEX_LLM_NUM_RETRIES == 1


def test_int_env_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("AGENTIC_TEST_INT", "42")
    assert _int_env("AGENTIC_TEST_INT", 7) == 42
    # Malformed value must NOT raise (it would take down the whole
    # agentic.knowledge module at import) — it falls back to the default.
    monkeypatch.setenv("AGENTIC_TEST_INT", "not-an-int")
    assert _int_env("AGENTIC_TEST_INT", 7) == 7
    monkeypatch.delenv("AGENTIC_TEST_INT", raising=False)
    assert _int_env("AGENTIC_TEST_INT", 7) == 7


def test_env_override_applies(monkeypatch):
    """A valid env override is honored when the module is (re)imported."""
    monkeypatch.setenv("PAGEINDEX_LLM_TIMEOUT", "120")
    monkeypatch.setenv("PAGEINDEX_LLM_NUM_RETRIES", "0")
    reloaded = importlib.reload(model_config)
    try:
        assert reloaded.PAGEINDEX_LLM_TIMEOUT == 120
        assert reloaded.PAGEINDEX_LLM_NUM_RETRIES == 0
    finally:
        # Restore module-level defaults for any later test importing it.
        monkeypatch.delenv("PAGEINDEX_LLM_TIMEOUT", raising=False)
        monkeypatch.delenv("PAGEINDEX_LLM_NUM_RETRIES", raising=False)
        importlib.reload(model_config)
