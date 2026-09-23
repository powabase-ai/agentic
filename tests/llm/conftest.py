"""Shared fixtures for tests/llm/."""

import pytest


@pytest.fixture
def restore_opus_5_5_model_cost():
    """Snapshot litellm.model_cost["claude-opus-5-5"] (present or absent) and
    restore that snapshot after the test, clearing litellm's get_model_info
    cache so a stale cached lookup can't leak into a later test.

    litellm.model_cost is one process-global dict shared by the whole pytest
    run, not scoped to a test or even a module. A test that pops the entry,
    sets a sentinel, or reloads agentic.llm.setup to re-register it would
    otherwise leave that mutation live for every test that runs after it --
    in this file or any other -- making the suite's outcome depend on
    collection order. Any test that mutates litellm.model_cost["claude-opus-5-5"]
    (directly, or indirectly via importlib.reload(setup)) should use this
    fixture.
    """
    import litellm

    had_key = "claude-opus-5-5" in litellm.model_cost
    original = litellm.model_cost.get("claude-opus-5-5")
    yield
    if had_key:
        litellm.model_cost["claude-opus-5-5"] = original
    else:
        litellm.model_cost.pop("claude-opus-5-5", None)
    if hasattr(litellm.get_model_info, "cache_clear"):
        litellm.get_model_info.cache_clear()


@pytest.fixture
def force_registered_opus_5_5(restore_opus_5_5_model_cost):
    """Pop claude-opus-5-5 from litellm.model_cost (if present) and reload
    agentic.llm.setup, so a test exercises setup.py's own registered entry
    deterministically.

    Without this, a test that just reads the ambient entry would pass
    vacuously in an environment where this process's litellm successfully
    downloaded a live cost map that already had the model (the registration
    guard then skips, and the test ends up checking litellm's live map
    instead of this fix) -- while still passing for the right reason in a
    network-less CI run, where the guard always fires. Forcing "absent" makes
    the test exercise setup.py's entry either way.
    """
    import importlib

    import litellm

    from agentic.llm import setup

    litellm.model_cost.pop("claude-opus-5-5", None)
    importlib.reload(setup)
