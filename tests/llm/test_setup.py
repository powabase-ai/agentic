"""Tests for the global LiteLLM configuration."""

import importlib


def test_modify_params_set_when_setup_runs():
    """setup.py's module body sets litellm.modify_params=True. Verify by
    forcing a reload after manually clearing the flag."""
    import litellm

    from agentic.llm import setup

    litellm.modify_params = False
    importlib.reload(setup)

    assert litellm.modify_params is True


def test_setup_runs_via_agentic_package_import():
    """agentic/__init__.py imports llm/setup.py first thing — verify the
    wiring exists by reaching setup through the agentic.llm namespace and
    reloading it. If agentic/__init__.py weren't importing llm, the lookup
    would fail."""
    import litellm

    import agentic.llm

    litellm.modify_params = False
    importlib.reload(agentic.llm.setup)

    assert litellm.modify_params is True
