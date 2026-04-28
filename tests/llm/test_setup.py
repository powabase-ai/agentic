"""Tests for the global LiteLLM configuration."""


def test_modify_params_set_after_setup_import():
    import litellm

    # Reset to default to make the test idempotent
    litellm.modify_params = False

    # Importing the setup module should set the flag
    from agentic.llm import setup  # noqa: F401

    assert litellm.modify_params is True


def test_setup_imported_via_agentic_package():
    """agentic/__init__.py imports llm/setup.py first thing."""
    import litellm

    litellm.modify_params = False

    # Force a reimport so we test the package init path
    import importlib

    import agentic

    importlib.reload(agentic)

    assert litellm.modify_params is True
