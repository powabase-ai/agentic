"""Tests for the global LiteLLM configuration."""

import importlib
import subprocess
import sys


def test_modify_params_set_when_setup_runs():
    """setup.py's module body sets litellm.modify_params=True. Verify by
    forcing a reload after manually clearing the flag."""
    import litellm

    from agentic.llm import setup

    litellm.modify_params = False
    importlib.reload(setup)

    assert litellm.modify_params is True


def test_setup_runs_via_agentic_package_import():
    """Verifies agentic.llm.setup is reachable via the agentic.llm namespace
    and reloading it re-fires the side effect. Note: this does NOT verify
    the `from . import llm` line in agentic/__init__.py — that wiring is
    covered by test_setup_runs_via_fresh_agentic_import below (subprocess)."""
    import litellm

    import agentic.llm

    litellm.modify_params = False
    importlib.reload(agentic.llm.setup)

    assert litellm.modify_params is True


def test_setup_runs_via_fresh_agentic_import():
    """Subprocess test: a fresh `import agentic` in a clean interpreter must
    set litellm.modify_params=True. This catches removal of the
    `from . import llm` line in agentic/__init__.py — the in-process tests
    above can't catch that because agentic.llm gets loaded directly via
    test 1's `from agentic.llm import setup`."""
    code = (
        "import litellm; "
        "assert litellm.modify_params is False, "
        "'precondition: modify_params is False before agentic import'; "
        "import agentic; "
        "import litellm; "
        "assert litellm.modify_params is True, "
        "'agentic import should have set modify_params via llm/setup.py'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"


def test_registers_claude_opus_5_5_when_absent_from_model_cost(
    restore_opus_5_5_model_cost,
):
    """setup.py registers claude-opus-5-5 with litellm when the installed
    litellm's cost map doesn't already know the model (e.g. the bundled map
    hasn't caught up yet, or the runtime download of the live map didn't
    happen/succeed). Simulate "missing" by removing any pre-existing entry
    (added either by a prior run of this same setup code, or by litellm's own
    successful live-map download) and reloading."""
    import litellm

    from agentic.llm import setup

    litellm.model_cost.pop("claude-opus-5-5", None)
    importlib.reload(setup)

    info = litellm.get_model_info("claude-opus-5-5")
    assert info["supports_reasoning"] is True
    assert info["max_output_tokens"] == 128000


def test_registers_claude_opus_5_5_against_a_genuinely_bundled_map():
    """End-to-end subprocess variant of the test above: force litellm to load
    its bundled cost map (no live-map download) in a fresh interpreter, then
    import agentic, then check the model is usable. This is the scenario the
    fix targets — litellm 1.90.1's bundled map lacks claude-opus-5-5."""
    code = (
        "import os; os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'; "
        "import litellm; "
        "import agentic; "
        "info = litellm.get_model_info('claude-opus-5-5'); "
        "assert info['supports_reasoning'] is True, info; "
        "assert info['max_output_tokens'] == 128000, info; "
        "assert litellm.supports_reasoning(model='claude-opus-5-5') is True"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"


def test_does_not_overwrite_an_existing_model_cost_entry(
    restore_opus_5_5_model_cost,
):
    """Never override an entry litellm already has — including a future
    litellm release that ships its own claude-opus-5-5 entry, or one filled in
    by a successful runtime download of the live cost map."""
    import litellm

    from agentic.llm import setup

    sentinel = {"litellm_provider": "anthropic", "mode": "chat", "sentinel": True}
    litellm.model_cost["claude-opus-5-5"] = sentinel
    importlib.reload(setup)

    assert litellm.model_cost["claude-opus-5-5"] == sentinel


def test_bare_registration_also_resolves_the_anthropic_prefixed_form(
    restore_opus_5_5_model_cost,
):
    """litellm strips the `anthropic/` prefix when looking up a model whose
    provider is anthropic, so registering only the bare `claude-opus-5-5` key
    (matching how litellm keys its own `claude-opus-4-8` entry) is enough to
    make `anthropic/claude-opus-5-5` resolve too."""
    import litellm

    from agentic.llm import setup

    litellm.model_cost.pop("claude-opus-5-5", None)
    importlib.reload(setup)

    info = litellm.get_model_info("anthropic/claude-opus-5-5")
    assert info["supports_reasoning"] is True
    assert info["max_output_tokens"] == 128000


def test_registered_entry_supports_response_schema(force_registered_opus_5_5):
    """claude-opus-5-5 supports strict structured output, same as
    claude-opus-4-8. Without `supports_response_schema: True` in the
    registered entry, `doc2json.py` falls back from strict `json_schema` to
    the looser `json_object` for this model.

    force_registered_opus_5_5 forces setup.py's own entry into place, so this
    doesn't pass vacuously against a live-downloaded map in a
    network-connected environment (see conftest.py)."""
    import litellm

    assert litellm.supports_response_schema("claude-opus-5-5") is True
