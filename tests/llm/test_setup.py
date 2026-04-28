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
