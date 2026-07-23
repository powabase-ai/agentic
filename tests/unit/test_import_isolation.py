"""Verify that agentic/ source files cannot import billing-service modules."""

import subprocess
from pathlib import Path

AGENTIC_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_SRC = AGENTIC_ROOT / "src"


def test_no_billing_imports_in_agentic_src():
    """Run ruff/flake8-tidy-imports against agentic/src — expect zero violations."""
    result = subprocess.run(
        ["ruff", "check", "--select=TID", str(AGENTIC_SRC)],
        capture_output=True,
        text=True,
        cwd=AGENTIC_ROOT,
    )
    assert (
        result.returncode == 0
    ), f"ruff reported import violations:\n{result.stdout}\n{result.stderr}"


def test_lint_catches_simulated_violation(tmp_path):
    """Plant a file that imports a banned top-level module; expect ruff to flag it."""
    test_file = AGENTIC_SRC / "agentic" / "_v15_lint_check.py"
    test_file.write_text("from agentic_billing_service import models  # type: ignore\n")
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=TID", str(test_file)],
            capture_output=True,
            text=True,
            cwd=AGENTIC_ROOT,
        )
        assert result.returncode != 0, (
            "Expected ruff to flag the banned import; got returncode=0.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "agentic_billing_service" in result.stdout
    finally:
        test_file.unlink()


def test_lint_catches_submodule_import(tmp_path):
    """Plant a file that imports from a banned submodule path; expect ruff to flag it.
    The banned-api rule must match `from X.sub.path import Y`, not just `from X import Y`."""
    test_file = AGENTIC_SRC / "agentic" / "_v15_submodule_check.py"
    test_file.write_text(
        "from agentic_billing_service.models.ledger import CreditLedger  # type: ignore\n"
    )
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=TID", str(test_file)],
            capture_output=True,
            text=True,
            cwd=AGENTIC_ROOT,
        )
        assert result.returncode != 0, (
            "Expected ruff to flag the submodule import; got returncode=0.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "agentic_billing_service" in result.stdout
    finally:
        test_file.unlink()
