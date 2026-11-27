"""Packaging metadata: PolyForm Noncommercial 1.0.0 license present + declared; no personal email."""

import tomllib
from pathlib import Path

AGENTIC_ROOT = Path(__file__).resolve().parents[2]

LICENSE_EXPRESSION = "PolyForm-Noncommercial-1.0.0"


def _pyproject() -> dict:
    return tomllib.loads((AGENTIC_ROOT / "pyproject.toml").read_text())


def test_license_file_is_polyform_noncommercial():
    text = (AGENTIC_ROOT / "LICENSE").read_text()
    assert text.startswith("# PolyForm Noncommercial License 1.0.0")
    assert "https://polyformproject.org/licenses/noncommercial/1.0.0" in text
    # Pin the full standard text (guard vs a stub): first and last sections.
    assert "## Acceptance" in text
    assert "## Noncommercial Organizations" in text
    assert "**Use** means anything you do with the software" in text
    assert "Required Notice: Copyright 2026 Powabase AI" in text
    assert "Apache License" not in text


def test_pyproject_declares_polyform_license():
    project = _pyproject()["project"]
    # PEP 639 SPDX expression; a license classifier alongside it is forbidden.
    assert project["license"] == LICENSE_EXPRESSION, project["license"]
    assert not any(
        c.startswith("License ::") for c in project.get("classifiers", [])
    ), project.get("classifiers")


def test_distributions_ship_license_files():
    files = _pyproject()["project"]["license-files"]
    assert "LICENSE" in files and "LICENSING.md" in files, files
    assert (AGENTIC_ROOT / "LICENSING.md").is_file()


def test_author_is_powabase_and_no_personal_email():
    authors = _pyproject()["project"]["authors"]
    assert authors == [{"name": "Powabase AI"}], authors
    raw = (AGENTIC_ROOT / "pyproject.toml").read_text()
    assert "gmail.com" not in raw


def test_readme_license_matches():
    readme = (AGENTIC_ROOT / "README.md").read_text()
    assert "PolyForm Noncommercial 1.0.0" in readme
    assert "LICENSING.md" in readme
    assert "Apache-2.0 — see [LICENSE]" not in readme  # the old claim is gone
    assert "\nMIT\n" not in readme  # the stale `## License\n\nMIT` line is gone
