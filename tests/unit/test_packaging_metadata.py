"""Packaging metadata: Apache-2.0 license present + declared; no personal email."""

import tomllib
from pathlib import Path

AGENTIC_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((AGENTIC_ROOT / "pyproject.toml").read_text())


def test_license_file_is_apache_2():
    text = (AGENTIC_ROOT / "LICENSE").read_text()
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    # Pin the "ship the WHOLE standard document" instruction (guard vs a stub):
    assert "APPENDIX: How to apply the Apache License to your work" in text
    assert len(text.splitlines()) >= 190


def test_pyproject_declares_apache_license():
    # PEP 639 SPDX string (preferred); tolerate the file/text form if hatchling needs it
    lic = _pyproject()["project"]["license"]
    assert lic in ("Apache-2.0", {"file": "LICENSE"}, {"text": "Apache-2.0"}), lic


def test_author_is_powabase_and_no_personal_email():
    authors = _pyproject()["project"]["authors"]
    assert authors == [{"name": "Powabase AI"}], authors
    raw = (AGENTIC_ROOT / "pyproject.toml").read_text()
    assert "gmail.com" not in raw


def test_readme_license_matches():
    readme = (AGENTIC_ROOT / "README.md").read_text()
    assert "Apache-2.0" in readme
    assert "\nMIT\n" not in readme  # the stale `## License\n\nMIT` line is gone
