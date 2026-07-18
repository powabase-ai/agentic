"""README must not carry stale claims: orchestration/workflow ship today, and
they are not '(placeholder)'. Also pins the platform_api E-relocation note."""

from pathlib import Path

AGENTIC_ROOT = Path(__file__).resolve().parents[2]


def test_readme_has_no_stale_claims():
    readme = (AGENTIC_ROOT / "README.md").read_text()
    assert "Coming Soon" not in readme
    assert "placeholder" not in readme.lower()


def test_platform_api_carries_e_relocation_note():
    src = (AGENTIC_ROOT / "src/agentic/workflow/blocks/platform_api.py").read_text()
    assert "Plan E" in src
    assert "powabase-ai" in src
