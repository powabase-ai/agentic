"""The workflow FunctionBlock / copilot code-execution sandbox is a RUNTIME
contract: user-written code may `import` any third-party lib in IMPORT_REGISTRY
(function.py), and copilot advertises the same set as "Pre-installed". This guard
proves every such lib stays a declared `agentic` runtime dependency, so
dependency hygiene can never silently drop one (which would ImportError user code).
"""

import tomllib
from pathlib import Path

from agentic.workflow.blocks.function import _IMPORT_TO_PIP, IMPORT_REGISTRY

# tests/unit/test_sandbox_contract.py -> parents[2] == agentic/
AGENTIC_ROOT = Path(__file__).resolve().parents[2]


def _declared_runtime_deps() -> set[str]:
    """pip package names declared in [project.dependencies], normalized lower-case."""
    data = tomllib.loads((AGENTIC_ROOT / "pyproject.toml").read_text())
    names: set[str] = set()
    for spec in data["project"]["dependencies"]:
        bare = spec.split("[")[
            0
        ]  # drop extras: markitdown[xlsx,xls]>=0.1 -> markitdown
        for sep in (">=", "<=", "==", "~=", "!=", ">", "<", ";", " "):
            bare = bare.split(sep)[0]
        names.add(bare.strip().lower())
    return names


def _sandbox_contract_pip_names() -> set[str]:
    """The third-party IMPORT_REGISTRY libs, as pip package names, lower-cased."""
    pip: set[str] = set()
    for _key, (module_name, _alias, is_stdlib) in IMPORT_REGISTRY.items():
        if is_stdlib:
            continue
        pip.add(_IMPORT_TO_PIP.get(module_name, module_name).lower())
    return pip


def test_sandbox_contract_libs_are_declared_runtime_deps():
    missing = sorted(_sandbox_contract_pip_names() - _declared_runtime_deps())
    assert not missing, (
        "sandbox-contract libs missing from agentic runtime deps — user "
        f"FunctionBlock/copilot code imports these at runtime: {missing}"
    )
