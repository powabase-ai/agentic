#!/usr/bin/env bash
# Separability gate for the `agentic` package (OSS Plan D1).
#
# Proves `agentic` installs + runs standalone from a built wheel, and that the
# workflow/copilot code-execution sandbox contract (function.py IMPORT_REGISTRY)
# is satisfied by the RUNTIME dependency closure ALONE (stage 2 uses no dev extras,
# so a sandbox lib demoted to dev-deps fails here). Reused by Plan E as anti-rot CI.
set -euo pipefail

AGENTIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$AGENTIC_DIR"

echo "== Stage 1: build wheel =="
uv build --wheel --out-dir "$WORK/dist"
WHEEL="$(ls "$WORK"/dist/agentic-*.whl)"
echo "built: $WHEEL"

echo "== Stage 2: RUNTIME-ONLY install + sandbox-contract smoke (NO dev extras) =="
uv venv "$WORK/venv-rt" >/dev/null
RT="$WORK/venv-rt/bin/python"
uv pip install --python "$RT" "$WHEEL"
# agentic imports, AND every third-party IMPORT_REGISTRY lib (the FunctionBlock /
# copilot sandbox contract) imports on the runtime closure. No src/ on path here,
# so `agentic` resolves to the installed wheel.
"$RT" - <<'PY'
import importlib
import agentic  # noqa: F401  (resolves to the installed wheel)
from agentic.workflow.blocks.function import IMPORT_REGISTRY

third_party = [m for _k, (m, _a, is_stdlib) in IMPORT_REGISTRY.items() if not is_stdlib]
missing = []
for module_name in third_party:
    try:
        importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{module_name} ({type(e).__name__}: {e})")
if missing:
    raise SystemExit(
        "SANDBOX CONTRACT BROKEN — not importable on the runtime wheel:\n  "
        + "\n  ".join(missing)
    )
print(f"sandbox contract OK: {len(third_party)} third-party libs import on runtime deps")
PY

echo "== Stage 3: dev extras + offline test suite =="
uv venv "$WORK/venv-dev" >/dev/null
DEV="$WORK/venv-dev/bin/python"
uv pip install --python "$DEV" "$WHEEL"
uv pip install --python "$DEV" pytest pytest-asyncio pytest-mock
# Run against the installed wheel (cwd = $WORK so `agentic` never resolves to src/).
cd "$WORK"
"$DEV" -m pytest "$AGENTIC_DIR/tests" -q --import-mode=importlib

echo "== SEPARABILITY GATE PASSED =="
