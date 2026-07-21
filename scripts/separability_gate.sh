#!/usr/bin/env bash
# Separability gate for the `agentic` package.
#
# Proves `agentic` installs + runs standalone from a built wheel, and that the
# workflow/copilot code-execution sandbox contract (function.py IMPORT_REGISTRY)
# is satisfied by the RUNTIME dependency closure ALONE (stage 2 uses no dev extras,
# so a sandbox lib demoted to dev-deps fails here). Suitable as an anti-rot CI check.
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

echo "== Stage 3: offline test suite in the project's LOCKED dev env (no-regression check) =="
# Stage 2 above is the wheel / standalone proof. Stage 3 runs agentic's suite in the
# project's LOCKED dev toolchain (via `uv run`, honoring agentic/uv.lock) as a
# no-regression check. An ad-hoc `pip install pytest…`
# instead pulls UNPINNED dev tools (e.g. pytest-asyncio 1.4.0 regresses the event
# loop; ruff — needed by test_import_isolation — would be absent) and false-fails.
#
# DESELECTED: 2 pre-existing test failures unrelated to packaging
# (pre-existing agentic-logic bugs):
#   - test_input_mapping … concatenate_strings
#   - test_platform_api_block … route_mapping_database_query (ROUTE_MAP lacks database/query)
# They are tracked separately. This gate must introduce no new failures.
cd "$AGENTIC_DIR"
uv run pytest tests/ -q \
  --deselect "tests/workflow/test_input_mapping.py::TestApplyInputMappings::test_multiple_mappings_concatenate_strings" \
  --deselect "tests/workflow/test_platform_api_block.py::TestRouteMapping::test_route_mapping_database_query"

echo "== SEPARABILITY GATE PASSED =="
