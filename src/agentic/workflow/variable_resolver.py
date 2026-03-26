"""
Variable resolver for workflow block configs.

Resolves references like ``<blockId.output>`` or ``<blockId.output.field>``
in block config values against previously computed block outputs.
"""

from __future__ import annotations

import re
from typing import Any

# Pattern: <blockId.output> or <Block Name.output.nested.field>
# Allows spaces and hyphens in the block name segment (first segment before the dot).
_VAR_PATTERN = re.compile(r"<([\w -]+\.[\w.]+)>")


def resolve_variable(ref: str, block_outputs: dict[str, Any]) -> Any:
    """Resolve a single dotted reference against block outputs.

    Args:
        ref: Dotted path like ``start.output`` or ``start.output.text``.
        block_outputs: Mapping of block_id -> output dict.

    Returns:
        The resolved value, or the original ``<ref>`` string if unresolvable.
    """
    parts = ref.split(".")
    if len(parts) < 2:
        return f"<{ref}>"

    block_id = parts[0]
    output = block_outputs.get(block_id)
    if output is None:
        return f"<{ref}>"

    # Walk remaining path
    current = output
    for part in parts[1:]:
        if isinstance(current, dict):
            current = current.get(part, f"<{ref}>")
        else:
            return f"<{ref}>"
    return current


def resolve_string(template: str, block_outputs: dict[str, Any]) -> str:
    """Resolve all ``<ref>`` placeholders in a string template."""

    def _replace(match: re.Match) -> str:
        val = resolve_variable(match.group(1), block_outputs)
        return str(val)

    return _VAR_PATTERN.sub(_replace, template)


def resolve_value(value: Any, block_outputs: dict[str, Any]) -> Any:
    """Recursively resolve variable references in any JSON-like value."""
    if isinstance(value, str):
        # If entire string is a single reference, return raw value (not str-coerced)
        m = _VAR_PATTERN.fullmatch(value)
        if m:
            return resolve_variable(m.group(1), block_outputs)
        return resolve_string(value, block_outputs)
    if isinstance(value, dict):
        return {k: resolve_value(v, block_outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_value(v, block_outputs) for v in value]
    return value
