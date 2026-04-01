"""Runtime input rule evaluation for tool execution."""

from __future__ import annotations

import json
import re
from typing import Any


def evaluate_rules(
    arguments: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Evaluate runtime input rules against tool arguments.

    Returns (allowed, denial_reason). First matching deny rule wins.
    """
    for rule in rules:
        condition = rule.get("condition", "")
        action = rule.get("action", "deny")
        message = rule.get("message", "Blocked by rule")

        if _evaluate_condition(condition, arguments):
            if action == "deny":
                return False, message

    return True, None


def _evaluate_condition(condition: str, arguments: dict[str, Any]) -> bool:
    """Parse and evaluate a single condition string.

    Format: "field_name [NOT] OPERATOR value"
    Supported operators: CONTAINS, STARTS_WITH, MATCHES, IN
    """
    condition = condition.strip()

    # Use regex to parse: field (NOT)? OPERATOR value
    # Match longest operators first to avoid partial matches (NOT IN before IN, etc.)
    pattern = re.compile(r"^(\w+)\s+(NOT\s+)?(CONTAINS|STARTS_WITH|MATCHES|IN)\s+(.+)$")
    m = pattern.match(condition)
    if not m:
        return False  # Unparseable condition doesn't match

    field = m.group(1)
    negated = m.group(2) is not None
    op = m.group(3)
    raw_value = m.group(4).strip()

    # Strip surrounding quotes for string values (not for IN lists)
    if op != "IN":
        raw_value = raw_value.strip("'\"")

    field_value = str(arguments.get(field, ""))
    result = _check_positive(op, field_value, raw_value)
    return not result if negated else result


def _check_positive(op: str, field_value: str, value: str) -> bool:
    if op == "CONTAINS":
        return value in field_value
    if op == "STARTS_WITH":
        return field_value.startswith(value)
    if op == "MATCHES":
        if len(value) > 500:  # Reject overly complex patterns
            return False
        try:
            pattern = re.compile(value, re.DOTALL)
            # Use fullmatch with a simpler approach to limit backtracking
            return bool(pattern.search(field_value[:10000]))  # Limit input length too
        except re.error:
            return False
    if op == "IN":
        try:
            # Support both JSON-style ["a", "b"] and Python-style ['a', 'b']
            normalized = value.replace("'", '"')
            items = json.loads(normalized)
            return field_value in items
        except (json.JSONDecodeError, TypeError):
            return False
    return False
