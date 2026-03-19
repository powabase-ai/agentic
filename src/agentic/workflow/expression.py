"""
Safe expression evaluator for condition blocks.

Uses AST parsing with a whitelist of allowed node types to prevent
arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# Supported comparison operators
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BOOL_OPS = {
    ast.And: all,
    ast.Or: any,
}

_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_ALLOWED_NODES = {
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Index,  # py3.8 compat
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
}


def _validate_ast(node: ast.AST) -> None:
    """Raise ValueError if the AST contains disallowed node types."""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Undefined variable: {node.id}")

    if isinstance(node, ast.Attribute):
        obj = _eval_node(node.value, variables)
        if isinstance(obj, dict):
            return obj.get(node.attr)
        return getattr(obj, node.attr, None)

    if isinstance(node, ast.Subscript):
        obj = _eval_node(node.value, variables)
        # Python 3.9+: slice is the index directly
        idx = _eval_node(node.slice, variables)
        return obj[idx]

    if isinstance(node, ast.BoolOp):
        fn = _BOOL_OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported bool op: {type(node.op).__name__}")
        return fn(_eval_node(v, variables) for v in node.values)

    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported bin op: {type(node.op).__name__}")
        return fn(_eval_node(node.left, variables), _eval_node(node.right, variables))

    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return fn(_eval_node(node.operand, variables))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op_node, comparator in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op_node))
            if not fn:
                raise ValueError(f"Unsupported compare: {type(op_node).__name__}")
            right = _eval_node(comparator, variables)
            if not fn(left, right):
                return False
            left = right
        return True

    raise ValueError(f"Cannot evaluate node: {type(node).__name__}")


def evaluate_expression(expression: str, variables: dict[str, Any]) -> Any:
    """Safely evaluate a Python expression with the given variables.

    Only comparison, boolean, arithmetic, attribute access, and subscript
    operations are allowed. No function calls.

    Args:
        expression: A Python expression string.
        variables: Variable bindings available in the expression.

    Returns:
        The result of evaluating the expression.

    Raises:
        ValueError: If the expression contains disallowed constructs.
    """
    tree = ast.parse(expression, mode="eval")
    _validate_ast(tree)
    return _eval_node(tree, variables)
