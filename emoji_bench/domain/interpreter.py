from __future__ import annotations

from emoji_bench.expressions import BinaryOp, Expression, SymbolLiteral, UnaryTransform
from emoji_bench.types import FormalSystem, Symbol


def evaluate(expr: Expression, system: FormalSystem) -> Symbol:
    """Recursively evaluate an expression tree to a single symbol."""
    match expr:
        case SymbolLiteral(symbol):
            return symbol

        case BinaryOp(op_name, left, right):
            left_val = evaluate(left, system)
            right_val = evaluate(right, system)
            return evaluate_binary(op_name, left_val, right_val, system)

        case UnaryTransform(transform_name, operand):
            operand_val = evaluate(operand, system)
            return evaluate_transform(transform_name, operand_val, system)

    raise ValueError(f"Unknown expression type: {type(expr)}")


def evaluate_binary(
    op_name: str, left: Symbol, right: Symbol, system: FormalSystem
) -> Symbol:
    """Evaluate a binary operation on two resolved symbols."""
    # Check base operations
    for base_op in system.base_operations:
        if base_op.name == op_name:
            return base_op.table[(left, right)]

    # Check derived operations
    for derived_op in system.derived_operations:
        if derived_op.name == op_name:
            return _evaluate_derived(derived_op.template_id, derived_op, left, right, system)

    raise ValueError(f"Unknown operation: {op_name}")


def evaluate_transform(
    transform_name: str, symbol: Symbol, system: FormalSystem
) -> Symbol:
    """Apply a transformation to a resolved symbol."""
    for transform in system.transformations:
        if transform.name == transform_name:
            return transform.mapping[symbol]

    raise ValueError(f"Unknown transformation: {transform_name}")


def _evaluate_derived(
    template_id: str,
    derived_op: object,
    left: Symbol,
    right: Symbol,
    system: FormalSystem,
) -> Symbol:
    """Expand and evaluate a derived operation template."""
    from emoji_bench.types import DerivedOperation

    assert isinstance(derived_op, DerivedOperation)
    base_op_name = derived_op.base_ops[0]

    match template_id:
        case "compose_left":
            # x ⊗ y = (x ⊕ y) ⊕ x
            intermediate = evaluate_binary(base_op_name, left, right, system)
            return evaluate_binary(base_op_name, intermediate, left, system)

        case "inv_compose":
            # x ⊗ y = inv(x ⊕ y)
            assert derived_op.transform_name is not None
            intermediate = evaluate_binary(base_op_name, left, right, system)
            return evaluate_transform(derived_op.transform_name, intermediate, system)

        case "double_left":
            # x ⊗ y = (x ⊕ x) ⊕ y
            doubled = evaluate_binary(base_op_name, left, left, system)
            return evaluate_binary(base_op_name, doubled, right, system)

    raise ValueError(f"Unknown derived operation template: {template_id}")
