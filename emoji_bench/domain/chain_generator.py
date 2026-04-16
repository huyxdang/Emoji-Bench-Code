from __future__ import annotations

import random

from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import (
    BinaryOp,
    Expression,
    SymbolLiteral,
    UnaryTransform,
    random_expression,
)
from emoji_bench.types import DerivedOperation, FormalSystem

# Path is a tuple of ints: 0 = left/operand, 1 = right (for BinaryOp)
Path = tuple[int, ...]


# --- Path-based tree helpers ---


def get_at_path(expr: Expression, path: Path) -> Expression:
    """Retrieve the subexpression at a path."""
    node = expr
    for idx in path:
        match node:
            case BinaryOp(_, left, right):
                node = left if idx == 0 else right
            case UnaryTransform(_, operand):
                node = operand  # idx must be 0
            case _:
                raise IndexError(f"Cannot descend into {type(node)} at index {idx}")
    return node


def replace_at_path(expr: Expression, path: Path, replacement: Expression) -> Expression:
    """Return a new expression tree with the node at `path` replaced."""
    if not path:
        return replacement

    idx, rest = path[0], path[1:]

    match expr:
        case BinaryOp(op_name, left, right):
            if idx == 0:
                return BinaryOp(op_name, replace_at_path(left, rest, replacement), right)
            else:
                return BinaryOp(op_name, left, replace_at_path(right, rest, replacement))
        case UnaryTransform(name, operand):
            return UnaryTransform(name, replace_at_path(operand, rest, replacement))
        case _:
            raise IndexError(f"Cannot descend into {type(expr)} at index {idx}")


def find_leftmost_innermost(expr: Expression) -> Path | None:
    """Find the path to the leftmost-innermost reducible subexpression.

    A reducible subexpression is a BinaryOp or UnaryTransform whose
    immediate children are all SymbolLiterals.
    """
    match expr:
        case SymbolLiteral():
            return None

        case BinaryOp(_, left, right):
            # Try left subtree first (leftmost)
            left_path = find_leftmost_innermost(left)
            if left_path is not None:
                return (0, *left_path)

            # Then right subtree
            right_path = find_leftmost_innermost(right)
            if right_path is not None:
                return (1, *right_path)

            # If both children are literals, this node is reducible
            if isinstance(left, SymbolLiteral) and isinstance(right, SymbolLiteral):
                return ()

            return None

        case UnaryTransform(_, operand):
            # Try operand first
            operand_path = find_leftmost_innermost(operand)
            if operand_path is not None:
                return (0, *operand_path)

            # If operand is a literal, this node is reducible
            if isinstance(operand, SymbolLiteral):
                return ()

            return None

    return None


def count_reducible_nodes(expr: Expression) -> int:
    """Count the number of reducible nodes in an expression.

    This equals the number of reduction steps needed to fully evaluate it.
    For derived ops, the expansion adds extra steps, so we account for that.
    """
    match expr:
        case SymbolLiteral():
            return 0
        case BinaryOp(_, left, right):
            return 1 + count_reducible_nodes(left) + count_reducible_nodes(right)
        case UnaryTransform(_, operand):
            return 1 + count_reducible_nodes(operand)
    return 0


# --- Derived operation expansion ---


def _expand_derived_op(
    derived_op: DerivedOperation,
    left: SymbolLiteral,
    right: SymbolLiteral,
    system: FormalSystem,
) -> Expression:
    """Expand a derived operation into its base-op equivalent expression tree."""
    base_op_name = derived_op.base_ops[0]

    match derived_op.template_id:
        case "compose_left":
            # x ⊗ y = (x ⊕ y) ⊕ x
            return BinaryOp(base_op_name, BinaryOp(base_op_name, left, right), left)
        case "inv_compose":
            # x ⊗ y = inv(x ⊕ y)
            assert derived_op.transform_name is not None
            return UnaryTransform(
                derived_op.transform_name,
                BinaryOp(base_op_name, left, right),
            )
        case "double_left":
            # x ⊗ y = (x ⊕ x) ⊕ y
            return BinaryOp(base_op_name, BinaryOp(base_op_name, left, left), right)

    raise ValueError(f"Unknown derived template: {derived_op.template_id}")


def _get_derived_op(op_name: str, system: FormalSystem) -> DerivedOperation | None:
    """Look up a derived operation by name, or return None if it's a base op."""
    for dop in system.derived_operations:
        if dop.name == op_name:
            return dop
    return None


def _get_rule_display(op_name: str, system: FormalSystem) -> tuple[str, str]:
    """Get (rule_used, rule_type) display strings for an operation or transform."""
    for op in system.base_operations:
        if op.name == op_name:
            return f"{op.symbol_id} table", "base_op"
    for dop in system.derived_operations:
        if dop.name == op_name:
            return f"definition of {dop.symbol_id}", "derived_op"
    for tr in system.transformations:
        if tr.name == op_name:
            return tr.name, "transform"
    raise ValueError(f"Unknown operation/transform: {op_name}")


# --- Core reducer ---


def reduce_expression(
    expr: Expression, system: FormalSystem
) -> tuple[ChainStep, ...]:
    """Fully reduce an expression step by step using leftmost-innermost order.

    Derived operations are expanded into their base-op equivalents before
    reducing, producing a "by definition of ⊗" step followed by base-op steps.
    """
    steps: list[ChainStep] = []
    current = expr
    step_num = 1

    while True:
        path = find_leftmost_innermost(current)
        if path is None:
            break

        subexpr = get_at_path(current, path)

        match subexpr:
            case BinaryOp(op_name, left, right) if (
                isinstance(left, SymbolLiteral) and isinstance(right, SymbolLiteral)
            ):
                dop = _get_derived_op(op_name, system)
                if dop is not None:
                    # Derived op: expand into base-op tree
                    expanded = _expand_derived_op(dop, left, right, system)
                    after = replace_at_path(current, path, expanded)
                    rule_display = f"definition of {dop.symbol_id}"
                    steps.append(ChainStep(
                        step_number=step_num,
                        before=current,
                        reduced_subexpr=subexpr,
                        result_symbol=None,
                        after=after,
                        rule_used=rule_display,
                        rule_type="derived_op",
                        expanded_to=expanded,
                    ))
                    step_num += 1
                    current = after
                    continue

                # Base operation: evaluate directly
                from emoji_bench.interpreter import evaluate_binary
                result = evaluate_binary(op_name, left.symbol, right.symbol, system)
                replacement = SymbolLiteral(result)
                after = replace_at_path(current, path, replacement)
                rule_used, rule_type = _get_rule_display(op_name, system)
                steps.append(ChainStep(
                    step_number=step_num,
                    before=current,
                    reduced_subexpr=subexpr,
                    result_symbol=result,
                    after=after,
                    rule_used=rule_used,
                    rule_type=rule_type,
                ))

            case UnaryTransform(transform_name, operand) if isinstance(operand, SymbolLiteral):
                from emoji_bench.interpreter import evaluate_transform
                result = evaluate_transform(transform_name, operand.symbol, system)
                replacement = SymbolLiteral(result)
                after = replace_at_path(current, path, replacement)
                rule_used, rule_type = _get_rule_display(transform_name, system)
                steps.append(ChainStep(
                    step_number=step_num,
                    before=current,
                    reduced_subexpr=subexpr,
                    result_symbol=result,
                    after=after,
                    rule_used=rule_used,
                    rule_type=rule_type,
                ))

            case _:
                raise RuntimeError(f"Unexpected reducible node: {subexpr}")

        step_num += 1
        current = after

    return tuple(steps)


# --- Expression builder with target step count ---


def _build_expression_with_target_steps(
    system: FormalSystem,
    target_steps: int,
    rng: random.Random,
) -> Expression:
    """Build a random expression targeting a specific number of reduction steps.

    Tries increasing depths until we get close to the target. Derived ops
    expand into multiple steps, so raw node count may differ from step count.
    """
    # Start with a rough depth estimate
    for depth in range(1, 10):
        for _ in range(20):
            expr = random_expression(system, depth, rng)
            if target_steps >= 1 and isinstance(expr, SymbolLiteral):
                continue
            # Rough estimate: count reducible nodes. Derived ops add ~1-2 extra steps.
            n_derived = sum(
                1 for _ in _iter_derived_ops(expr, system)
            )
            node_count = count_reducible_nodes(expr)
            estimated_steps = node_count + n_derived  # each derived op adds ~1 expansion step
            if abs(estimated_steps - target_steps) <= 2:
                return expr
            if estimated_steps > target_steps + 3:
                break  # depth too high, try next attempt at same depth

    # Fallback: keep sampling until we get a non-literal expression.
    depth = max(1, target_steps // 2)
    for _ in range(100):
        expr = random_expression(system, depth, rng)
        if target_steps < 1 or not isinstance(expr, SymbolLiteral):
            return expr

    raise RuntimeError("Failed to build a non-literal expression for a positive target length")


def _iter_derived_ops(expr: Expression, system: FormalSystem):
    """Yield all BinaryOp nodes that reference a derived operation."""
    match expr:
        case BinaryOp(op_name, left, right):
            if _get_derived_op(op_name, system) is not None:
                yield expr
            yield from _iter_derived_ops(left, system)
            yield from _iter_derived_ops(right, system)
        case UnaryTransform(_, operand):
            yield from _iter_derived_ops(operand, system)
        case SymbolLiteral():
            pass


# --- Public API ---


def generate_chain(
    system: FormalSystem,
    length: int,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> DerivationChain:
    """Generate a derivation chain of approximately `length` steps.

    Pass either an explicit seed or an existing Random instance. When
    a seed is provided, it is stored on the returned chain for
    reproducibility.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    if seed is not None:
        rng = random.Random(seed)
    elif rng is None:
        raise ValueError("Pass either seed or rng")

    assert rng is not None

    best_chain: tuple[ChainStep, ...] | None = None
    best_expr: Expression | None = None
    best_diff = float("inf")

    for _ in range(50):
        expr = _build_expression_with_target_steps(system, length, rng)
        steps = reduce_expression(expr, system)
        if length >= 1 and not steps:
            continue

        diff = abs(len(steps) - length)
        if diff < best_diff:
            best_diff = diff
            best_chain = steps
            best_expr = expr

        if diff <= 2:
            break

    assert best_chain is not None and best_expr is not None

    # Determine final result
    if best_chain:
        final_step = best_chain[-1]
        assert isinstance(final_step.after, SymbolLiteral)
        final_result = final_step.after.symbol
    else:
        # Expression was already a literal
        assert isinstance(best_expr, SymbolLiteral)
        final_result = best_expr.symbol

    return DerivationChain(
        starting_expression=best_expr,
        steps=best_chain,
        final_result=final_result,
        seed=seed,
    )
