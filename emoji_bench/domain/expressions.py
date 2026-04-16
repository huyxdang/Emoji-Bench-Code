from __future__ import annotations

import random
from dataclasses import dataclass

from emoji_bench.types import FormalSystem, Symbol


@dataclass(frozen=True)
class SymbolLiteral:
    symbol: Symbol


@dataclass(frozen=True)
class BinaryOp:
    op_name: str  # name of the operation (base or derived)
    left: Expression
    right: Expression


@dataclass(frozen=True)
class UnaryTransform:
    transform_name: str
    operand: Expression


Expression = SymbolLiteral | BinaryOp | UnaryTransform


def expr_to_str(expr: Expression) -> str:
    """Render an expression as a human-readable string.

    Binary sub-expressions are always parenthesized (no precedence for novel ops).
    """
    match expr:
        case SymbolLiteral(symbol):
            return symbol.emoji
        case BinaryOp(op_name, left, right):
            # Look up the symbol_id would require the system; use op_name as fallback.
            # The caller can pass a symbol_id mapping if needed.
            left_str = expr_to_str(left)
            right_str = expr_to_str(right)
            return f"({left_str} {op_name} {right_str})"
        case UnaryTransform(transform_name, operand):
            return f"{transform_name}({expr_to_str(operand)})"


def expr_to_str_with_system(expr: Expression, system: FormalSystem) -> str:
    """Render an expression using the system's operator symbols (e.g. ⊕ instead of op name)."""
    op_symbol_map: dict[str, str] = {}
    for op in system.base_operations:
        op_symbol_map[op.name] = op.symbol_id
    for op in system.derived_operations:
        op_symbol_map[op.name] = op.symbol_id

    def _render(e: Expression) -> str:
        match e:
            case SymbolLiteral(symbol):
                return symbol.emoji
            case BinaryOp(op_name, left, right):
                sym = op_symbol_map.get(op_name, op_name)
                return f"({_render(left)} {sym} {_render(right)})"
            case UnaryTransform(transform_name, operand):
                return f"{transform_name}({_render(operand)})"

    return _render(expr)


def random_expression(
    system: FormalSystem,
    depth: int,
    rng: random.Random,
) -> Expression:
    """Generate a random expression tree of a given depth.

    depth=0 produces a literal. depth=N produces a tree with max depth N.
    """
    if depth <= 0:
        return SymbolLiteral(rng.choice(system.symbols))

    # Collect available operation names
    op_names = [op.name for op in system.base_operations] + [
        op.name for op in system.derived_operations
    ]
    transform_names = [t.name for t in system.transformations]

    # Decide what to build: binary op, unary transform, or literal
    choices: list[str] = ["binary"] * len(op_names)
    if transform_names:
        choices.append("transform")
    choices.append("literal")

    pick = rng.choice(choices)

    if pick == "literal":
        return SymbolLiteral(rng.choice(system.symbols))
    elif pick == "transform":
        return UnaryTransform(
            transform_name=rng.choice(transform_names),
            operand=random_expression(system, depth - 1, rng),
        )
    else:
        return BinaryOp(
            op_name=rng.choice(op_names),
            left=random_expression(system, depth - 1, rng),
            right=random_expression(system, depth - 1, rng),
        )
