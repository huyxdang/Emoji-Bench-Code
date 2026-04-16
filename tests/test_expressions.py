import random

from emoji_bench.domain.expressions import (
    BinaryOp,
    SymbolLiteral,
    UnaryTransform,
    expr_to_str,
    expr_to_str_with_system,
    random_expression,
)
from emoji_bench.domain.generator import generate_system
from emoji_bench.domain.types import Symbol


def test_expr_to_str_literal():
    s = Symbol("🦩")
    assert expr_to_str(SymbolLiteral(s)) == "🦩"


def test_expr_to_str_binary():
    a, b = Symbol("🦩"), Symbol("🧲")
    expr = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    assert expr_to_str(expr) == "(🦩 ⊕ 🧲)"


def test_expr_to_str_nested():
    a, b, c = Symbol("🦩"), Symbol("🧲"), Symbol("🪣")
    inner = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    outer = BinaryOp("⊗", inner, SymbolLiteral(c))
    assert expr_to_str(outer) == "((🦩 ⊕ 🧲) ⊗ 🪣)"


def test_expr_to_str_transform():
    s = Symbol("🦩")
    expr = UnaryTransform("inv", SymbolLiteral(s))
    assert expr_to_str(expr) == "inv(🦩)"


def test_expr_to_str_transform_nested():
    a, b = Symbol("🦩"), Symbol("🧲")
    inner = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    expr = UnaryTransform("inv", inner)
    assert expr_to_str(expr) == "inv((🦩 ⊕ 🧲))"


def test_expr_to_str_with_system_uses_display_symbols():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    a, b, c = system.symbols[:3]
    base_op = system.base_operations[0]
    derived_op = system.derived_operations[0]
    transform = system.transformations[0]

    expr = BinaryOp(
        base_op.name,
        SymbolLiteral(a),
        UnaryTransform(
            transform.name,
            BinaryOp(derived_op.name, SymbolLiteral(b), SymbolLiteral(c)),
        ),
    )

    rendered = expr_to_str_with_system(expr, system)

    assert rendered == (
        f"({a.emoji} {base_op.symbol_id} "
        f"{transform.name}(({b.emoji} {derived_op.symbol_id} {c.emoji})))"
    )
    assert base_op.name not in rendered
    assert derived_op.name not in rendered


def test_random_expression_depth_0():
    system = generate_system(n_symbols=3, random_seed=42)
    expr = random_expression(system, depth=0, rng=random.Random(1))
    assert isinstance(expr, SymbolLiteral)


def test_random_expression_produces_valid():
    """random_expression should only reference ops/transforms that exist in the system."""
    system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=99)
    rng = random.Random(7)
    op_names = {op.name for op in system.base_operations} | {op.name for op in system.derived_operations}
    tr_names = {t.name for t in system.transformations}

    for _ in range(20):
        expr = random_expression(system, depth=3, rng=rng)
        _check_names(expr, op_names, tr_names, set(system.symbols))


def test_random_expression_respects_max_depth():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=99
    )
    rng = random.Random(11)

    for max_depth in range(5):
        for _ in range(25):
            expr = random_expression(system, depth=max_depth, rng=rng)
            assert _expr_depth(expr) <= max_depth


def _check_names(expr, op_names, tr_names, symbols):
    match expr:
        case SymbolLiteral(s):
            assert s in symbols
        case BinaryOp(op_name, left, right):
            assert op_name in op_names
            _check_names(left, op_names, tr_names, symbols)
            _check_names(right, op_names, tr_names, symbols)
        case UnaryTransform(name, operand):
            assert name in tr_names
            _check_names(operand, op_names, tr_names, symbols)


def _expr_depth(expr):
    match expr:
        case SymbolLiteral():
            return 0
        case BinaryOp(_, left, right):
            return 1 + max(_expr_depth(left), _expr_depth(right))
        case UnaryTransform(_, operand):
            return 1 + _expr_depth(operand)
