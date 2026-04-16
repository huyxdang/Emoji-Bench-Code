import pytest

from emoji_bench.domain.expressions import BinaryOp, SymbolLiteral, UnaryTransform
from emoji_bench.domain.interpreter import evaluate, evaluate_binary, evaluate_transform
from emoji_bench.domain.types import (
    DerivedOperation,
    FormalSystem,
    OperationTable,
    Symbol,
    TransformationRule,
)


def _zelta_system() -> FormalSystem:
    """Build the Zelta Algebra from the README spec."""
    a, b, c = Symbol("🦩"), Symbol("🧲"), Symbol("🪣")
    symbols = (a, b, c)

    # ⊕ table from README:
    #     🦩  🧲  🪣
    # 🦩  🧲  🪣  🦩
    # 🧲  🪣  🦩  🧲
    # 🪣  🦩  🧲  🪣
    table = {
        (a, a): b, (a, b): c, (a, c): a,
        (b, a): c, (b, b): a, (b, c): b,
        (c, a): a, (c, b): b, (c, c): c,
    }
    base_op = OperationTable(name="op0", symbol_id="⊕", symbols=symbols, table=table)

    # inv: 🦩↔🧲, 🪣→🪣 (the x→2x automorphism of Z/3Z)
    inv = TransformationRule(
        name="inv",
        mapping={a: b, b: a, c: c},
        distributes_over=("op0",),
    )

    # Derived: x ⊗ y = (x ⊕ y) ⊕ x
    derived = DerivedOperation(
        name="dop0",
        symbol_id="⊗",
        template_id="compose_left",
        base_ops=("op0",),
        transform_name=None,
    )

    return FormalSystem(
        name="Zelta Algebra",
        seed=0,
        symbols=symbols,
        base_operations=(base_op,),
        derived_operations=(derived,),
        transformations=(inv,),
    )


def test_evaluate_literal():
    system = _zelta_system()
    s = system.symbols[0]
    assert evaluate(SymbolLiteral(s), system) == s


def test_evaluate_base_op():
    system = _zelta_system()
    a, b, c = system.symbols
    # 🦩 ⊕ 🧲 = 🪣
    expr = BinaryOp("op0", SymbolLiteral(a), SymbolLiteral(b))
    assert evaluate(expr, system) == c


def test_evaluate_base_op_all_entries():
    """Verify several entries from the Zelta ⊕ table."""
    system = _zelta_system()
    a, b, c = system.symbols
    assert evaluate_binary("op0", a, a, system) == b  # 🦩⊕🦩 = 🧲
    assert evaluate_binary("op0", b, b, system) == a  # 🧲⊕🧲 = 🦩
    assert evaluate_binary("op0", c, c, system) == c  # 🪣⊕🪣 = 🪣
    assert evaluate_binary("op0", c, a, system) == a  # 🪣⊕🦩 = 🦩


def test_evaluate_transform():
    system = _zelta_system()
    a, b, c = system.symbols
    assert evaluate_transform("inv", a, system) == b  # inv(🦩) = 🧲
    assert evaluate_transform("inv", b, system) == a  # inv(🧲) = 🦩
    assert evaluate_transform("inv", c, system) == c  # inv(🪣) = 🪣


def test_evaluate_derived_compose_left():
    """x ⊗ y = (x ⊕ y) ⊕ x for the Zelta system."""
    system = _zelta_system()
    a, b, c = system.symbols

    # 🪣 ⊗ 🪣 = (🪣 ⊕ 🪣) ⊕ 🪣 = 🪣 ⊕ 🪣 = 🪣
    assert evaluate_binary("dop0", c, c, system) == c

    # 🦩 ⊗ 🧲 = (🦩 ⊕ 🧲) ⊕ 🦩 = 🪣 ⊕ 🦩 = 🦩
    assert evaluate_binary("dop0", a, b, system) == a


def test_evaluate_nested_expression():
    """(🦩 ⊕ 🧲) ⊗ 🪣 from the README example."""
    system = _zelta_system()
    a, b, c = system.symbols

    # Step 1: 🦩 ⊕ 🧲 = 🪣
    # Step 2: 🪣 ⊗ 🪣 = (🪣 ⊕ 🪣) ⊕ 🪣 = 🪣 ⊕ 🪣 = 🪣
    inner = BinaryOp("op0", SymbolLiteral(a), SymbolLiteral(b))
    outer = BinaryOp("dop0", inner, SymbolLiteral(c))
    assert evaluate(outer, system) == c


def test_evaluate_transform_expression():
    system = _zelta_system()
    a, b, c = system.symbols
    # inv(🦩 ⊕ 🧲) = inv(🪣) = 🪣 (🪣 is fixed point of inv)
    inner = BinaryOp("op0", SymbolLiteral(a), SymbolLiteral(b))
    expr = UnaryTransform("inv", inner)
    assert evaluate(expr, system) == c


def test_distribution_property_zelta():
    """Verify inv distributes over ⊕ in Zelta: inv(x⊕y) == inv(x) ⊕ inv(y)."""
    system = _zelta_system()
    for x in system.symbols:
        for y in system.symbols:
            # LHS: inv(x ⊕ y)
            xy = evaluate_binary("op0", x, y, system)
            lhs = evaluate_transform("inv", xy, system)
            # RHS: inv(x) ⊕ inv(y)
            inv_x = evaluate_transform("inv", x, system)
            inv_y = evaluate_transform("inv", y, system)
            rhs = evaluate_binary("op0", inv_x, inv_y, system)
            assert lhs == rhs, f"Distribution failed for ({x}, {y}): {lhs} != {rhs}"


def test_evaluate_binary_unknown_operation_raises():
    system = _zelta_system()
    a, b, _ = system.symbols
    with pytest.raises(ValueError, match="Unknown operation"):
        evaluate_binary("missing", a, b, system)


def test_evaluate_transform_unknown_transform_raises():
    system = _zelta_system()
    a, _, _ = system.symbols
    with pytest.raises(ValueError, match="Unknown transformation"):
        evaluate_transform("missing", a, system)


def test_evaluate_binary_unknown_derived_template_raises():
    system = _zelta_system()
    malformed = DerivedOperation(
        name="bad_dop",
        symbol_id="⊛",
        template_id="unknown_template",
        base_ops=("op0",),
        transform_name=None,
    )
    malformed_system = FormalSystem(
        name=system.name,
        seed=system.seed,
        symbols=system.symbols,
        base_operations=system.base_operations,
        derived_operations=(malformed,),
        transformations=system.transformations,
    )
    a, b, _ = system.symbols
    with pytest.raises(ValueError, match="Unknown derived operation template"):
        evaluate_binary("bad_dop", a, b, malformed_system)


def test_evaluate_binary_inv_compose_without_transform_name_raises():
    system = _zelta_system()
    malformed = DerivedOperation(
        name="bad_inv",
        symbol_id="⊛",
        template_id="inv_compose",
        base_ops=("op0",),
        transform_name=None,
    )
    malformed_system = FormalSystem(
        name=system.name,
        seed=system.seed,
        symbols=system.symbols,
        base_operations=system.base_operations,
        derived_operations=(malformed,),
        transformations=system.transformations,
    )
    a, b, _ = system.symbols
    with pytest.raises(AssertionError):
        evaluate_binary("bad_inv", a, b, malformed_system)
