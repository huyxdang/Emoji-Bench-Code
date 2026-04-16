import random

import pytest

from emoji_bench.domain.chain_generator import (
    count_reducible_nodes,
    find_leftmost_innermost,
    generate_chain,
    get_at_path,
    reduce_expression,
    replace_at_path,
)
from emoji_bench.domain.expressions import (
    BinaryOp,
    SymbolLiteral,
    UnaryTransform,
    random_expression,
)
from emoji_bench.domain.generator import generate_system
from emoji_bench.domain.interpreter import evaluate
from emoji_bench.domain.types import (
    DerivedOperation,
    FormalSystem,
    OperationTable,
    Symbol,
    TransformationRule,
)


def _system_with_template(template_id: str) -> FormalSystem:
    a, b, c = Symbol("🦩"), Symbol("🧲"), Symbol("🪣")
    symbols = (a, b, c)
    table = {
        (a, a): b, (a, b): c, (a, c): a,
        (b, a): c, (b, b): a, (b, c): b,
        (c, a): a, (c, b): b, (c, c): c,
    }
    base_op = OperationTable(name="op0", symbol_id="⊕", symbols=symbols, table=table)
    inv = TransformationRule(
        name="inv",
        mapping={a: b, b: a, c: c},
        distributes_over=("op0",),
    )
    derived = DerivedOperation(
        name="dop0",
        symbol_id="⊗",
        template_id=template_id,
        base_ops=("op0",),
        transform_name="inv" if template_id == "inv_compose" else None,
    )
    transformations = (inv,) if template_id == "inv_compose" else ()
    return FormalSystem(
        name=f"{template_id} System",
        seed=0,
        symbols=symbols,
        base_operations=(base_op,),
        derived_operations=(derived,),
        transformations=transformations,
    )


# --- Path helper tests ---


def test_get_at_path_root():
    s = SymbolLiteral(Symbol("🦩"))
    assert get_at_path(s, ()) is s


def test_get_at_path_binary():
    a, b = SymbolLiteral(Symbol("🦩")), SymbolLiteral(Symbol("🧲"))
    expr = BinaryOp("op", a, b)
    assert get_at_path(expr, (0,)) is a
    assert get_at_path(expr, (1,)) is b


def test_get_at_path_nested():
    a = SymbolLiteral(Symbol("🦩"))
    b = SymbolLiteral(Symbol("🧲"))
    c = SymbolLiteral(Symbol("🪣"))
    inner = BinaryOp("op", a, b)
    outer = BinaryOp("op", inner, c)
    assert get_at_path(outer, (0, 0)) is a
    assert get_at_path(outer, (0, 1)) is b
    assert get_at_path(outer, (1,)) is c


def test_replace_at_path_root():
    old = SymbolLiteral(Symbol("🦩"))
    new = SymbolLiteral(Symbol("🧲"))
    assert replace_at_path(old, (), new) is new


def test_replace_at_path_binary():
    a = SymbolLiteral(Symbol("🦩"))
    b = SymbolLiteral(Symbol("🧲"))
    c = SymbolLiteral(Symbol("🪣"))
    expr = BinaryOp("op", a, b)
    result = replace_at_path(expr, (0,), c)
    assert isinstance(result, BinaryOp)
    assert result.left is c
    assert result.right is b


def test_replace_at_path_nested():
    a = SymbolLiteral(Symbol("🦩"))
    b = SymbolLiteral(Symbol("🧲"))
    c = SymbolLiteral(Symbol("🪣"))
    r = SymbolLiteral(Symbol("X"))
    inner = BinaryOp("op", a, b)
    outer = BinaryOp("op", inner, c)
    result = replace_at_path(outer, (0, 1), r)
    assert get_at_path(result, (0, 1)) is r
    assert get_at_path(result, (0, 0)) is a
    assert get_at_path(result, (1,)) is c


# --- find_leftmost_innermost tests ---


def test_find_lmi_literal_returns_none():
    assert find_leftmost_innermost(SymbolLiteral(Symbol("🦩"))) is None


def test_find_lmi_simple_binary():
    a, b = SymbolLiteral(Symbol("🦩")), SymbolLiteral(Symbol("🧲"))
    expr = BinaryOp("op", a, b)
    assert find_leftmost_innermost(expr) == ()


def test_find_lmi_nested_left():
    a = SymbolLiteral(Symbol("🦩"))
    b = SymbolLiteral(Symbol("🧲"))
    c = SymbolLiteral(Symbol("🪣"))
    inner = BinaryOp("op", a, b)
    outer = BinaryOp("op", inner, c)
    # Leftmost-innermost is the inner (left child)
    assert find_leftmost_innermost(outer) == (0,)


def test_find_lmi_transform():
    a = SymbolLiteral(Symbol("🦩"))
    expr = UnaryTransform("inv", a)
    assert find_leftmost_innermost(expr) == ()


def test_find_lmi_prefers_left():
    a = SymbolLiteral(Symbol("🦩"))
    b = SymbolLiteral(Symbol("🧲"))
    c = SymbolLiteral(Symbol("🪣"))
    d = SymbolLiteral(Symbol("X"))
    left = BinaryOp("op", a, b)
    right = BinaryOp("op", c, d)
    expr = BinaryOp("op", left, right)
    # Should pick left child first
    assert find_leftmost_innermost(expr) == (0,)


# --- Reducer tests ---


def test_reduce_produces_correct_result():
    """Final result of reduction must match evaluate()."""
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    rng = random.Random(7)
    for _ in range(20):
        expr = random_expression(system, depth=3, rng=rng)
        expected = evaluate(expr, system)
        steps = reduce_expression(expr, system)
        if isinstance(expr, SymbolLiteral):
            assert len(steps) == 0  # already reduced
        else:
            assert len(steps) > 0
            final = steps[-1].after
            assert isinstance(final, SymbolLiteral)
            assert final.symbol == expected


def test_each_step_is_valid():
    """For non-expansion steps, evaluate(reduced_subexpr) must equal result_symbol."""
    system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77)
    rng = random.Random(42)
    for _ in range(10):
        expr = random_expression(system, depth=3, rng=rng)
        steps = reduce_expression(expr, system)
        for step in steps:
            if step.rule_type != "derived_op":
                assert step.result_symbol is not None
                actual = evaluate(step.reduced_subexpr, system)
                assert step.result_symbol == actual, (
                    f"Step {step.step_number}: expected {actual}, got {step.result_symbol}"
                )


def test_consecutive_steps_connect():
    """step[i].after must equal step[i+1].before."""
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    rng = random.Random(99)
    expr = random_expression(system, depth=4, rng=rng)
    steps = reduce_expression(expr, system)
    for i in range(len(steps) - 1):
        assert steps[i].after == steps[i + 1].before, (
            f"Steps {i + 1} and {i + 2} are not connected"
        )


def test_each_step_preserves_expression_value():
    """Every correct rewrite should preserve the full expression's final value."""
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    rng = random.Random(123)

    for _ in range(15):
        expr = random_expression(system, depth=4, rng=rng)
        steps = reduce_expression(expr, system)
        for step in steps:
            assert evaluate(step.before, system) == evaluate(step.after, system)


def test_first_step_before_is_starting_expr():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    rng = random.Random(1)
    expr = random_expression(system, depth=3, rng=rng)
    steps = reduce_expression(expr, system)
    assert steps[0].before == expr


def test_final_step_is_literal():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    # Use a seed that produces a non-trivial expression
    for seed in range(100):
        rng = random.Random(seed)
        expr = random_expression(system, depth=3, rng=rng)
        if not isinstance(expr, SymbolLiteral):
            steps = reduce_expression(expr, system)
            assert isinstance(steps[-1].after, SymbolLiteral)
            return
    assert False, "Never generated a non-trivial expression"


def test_steps_are_leftmost_innermost():
    """Reduced subexprs must have all-literal operands (they are innermost)."""
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    rng = random.Random(3)
    expr = random_expression(system, depth=4, rng=rng)
    steps = reduce_expression(expr, system)
    for step in steps:
        match step.reduced_subexpr:
            case BinaryOp(_, left, right):
                assert isinstance(left, SymbolLiteral)
                assert isinstance(right, SymbolLiteral)
            case UnaryTransform(_, operand):
                assert isinstance(operand, SymbolLiteral)


def test_derived_op_expansion_step():
    """Derived ops should produce an expansion step followed by base-op reductions."""
    system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77)
    rng = random.Random(42)
    # Keep generating until we get a chain with a derived op
    for _ in range(50):
        expr = random_expression(system, depth=2, rng=rng)
        steps = reduce_expression(expr, system)
        expansion_steps = [s for s in steps if s.rule_type == "derived_op"]
        if expansion_steps:
            step = expansion_steps[0]
            assert step.expanded_to is not None
            assert step.result_symbol is None
            assert isinstance(step.reduced_subexpr, BinaryOp)
            return
    # If we never found a derived op, that's okay for some seeds
    # but unlikely — at least warn
    assert False, "Never generated a derived op step in 50 attempts"


def test_compose_left_expands_exactly():
    system = _system_with_template("compose_left")
    derived = system.derived_operations[0]
    base_op_name = derived.base_ops[0]
    a, b = system.symbols[:2]
    expr = BinaryOp(derived.name, SymbolLiteral(a), SymbolLiteral(b))

    steps = reduce_expression(expr, system)

    assert steps[0].expanded_to == BinaryOp(
        base_op_name,
        BinaryOp(base_op_name, SymbolLiteral(a), SymbolLiteral(b)),
        SymbolLiteral(a),
    )


def test_double_left_expands_exactly():
    system = _system_with_template("double_left")
    derived = system.derived_operations[0]
    base_op_name = derived.base_ops[0]
    a, b = system.symbols[:2]
    expr = BinaryOp(derived.name, SymbolLiteral(a), SymbolLiteral(b))

    steps = reduce_expression(expr, system)

    assert steps[0].expanded_to == BinaryOp(
        base_op_name,
        BinaryOp(base_op_name, SymbolLiteral(a), SymbolLiteral(a)),
        SymbolLiteral(b),
    )


def test_inv_compose_expands_exactly():
    system = _system_with_template("inv_compose")
    derived = system.derived_operations[0]
    base_op_name = derived.base_ops[0]
    a, b = system.symbols[:2]
    expr = BinaryOp(derived.name, SymbolLiteral(a), SymbolLiteral(b))

    steps = reduce_expression(expr, system)

    assert steps[0].expanded_to == UnaryTransform(
        derived.transform_name,
        BinaryOp(base_op_name, SymbolLiteral(a), SymbolLiteral(b)),
    )


# --- generate_chain tests ---


def test_chain_deterministic():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    c1 = generate_chain(system, length=5, seed=7)
    c2 = generate_chain(system, length=5, seed=7)
    assert c1.starting_expression == c2.starting_expression
    assert c1.final_result == c2.final_result
    assert len(c1.steps) == len(c2.steps)
    assert c1.seed == c2.seed == 7


def test_chain_length_within_tolerance():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    for target in [3, 5, 7, 10]:
        chain = generate_chain(system, length=target, rng=random.Random(target))
        assert abs(len(chain.steps) - target) <= 3, (
            f"Requested {target} steps, got {len(chain.steps)}"
        )


def test_chain_positive_length_guarantees_at_least_one_step():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    for seed in range(50):
        chain = generate_chain(system, length=1, seed=seed)
        assert len(chain.steps) >= 1


def test_chain_at_all_difficulties():
    configs = [
        dict(n_symbols=3, n_base_ops=1, random_seed=42),
        dict(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77),
        dict(n_symbols=5, n_base_ops=2, n_derived_ops=1, n_transformations=1, random_seed=123),
        dict(n_symbols=6, n_base_ops=2, n_derived_ops=2, n_transformations=2, random_seed=200),
    ]
    for cfg in configs:
        system = generate_system(**cfg)
        chain = generate_chain(system, length=5, rng=random.Random(1))
        expected = evaluate(chain.starting_expression, system)
        assert chain.final_result == expected


def test_chain_final_result_matches_evaluate():
    """Stress test: chain's final_result must match evaluate(starting_expression)."""
    system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77)
    for seed in range(30):
        chain = generate_chain(system, length=5, rng=random.Random(seed))
        expected = evaluate(chain.starting_expression, system)
        assert chain.final_result == expected, f"Seed {seed}: {chain.final_result} != {expected}"


def test_chain_seed_is_none_when_rng_is_used():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    assert chain.seed is None


def test_chain_step_numbers_are_contiguous():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=7, seed=42)
    assert [step.step_number for step in chain.steps] == list(range(1, len(chain.steps) + 1))


def test_chain_last_step_matches_final_result():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=7, seed=42)
    assert isinstance(chain.steps[-1].after, SymbolLiteral)
    assert chain.steps[-1].after.symbol == chain.final_result


def test_generate_chain_rejects_invalid_length():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    with pytest.raises(ValueError, match="length must be >= 1"):
        generate_chain(system, length=0, seed=1)


def test_generate_chain_rejects_seed_and_rng_together():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    with pytest.raises(ValueError, match="either seed or rng"):
        generate_chain(system, length=3, seed=1, rng=random.Random(1))


def test_generate_chain_requires_seed_or_rng():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    with pytest.raises(ValueError, match="either seed or rng"):
        generate_chain(system, length=3)
