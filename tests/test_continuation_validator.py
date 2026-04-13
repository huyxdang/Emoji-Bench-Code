import pytest

from emoji_bench.continuation_validator import (
    ParseError,
    _parse_expression,
    parse_continuation_steps,
    validate_derivation,
)
from emoji_bench.expressions import BinaryOp, SymbolLiteral, UnaryTransform
from emoji_bench.generator import generate_system


@pytest.fixture
def easy_system():
    return generate_system(
        n_symbols=3, n_base_ops=1, n_derived_ops=0, n_transformations=0,
        random_seed=11,
    )


@pytest.fixture
def medium_system():
    return generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1,
        random_seed=17,
    )


# --- Expression parser ----------------------------------------------------


def test_parse_symbol_literal(easy_system):
    sym = easy_system.symbols[0]
    parsed = _parse_expression(sym.emoji, easy_system)
    assert parsed == SymbolLiteral(symbol=sym)


def test_parse_nested_binary(easy_system):
    base_op = easy_system.base_operations[0]
    a, b, c = easy_system.symbols
    rendered = f"(({a.emoji} {base_op.symbol_id} {b.emoji}) {base_op.symbol_id} {c.emoji})"
    parsed = _parse_expression(rendered, easy_system)
    assert parsed == BinaryOp(
        op_name=base_op.name,
        left=BinaryOp(
            op_name=base_op.name,
            left=SymbolLiteral(a),
            right=SymbolLiteral(b),
        ),
        right=SymbolLiteral(c),
    )


def test_parse_unary_transform(medium_system):
    tr = medium_system.transformations[0]
    sym = medium_system.symbols[0]
    parsed = _parse_expression(f"{tr.name}({sym.emoji})", medium_system)
    assert parsed == UnaryTransform(transform_name=tr.name, operand=SymbolLiteral(sym))


def test_parse_rejects_garbage(easy_system):
    with pytest.raises(ParseError):
        _parse_expression("not a real expression", easy_system)


# --- Step-line parser ------------------------------------------------------


def test_parse_continuation_steps_basic(easy_system):
    a, b, c = easy_system.symbols
    base_op = easy_system.base_operations[0]
    text = (
        f"Step 3: ({a.emoji} {base_op.symbol_id} {b.emoji}) = {c.emoji}    "
        f"[by {base_op.symbol_id} table]\n"
    )
    steps = parse_continuation_steps(text, easy_system)
    assert steps is not None
    assert len(steps) == 1
    assert steps[0].step_number == 3
    assert steps[0].before == BinaryOp(
        op_name=base_op.name, left=SymbolLiteral(a), right=SymbolLiteral(b),
    )
    assert steps[0].after == SymbolLiteral(symbol=c)


def test_parse_continuation_steps_ignores_prose(easy_system):
    a, _, _ = easy_system.symbols
    text = (
        "I'm going to continue carefully now.\n"
        "\n"
        f"Step 1: {a.emoji} = {a.emoji}\n"
        "\n"
        "Final Output: " + a.emoji + "\n"
    )
    steps = parse_continuation_steps(text, easy_system)
    assert steps is not None
    assert len(steps) == 1


def test_parse_continuation_steps_returns_none_when_no_steps(easy_system):
    assert parse_continuation_steps("no step lines here", easy_system) is None


def test_parse_continuation_steps_returns_none_on_bad_expression(easy_system):
    text = "Step 1: garbage = also garbage    [by ??? table]\n"
    assert parse_continuation_steps(text, easy_system) is None


# --- Full validator --------------------------------------------------------


def _render_step(n, before_str, after_str, rule):
    return f"Step {n}: {before_str} = {after_str}    [by {rule}]"


def test_validate_clean_chain_reaches_gt(easy_system):
    base_op = easy_system.base_operations[0]
    a, b, c = easy_system.symbols
    # From the generated table, compute a genuine chain.
    left_val = base_op.table[(a, b)]
    result_val = base_op.table[(left_val, c)]
    text = "\n".join(
        [
            _render_step(
                1,
                f"(({a.emoji} {base_op.symbol_id} {b.emoji}) {base_op.symbol_id} {c.emoji})",
                f"({left_val.emoji} {base_op.symbol_id} {c.emoji})",
                f"{base_op.symbol_id} table",
            ),
            _render_step(
                2,
                f"({left_val.emoji} {base_op.symbol_id} {c.emoji})",
                result_val.emoji,
                f"{base_op.symbol_id} table",
            ),
        ]
    )
    result = validate_derivation(text, easy_system, result_val.emoji)
    assert result.parseable
    assert result.derivation_valid
    assert result.terminal_matches_gt
    assert result.parsed_step_count == 2


def test_validate_catches_wrong_per_step(easy_system):
    base_op = easy_system.base_operations[0]
    a, b, c = easy_system.symbols
    correct = base_op.table[(a, b)]
    wrong = next(s for s in easy_system.symbols if s != correct)
    text = _render_step(
        1,
        f"({a.emoji} {base_op.symbol_id} {b.emoji})",
        wrong.emoji,
        f"{base_op.symbol_id} table",
    )
    result = validate_derivation(text, easy_system, correct.emoji)
    assert result.parseable
    assert not result.derivation_valid
    assert result.first_invalid_step == 1


def test_validate_catches_compensating_errors(easy_system):
    """Two consecutive wrong steps whose final happens to match gt."""
    base_op = easy_system.base_operations[0]
    a, b, c = easy_system.symbols
    # Build the clean chain and its expected terminal.
    left_val = base_op.table[(a, b)]
    correct_terminal = base_op.table[(left_val, c)]
    # Now corrupt step 1 deliberately to a wrong intermediate ...
    wrong_intermediate = next(
        s for s in easy_system.symbols if s != left_val
    )
    # ... and then claim a step-2 reduction that happens to land on
    # correct_terminal despite starting from the wrong intermediate.
    text = "\n".join(
        [
            _render_step(
                1,
                f"(({a.emoji} {base_op.symbol_id} {b.emoji}) {base_op.symbol_id} {c.emoji})",
                f"({wrong_intermediate.emoji} {base_op.symbol_id} {c.emoji})",
                f"{base_op.symbol_id} table",
            ),
            _render_step(
                2,
                f"({wrong_intermediate.emoji} {base_op.symbol_id} {c.emoji})",
                correct_terminal.emoji,
                f"{base_op.symbol_id} table",
            ),
        ]
    )
    result = validate_derivation(text, easy_system, correct_terminal.emoji)
    # Compensating chain reaches the right terminal, but step 1 is invalid.
    assert result.parseable
    assert not result.derivation_valid
    assert result.first_invalid_step == 1
    # Terminal match is set False when derivation is invalid — the new metric
    # only trusts terminal match when the whole chain validates.
    assert not result.terminal_matches_gt


def test_validate_catches_continuity_break(easy_system):
    """Step 2's before doesn't match step 1's after."""
    base_op = easy_system.base_operations[0]
    a, b, c = easy_system.symbols
    left_val = base_op.table[(a, b)]
    # Pick a symbol guaranteed different from left_val so the discontinuity
    # is real regardless of what the generated table produces.
    step2_left = next(s for s in easy_system.symbols if s != left_val)
    after_step2 = base_op.table[(step2_left, c)]
    text = "\n".join(
        [
            _render_step(
                1,
                f"(({a.emoji} {base_op.symbol_id} {b.emoji}) {base_op.symbol_id} {c.emoji})",
                f"({left_val.emoji} {base_op.symbol_id} {c.emoji})",
                f"{base_op.symbol_id} table",
            ),
            _render_step(
                2,
                f"({step2_left.emoji} {base_op.symbol_id} {c.emoji})",
                after_step2.emoji,
                f"{base_op.symbol_id} table",
            ),
        ]
    )
    result = validate_derivation(text, easy_system, after_step2.emoji)
    assert result.parseable
    assert not result.derivation_valid
    assert result.first_discontinuity_step == 2


def test_validate_reports_unparseable(easy_system):
    result = validate_derivation(
        "I give up no steps here", easy_system, easy_system.symbols[0].emoji,
    )
    assert not result.parseable
    assert not result.derivation_valid
    assert not result.terminal_matches_gt
    assert result.parsed_step_count == 0


def test_validate_flags_non_symbol_terminal(easy_system):
    base_op = easy_system.base_operations[0]
    a, b, _ = easy_system.symbols
    # A "valid" single step that doesn't reduce all the way to a symbol.
    text = _render_step(
        1,
        f"({a.emoji} {base_op.symbol_id} {b.emoji})",
        f"({a.emoji} {base_op.symbol_id} {b.emoji})",
        f"{base_op.symbol_id} table",
    )
    result = validate_derivation(text, easy_system, a.emoji)
    assert result.parseable
    # evaluate(before) == evaluate(after) trivially, so per-step validity
    # passes, but the terminal isn't a SymbolLiteral so we flag False.
    assert result.derivation_valid
    assert not result.terminal_matches_gt


def test_validate_accepts_terminal_matching_single_symbol(easy_system):
    base_op = easy_system.base_operations[0]
    a, b, _ = easy_system.symbols
    correct = base_op.table[(a, b)]
    text = _render_step(
        1,
        f"({a.emoji} {base_op.symbol_id} {b.emoji})",
        correct.emoji,
        f"{base_op.symbol_id} table",
    )
    result = validate_derivation(text, easy_system, correct.emoji)
    assert result.parseable
    assert result.derivation_valid
    assert result.terminal_matches_gt
