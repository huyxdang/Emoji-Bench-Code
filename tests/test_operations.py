import random

from emoji_bench.domain.operations import (
    generate_group_table,
    generate_operation_table,
    get_group_automorphisms,
)
from emoji_bench.domain.symbols import sample_symbols
from emoji_bench.domain.transforms import validate_distribution_property


def _make_symbols(n=3, seed=42):
    return sample_symbols(n, random.Random(seed))


def test_table_completeness():
    syms = _make_symbols(4)
    rng = random.Random(99)
    op = generate_operation_table(syms, rng, "test", "⊕")
    for a in syms:
        for b in syms:
            assert (a, b) in op.table


def test_table_closure():
    syms = _make_symbols(4)
    rng = random.Random(99)
    op = generate_operation_table(syms, rng, "test", "⊕")
    sym_set = set(syms)
    for result in op.table.values():
        assert result in sym_set


def test_table_deterministic():
    syms = _make_symbols(3)
    op1 = generate_operation_table(syms, random.Random(77), "t", "⊕")
    op2 = generate_operation_table(syms, random.Random(77), "t", "⊕")
    assert op1.table == op2.table


def test_commutative():
    syms = _make_symbols(4)
    rng = random.Random(55)
    op = generate_operation_table(syms, rng, "comm", "⊕", commutative=True)
    for a in syms:
        for b in syms:
            assert op.table[(a, b)] == op.table[(b, a)]


def test_generate_group_table_has_identity_and_latin_square_structure():
    syms = _make_symbols(5, seed=7)
    op = generate_group_table(syms, random.Random(123), "group", "⊕")

    identities = [
        s
        for s in syms
        if all(op.table[(s, x)] == x and op.table[(x, s)] == x for x in syms)
    ]
    assert len(identities) == 1

    for row in syms:
        assert set(op.table[(row, col)] for col in syms) == set(syms)

    for col in syms:
        assert set(op.table[(row, col)] for row in syms) == set(syms)


def test_get_group_automorphisms_returns_valid_non_identity_maps():
    syms = _make_symbols(5, seed=11)
    op = generate_group_table(syms, random.Random(88), "group", "⊕")
    automorphisms = get_group_automorphisms(syms, op)

    assert len(automorphisms) == 3
    assert len({frozenset(mapping.items()) for mapping in automorphisms}) == 3
    assert {s: s for s in syms} not in automorphisms

    for mapping in automorphisms:
        assert set(mapping.keys()) == set(syms)
        assert set(mapping.values()) == set(syms)
        assert validate_distribution_property(mapping, op)
