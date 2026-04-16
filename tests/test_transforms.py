from emoji_bench.domain.transforms import find_valid_transformations, validate_distribution_property
from emoji_bench.domain.types import OperationTable, Symbol


def _zelta_op():
    """Build the Zelta ⊕ table."""
    a, b, c = Symbol("🦩"), Symbol("🧲"), Symbol("🪣")
    symbols = (a, b, c)
    table = {
        (a, a): b, (a, b): c, (a, c): a,
        (b, a): c, (b, b): a, (b, c): b,
        (c, a): a, (c, b): b, (c, c): c,
    }
    return OperationTable(name="op0", symbol_id="⊕", symbols=symbols, table=table)


def test_zelta_automorphism_is_valid():
    """The automorphism 🦩↔🧲, 🪣→🪣 (x→2x in Z/3Z) should satisfy distribution."""
    op = _zelta_op()
    a, b, c = op.symbols
    # Zelta is Z/3Z with 🪣=0, 🦩=1, 🧲=2. Automorphism x→2x: 0→0, 1→2, 2→1
    mapping = {a: b, b: a, c: c}
    assert validate_distribution_property(mapping, op) is True


def test_identity_always_valid():
    """The identity permutation always satisfies the distribution property."""
    op = _zelta_op()
    mapping = {s: s for s in op.symbols}
    assert validate_distribution_property(mapping, op) is True


def test_find_valid_includes_automorphism():
    """find_valid_transformations should include the x→2x automorphism for Zelta."""
    op = _zelta_op()
    a, b, c = op.symbols
    valid = find_valid_transformations(op.symbols, (op,), exclude_identity=True)
    auto = {a: b, b: a, c: c}
    assert auto in valid


def test_find_valid_excludes_identity():
    op = _zelta_op()
    valid = find_valid_transformations(op.symbols, (op,), exclude_identity=True)
    identity = {s: s for s in op.symbols}
    assert identity not in valid


def test_find_valid_includes_identity_when_requested():
    op = _zelta_op()
    valid = find_valid_transformations(op.symbols, (op,), exclude_identity=False)
    identity = {s: s for s in op.symbols}
    assert identity in valid


def test_invalid_perm_rejected():
    """The cyclic translation 🦩→🧲→🪣→🦩 is NOT an automorphism of Zelta."""
    op = _zelta_op()
    a, b, c = op.symbols
    cyclic_translation = {a: b, b: c, c: a}
    assert validate_distribution_property(cyclic_translation, op) is False
