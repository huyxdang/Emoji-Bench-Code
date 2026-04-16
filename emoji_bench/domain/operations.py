from __future__ import annotations

import math
import random

from emoji_bench.types import OperationTable, Symbol


def generate_operation_table(
    symbols: tuple[Symbol, ...],
    rng: random.Random,
    name: str,
    symbol_id: str,
    commutative: bool = False,
) -> OperationTable:
    """Generate a random n x n operation table over the given symbols.

    This produces a purely random magma (no algebraic guarantees).
    For tables that need automorphisms, use generate_group_table instead.
    """
    table: dict[tuple[Symbol, Symbol], Symbol] = {}

    if commutative:
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if j >= i:
                    result = rng.choice(symbols)
                    table[(a, b)] = result
                    table[(b, a)] = result
    else:
        for a in symbols:
            for b in symbols:
                table[(a, b)] = rng.choice(symbols)

    return OperationTable(
        name=name,
        symbol_id=symbol_id,
        symbols=symbols,
        table=table,
    )


def generate_group_table(
    symbols: tuple[Symbol, ...],
    rng: random.Random,
    name: str,
    symbol_id: str,
) -> OperationTable:
    """Generate an operation table based on a group structure.

    Uses cyclic group Z/nZ with a random symbol-to-label assignment.
    This guarantees the existence of automorphisms (x -> k*x mod n for
    k coprime to n), making transformation generation reliable.

    The random permutation of labels makes the table look arbitrary
    to the model — it can't tell it's a group without checking.
    """
    n = len(symbols)

    # Random assignment of symbols to group elements 0..n-1
    labels = list(range(n))
    rng.shuffle(labels)
    sym_to_label = {symbols[i]: labels[i] for i in range(n)}
    label_to_sym = {labels[i]: symbols[i] for i in range(n)}

    # Build Cayley table for Z/nZ under addition
    table: dict[tuple[Symbol, Symbol], Symbol] = {}
    for a in symbols:
        for b in symbols:
            result_label = (sym_to_label[a] + sym_to_label[b]) % n
            table[(a, b)] = label_to_sym[result_label]

    return OperationTable(
        name=name,
        symbol_id=symbol_id,
        symbols=symbols,
        table=table,
    )


def get_group_automorphisms(
    symbols: tuple[Symbol, ...],
    op: OperationTable,
) -> list[dict[Symbol, Symbol]]:
    """Find automorphisms of a cyclic group table.

    For Z/nZ, the automorphisms are x -> k*x (mod n) for k coprime to n.
    We recover the label assignment from the table and compute all such maps.
    Returns non-identity automorphisms as symbol-to-symbol dicts.
    """
    n = len(symbols)

    # Recover the identity element: e is the symbol where e⊕e⊕...=e (or simpler,
    # e⊕x = x for all x). For Z/nZ, the identity is the symbol mapped to label 0.
    identity_sym = None
    for candidate in symbols:
        if all(op.table[(candidate, x)] == x for x in symbols):
            identity_sym = candidate
            break
    assert identity_sym is not None, "No identity element found"

    # Recover label assignment: find a generator (symbol g where g, g+g, g+g+g, ... cycles through all)
    # In Z/nZ, any element coprime to n is a generator. We just need to find one.
    generator_sym = None
    sym_to_label: dict[Symbol, int] = {}
    for candidate in symbols:
        # Try this symbol as generator: compute its powers under ⊕
        seen: dict[Symbol, int] = {identity_sym: 0}
        current = candidate
        for power in range(1, n + 1):
            if current in seen:
                break
            seen[current] = power
            current = op.table[(current, candidate)]
        if len(seen) == n:
            generator_sym = candidate
            sym_to_label = seen
            break

    assert generator_sym is not None, "No generator found — table may not be cyclic"
    label_to_sym = {v: k for k, v in sym_to_label.items()}

    # Automorphisms: x -> k*x mod n for k coprime to n, excluding k=1 (identity)
    automorphisms: list[dict[Symbol, Symbol]] = []
    for k in range(2, n):
        if math.gcd(k, n) == 1:
            mapping = {}
            for sym, label in sym_to_label.items():
                new_label = (k * label) % n
                mapping[sym] = label_to_sym[new_label]
            automorphisms.append(mapping)

    return automorphisms
