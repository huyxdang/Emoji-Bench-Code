from __future__ import annotations

import itertools

from emoji_bench.types import OperationTable, Symbol, TransformationRule


def validate_distribution_property(
    mapping: dict[Symbol, Symbol],
    op: OperationTable,
) -> bool:
    """Check that mapping distributes over op: t(op(a,b)) == op(t(a),t(b)) for all a,b."""
    for a in op.symbols:
        for b in op.symbols:
            lhs = mapping[op.table[(a, b)]]
            rhs = op.table[(mapping[a], mapping[b])]
            if lhs != rhs:
                return False
    return True


def find_valid_transformations(
    symbols: tuple[Symbol, ...],
    operations: tuple[OperationTable, ...],
    exclude_identity: bool = True,
) -> list[dict[Symbol, Symbol]]:
    """Find all permutations of symbols that satisfy the distribution property
    over ALL given operations.

    For n<=6 (max 720 permutations), this is exhaustively fast.
    """
    valid: list[dict[Symbol, Symbol]] = []
    symbol_list = list(symbols)

    for perm in itertools.permutations(symbol_list):
        mapping = dict(zip(symbol_list, perm))

        # Skip identity if requested
        if exclude_identity and all(mapping[s] == s for s in symbol_list):
            continue

        # Check distribution property over all operations
        if all(validate_distribution_property(mapping, op) for op in operations):
            valid.append(mapping)

    return valid


def make_transformation_rule(
    name: str,
    mapping: dict[Symbol, Symbol],
    distributes_over: tuple[str, ...],
) -> TransformationRule:
    """Construct a TransformationRule from a validated mapping."""
    return TransformationRule(
        name=name,
        mapping=mapping,
        distributes_over=distributes_over,
    )
