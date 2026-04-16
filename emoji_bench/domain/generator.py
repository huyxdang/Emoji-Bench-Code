from __future__ import annotations

import math
import random

from emoji_bench.operations import (
    generate_group_table,
    generate_operation_table,
    get_group_automorphisms,
)
from emoji_bench.symbols import sample_symbols
from emoji_bench.transforms import (
    find_valid_transformations,
    make_transformation_rule,
    validate_distribution_property,
)
from emoji_bench.types import (
    DerivedOperation,
    FormalSystem,
    OperationTable,
    TransformationRule,
)

# Operator symbols assigned in order to base ops then derived ops
OPERATOR_SYMBOLS: tuple[str, ...] = ("⊕", "⊗", "⊛", "⊞")

# Transform names assigned in order
TRANSFORM_NAMES: tuple[str, ...] = ("inv", "rot", "flip", "dual")

# Derived operation templates
# Each is (template_id, needs_transform)
DERIVED_TEMPLATES: tuple[tuple[str, bool], ...] = (
    ("compose_left", False),   # x ⊗ y = (x ⊕ y) ⊕ x
    ("inv_compose", True),     # x ⊗ y = inv(x ⊕ y)
    ("double_left", False),    # x ⊗ y = (x ⊕ x) ⊕ y
)

# System name components
_PREFIXES = (
    "Zelta", "Vorn", "Quix", "Drenn", "Flax", "Kova", "Myre", "Thal",
    "Orix", "Plith", "Bren", "Sylk", "Jace", "Nev", "Wren", "Cael",
)
_SUFFIXES = ("Algebra", "System", "Calculus", "Structure", "Field", "Ring")

MAX_RETRIES = 100


def _generate_system_name(rng: random.Random) -> str:
    return f"{rng.choice(_PREFIXES)} {rng.choice(_SUFFIXES)}"


def generate_system(
    n_symbols: int = 3,
    n_base_ops: int = 1,
    n_derived_ops: int = 0,
    n_transformations: int = 0,
    random_seed: int = 42,
) -> FormalSystem:
    """Generate a complete formal system with the given parameters.

    When transformations are needed, base operations are built from cyclic
    group tables (Z/nZ) with random label assignments. This guarantees
    automorphisms exist. When no transformations are needed, purely random
    tables are used for maximum novelty.
    """
    if not 3 <= n_symbols <= 6:
        raise ValueError(f"n_symbols must be 3-6, got {n_symbols}")
    if not 1 <= n_base_ops <= 2:
        raise ValueError(f"n_base_ops must be 1-2, got {n_base_ops}")
    if not 0 <= n_derived_ops <= 2:
        raise ValueError(f"n_derived_ops must be 0-2, got {n_derived_ops}")
    if not 0 <= n_transformations <= 2:
        raise ValueError(f"n_transformations must be 0-2, got {n_transformations}")
    if n_base_ops + n_derived_ops > len(OPERATOR_SYMBOLS):
        raise ValueError("Too many operations for available operator symbols")

    rng = random.Random(random_seed)

    # Step 1: Sample symbols (fixed regardless of retries)
    symbols = sample_symbols(n_symbols, rng)

    # Step 2: System name
    name = _generate_system_name(rng)

    # Step 3: Generate base operation tables
    if n_transformations == 0:
        # No transforms needed — use purely random tables for max novelty
        base_operations = _generate_random_base_ops(symbols, n_base_ops, rng)
        transformations: tuple[TransformationRule, ...] = ()
    else:
        # Need transforms — use group-based tables that guarantee automorphisms
        base_operations, transformations = _generate_base_ops_with_transforms(
            symbols, n_base_ops, n_transformations, rng
        )

    # Step 4: Generate derived operations
    derived_operations = _generate_derived_ops(
        n_derived_ops=n_derived_ops,
        base_operations=base_operations,
        transformations=transformations,
        n_base_ops=n_base_ops,
        rng=rng,
    )

    system = FormalSystem(
        name=name,
        seed=random_seed,
        symbols=symbols,
        base_operations=base_operations,
        derived_operations=derived_operations,
        transformations=transformations,
    )

    # Final sanity check
    _validate_system(system)
    return system


def _generate_random_base_ops(
    symbols: tuple, n_base_ops: int, rng: random.Random
) -> tuple[OperationTable, ...]:
    """Generate purely random operation tables (no algebraic guarantees)."""
    ops: list[OperationTable] = []
    for i in range(n_base_ops):
        ops.append(generate_operation_table(
            symbols=symbols,
            rng=rng,
            name=f"op{i}",
            symbol_id=OPERATOR_SYMBOLS[i],
        ))
    return tuple(ops)


def _generate_base_ops_with_transforms(
    symbols: tuple,
    n_base_ops: int,
    n_transformations: int,
    rng: random.Random,
) -> tuple[tuple[OperationTable, ...], tuple[TransformationRule, ...]]:
    """Generate operation tables guaranteed to have the required automorphisms.

    Strategy: pick a single random permutation σ of order >= n_transformations+1.
    The powers σ, σ², ..., σ^(k-1) form a cyclic automorphism group.
    Then build operation tables compatible with σ (which automatically makes
    them compatible with all powers of σ).

    For each table, group (a,b) pairs into orbits under σ, pick a random
    result for one representative per orbit, and propagate via
    σ(a⊕b) = σ(a) ⊕ σ(b).
    """
    n = len(symbols)
    sym_list = list(symbols)

    for _attempt in range(MAX_RETRIES):
        sub_rng = random.Random(rng.randint(0, 2**63))

        # Step 1: Find a random permutation with order >= n_transformations + 1
        sigma = _random_perm_with_min_order(sym_list, n_transformations + 1, sub_rng)
        if sigma is None:
            continue

        # Compute σ's order and all non-identity powers
        order = _perm_order(sigma, sym_list)
        powers: list[dict] = []
        current = dict(sigma)
        for _ in range(order - 1):
            powers.append(dict(current))
            # Compose: current = sigma ∘ current
            current = {s: sigma[current[s]] for s in sym_list}

        # Pick n_transformations powers as our transforms
        selected_perms = powers[:n_transformations]

        # Step 2: Build operation tables compatible with σ.
        ops: list[OperationTable] = []
        for op_idx in range(n_base_ops):
            table = _build_compatible_table(symbols, sigma, sub_rng)
            ops.append(OperationTable(
                name=f"op{op_idx}",
                symbol_id=OPERATOR_SYMBOLS[op_idx],
                symbols=symbols,
                table=table,
            ))

        base_operations = tuple(ops)

        # Verify all selected transforms distribute over all ops
        all_valid = all(
            validate_distribution_property(perm, op)
            for perm in selected_perms
            for op in base_operations
        )

        if all_valid:
            op_names = tuple(op.name for op in base_operations)
            trans_list = [
                make_transformation_rule(
                    name=TRANSFORM_NAMES[i],
                    mapping=selected_perms[i],
                    distributes_over=op_names,
                )
                for i in range(n_transformations)
            ]
            return base_operations, tuple(trans_list)

    raise RuntimeError(
        f"Failed to generate a valid system after {MAX_RETRIES} retries. "
        f"Could not find {n_transformations} valid transformation(s) for "
        f"{len(symbols)} symbols and {n_base_ops} base operation(s)."
    )


def _random_perm_with_min_order(
    sym_list: list, min_order: int, rng: random.Random
) -> dict | None:
    """Generate a random permutation of sym_list with order >= min_order."""
    for _ in range(100):
        perm = sym_list[:]
        rng.shuffle(perm)
        mapping = dict(zip(sym_list, perm))
        if _perm_order(mapping, sym_list) >= min_order:
            return mapping
    return None


def _perm_order(perm: dict, sym_list: list) -> int:
    """Compute the order of a permutation (smallest k where σ^k = id)."""
    # Order = lcm of cycle lengths
    visited: set = set()
    cycle_lengths: list[int] = []
    for s in sym_list:
        if s in visited:
            continue
        cycle_len = 0
        current = s
        while current not in visited:
            visited.add(current)
            current = perm[current]
            cycle_len += 1
        cycle_lengths.append(cycle_len)

    result = 1
    for cl in cycle_lengths:
        result = result * cl // math.gcd(result, cl)
    return result


def _build_compatible_table(
    symbols: tuple,
    sigma: dict,
    rng: random.Random,
) -> dict:
    """Build a random operation table compatible with a single permutation σ.

    Groups (a,b) pairs into orbits under σ (applied simultaneously to both
    components). For each orbit, picks a random result for the representative
    and propagates: if a⊕b=r then σ(a)⊕σ(b)=σ(r).

    Since σ generates a cyclic group, this is conflict-free by construction:
    each orbit is a simple cycle, so propagation never revisits a pair with
    a different required value.
    """
    table: dict = {}
    sym_list = list(symbols)
    assigned: set = set()

    for a in sym_list:
        for b in sym_list:
            if (a, b) in assigned:
                continue

            # Pick a random result for (a, b)
            result = rng.choice(sym_list)

            # Propagate through the orbit of (a, b) under σ
            x, y, r = a, b, result
            while (x, y) not in assigned:
                table[(x, y)] = r
                assigned.add((x, y))
                x, y, r = sigma[x], sigma[y], sigma[r]

    return table


def _generate_derived_ops(
    n_derived_ops: int,
    base_operations: tuple[OperationTable, ...],
    transformations: tuple[TransformationRule, ...],
    n_base_ops: int,
    rng: random.Random,
) -> tuple[DerivedOperation, ...]:
    """Select derived operation templates and build DerivedOperation objects."""
    if n_derived_ops == 0:
        return ()

    # Filter templates: only offer inv_compose if transforms exist
    has_transforms = len(transformations) > 0
    available = [
        (tid, needs_t) for tid, needs_t in DERIVED_TEMPLATES
        if not needs_t or has_transforms
    ]

    if len(available) < n_derived_ops:
        raise ValueError(
            f"Not enough derived operation templates available "
            f"({len(available)}) for {n_derived_ops} derived ops"
        )

    # Sample without replacement
    rng.shuffle(available)
    selected = available[:n_derived_ops]

    result: list[DerivedOperation] = []
    for i, (template_id, needs_transform) in enumerate(selected):
        base_op = rng.choice(base_operations)
        transform_name = transformations[0].name if needs_transform else None
        op_idx = n_base_ops + i

        result.append(DerivedOperation(
            name=f"dop{i}",
            symbol_id=OPERATOR_SYMBOLS[op_idx],
            template_id=template_id,
            base_ops=(base_op.name,),
            transform_name=transform_name,
        ))

    return tuple(result)


def _validate_system(system: FormalSystem) -> None:
    """Run all consistency checks on a generated system. Raises on failure."""
    from emoji_bench.interpreter import evaluate_binary
    from emoji_bench.transforms import validate_distribution_property

    symbol_set = set(system.symbols)

    # Check base operation tables
    for op in system.base_operations:
        for a in system.symbols:
            for b in system.symbols:
                result = op.table.get((a, b))
                if result is None:
                    raise AssertionError(f"Missing entry in {op.name}: ({a}, {b})")
                if result not in symbol_set:
                    raise AssertionError(
                        f"Result {result} not in symbol set for {op.name}({a}, {b})"
                    )

    # Check transformation distribution properties
    for transform in system.transformations:
        for op_name in transform.distributes_over:
            op = next(o for o in system.base_operations if o.name == op_name)
            if not validate_distribution_property(transform.mapping, op):
                raise AssertionError(
                    f"Transformation '{transform.name}' does not distribute "
                    f"over operation '{op_name}'"
                )

    # Check derived operations produce valid results
    for dop in system.derived_operations:
        for a in system.symbols:
            for b in system.symbols:
                result = evaluate_binary(dop.name, a, b, system)
                if result not in symbol_set:
                    raise AssertionError(
                        f"Derived op {dop.name}({a}, {b}) = {result} not in symbol set"
                    )
