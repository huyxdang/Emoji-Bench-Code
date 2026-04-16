import pytest

from emoji_bench.domain.generator import generate_system
from emoji_bench.domain.interpreter import evaluate_binary
from emoji_bench.domain.transforms import validate_distribution_property


def test_generate_easy():
    """Easy: 3 symbols, 1 base op, 0 derived, 0 transforms."""
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    assert len(system.symbols) == 3
    assert len(system.base_operations) == 1
    assert len(system.derived_operations) == 0
    assert len(system.transformations) == 0


def test_generate_medium():
    """Medium: 4 symbols, 1 base op, 1 derived, 1 transform."""
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    assert len(system.symbols) == 4
    assert len(system.base_operations) == 1
    assert len(system.derived_operations) == 1
    assert len(system.transformations) == 1


def test_generate_hard():
    """Hard: 5 symbols, 2 base ops, 1 derived, 1 transform."""
    system = generate_system(
        n_symbols=5, n_base_ops=2, n_derived_ops=1, n_transformations=1, random_seed=123
    )
    assert len(system.symbols) == 5
    assert len(system.base_operations) == 2
    assert len(system.derived_operations) == 1
    assert len(system.transformations) == 1


def test_generate_expert():
    """Expert: 6 symbols, 2 base ops, 2 derived, 2 transforms."""
    system = generate_system(
        n_symbols=6, n_base_ops=2, n_derived_ops=2, n_transformations=2, random_seed=200
    )
    assert len(system.symbols) == 6
    assert len(system.base_operations) == 2
    assert len(system.derived_operations) == 2
    assert len(system.transformations) == 2


def test_reproducibility():
    s1 = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=50)
    s2 = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=50)
    assert s1.symbols == s2.symbols
    assert s1.base_operations == s2.base_operations
    assert s1.transformations == s2.transformations
    assert s1.name == s2.name


def test_different_seeds():
    s1 = generate_system(n_symbols=3, random_seed=1)
    s2 = generate_system(n_symbols=3, random_seed=2)
    # Very unlikely to be identical
    assert s1.symbols != s2.symbols or s1.base_operations != s2.base_operations


def test_consistency_stress():
    """Generate 50 systems with various configs and verify all pass validation."""
    configs = [
        (3, 1, 0, 0),
        (4, 1, 1, 1),
        (5, 2, 1, 1),
        (6, 2, 2, 2),
    ]
    for n_sym, n_bop, n_dop, n_tr in configs:
        for seed in range(50):
            system = generate_system(
                n_symbols=n_sym,
                n_base_ops=n_bop,
                n_derived_ops=n_dop,
                n_transformations=n_tr,
                random_seed=seed * 1000 + n_sym,
            )
            # Verify all transforms distribute
            for tr in system.transformations:
                for op_name in tr.distributes_over:
                    op = next(o for o in system.base_operations if o.name == op_name)
                    assert validate_distribution_property(tr.mapping, op)

            # Verify all derived ops produce valid symbols
            sym_set = set(system.symbols)
            for dop in system.derived_operations:
                for a in system.symbols:
                    for b in system.symbols:
                        result = evaluate_binary(dop.name, a, b, system)
                        assert result in sym_set


def test_invalid_params():
    with pytest.raises(ValueError):
        generate_system(n_symbols=2)
    with pytest.raises(ValueError):
        generate_system(n_symbols=7)
    with pytest.raises(ValueError):
        generate_system(n_base_ops=0)
    with pytest.raises(ValueError):
        generate_system(n_base_ops=3)
