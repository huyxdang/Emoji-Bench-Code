import random

import pytest

from emoji_bench.benchmark_types import ErrorType
from emoji_bench.chain_generator import (
    find_leftmost_innermost,
    generate_chain,
    replace_at_path,
)
from emoji_bench.error_injector import (
    get_cascading_eligible_steps,
    inject_cascading_wrong_result,
)
from emoji_bench.expressions import SymbolLiteral
from emoji_bench.generator import generate_system
from emoji_bench.interpreter import evaluate


def _system():
    return generate_system(
        n_symbols=4,
        n_base_ops=1,
        n_derived_ops=1,
        n_transformations=1,
        random_seed=77,
    )


def test_cascading_eligible_steps_are_non_terminal_result_steps():
    chain = generate_chain(_system(), length=6, seed=12)

    eligible = get_cascading_eligible_steps(chain)

    assert eligible
    assert all(step.result_symbol is not None for step in eligible)
    assert all(step.step_number < chain.steps[-1].step_number for step in eligible)


def test_inject_cascading_wrong_result_changes_suffix_but_not_prefix():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)
    step_number = get_cascading_eligible_steps(chain)[0].step_number

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=step_number,
        seed=99,
    )

    assert injected_chain.steps[: step_number - 1] == chain.steps[: step_number - 1]
    assert injected_chain.steps[step_number - 1].before == chain.steps[step_number - 1].before
    assert injected_chain.steps[step_number - 1].result_symbol != chain.steps[step_number - 1].result_symbol
    assert injected_chain.steps[step_number - 1 :] != chain.steps[step_number - 1 :]
    assert error_info.original_chain == chain


def test_inject_cascading_wrong_result_recomputes_locally_valid_suffix():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)
    step_number = get_cascading_eligible_steps(chain)[0].step_number

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=step_number,
        seed=99,
    )

    root_step = injected_chain.steps[step_number - 1]
    assert error_info.error_type is ErrorType.E_CASC
    assert evaluate(root_step.reduced_subexpr, system) == error_info.correct_result
    assert root_step.result_symbol == error_info.injected_result
    assert root_step.result_symbol != evaluate(root_step.reduced_subexpr, system)

    for step in injected_chain.steps[step_number:]:
        if step.result_symbol is not None:
            assert evaluate(step.reduced_subexpr, system) == step.result_symbol


def test_inject_cascading_wrong_result_preserves_outer_expression_shape():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)
    target = next(
        step for step in get_cascading_eligible_steps(chain)
        if not isinstance(step.after, SymbolLiteral)
    )

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=target.step_number,
        seed=99,
    )

    injected_step = injected_chain.steps[target.step_number - 1]
    path = find_leftmost_innermost(target.before)

    assert path is not None
    assert not isinstance(injected_step.after, SymbolLiteral)
    assert injected_chain.steps[target.step_number].before == injected_step.after
    assert injected_step.after == replace_at_path(
        target.before,
        path,
        SymbolLiteral(error_info.injected_result),
    )


def test_inject_cascading_wrong_result_updates_final_result():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, _ = inject_cascading_wrong_result(chain, system, seed=99)

    assert injected_chain.final_result != chain.final_result
    assert injected_chain.final_result == injected_chain.steps[-1].after.symbol


def test_inject_cascading_wrong_result_rejects_seed_and_rng_together():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="either seed or rng"):
        inject_cascading_wrong_result(chain, system, seed=99, rng=random.Random(99))


def test_inject_cascading_wrong_result_rejects_ineligible_step_number():
    system = _system()
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="not eligible"):
        inject_cascading_wrong_result(
            chain,
            system,
            step_number=chain.steps[-1].step_number,
            seed=99,
        )
