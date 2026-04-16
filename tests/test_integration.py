import random

from emoji_bench.domain.expressions import expr_to_str_with_system, random_expression
from emoji_bench.domain.formatter import format_system_for_prompt, system_from_json, system_to_json
from emoji_bench.domain.generator import generate_system
from emoji_bench.domain.interpreter import evaluate


def test_generated_system_round_trip_preserves_expression_results():
    system = generate_system(
        n_symbols=5, n_base_ops=2, n_derived_ops=1, n_transformations=1, random_seed=314
    )
    restored = system_from_json(system_to_json(system))
    rng = random.Random(2718)

    assert format_system_for_prompt(restored) == format_system_for_prompt(system)

    for _ in range(25):
        expr = random_expression(system, depth=4, rng=rng)
        assert expr_to_str_with_system(expr, restored) == expr_to_str_with_system(expr, system)
        assert evaluate(expr, restored) == evaluate(expr, system)
