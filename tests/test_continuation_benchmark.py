from emoji_bench.dataset.continuation_benchmark import (
    ContinuationInstance,
    continuation_record,
    generate_continuation_instance,
)
from emoji_bench.domain.generator import generate_system


def _easy_system(seed: int = 11):
    return generate_system(
        n_symbols=3, n_base_ops=1, n_derived_ops=0, n_transformations=0,
        random_seed=seed,
    )


def test_generate_continuation_instance_produces_non_convergent_error():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    assert isinstance(instance, ContinuationInstance)
    assert instance.ground_truth_final_output != instance.wrong_branch_final_output
    assert instance.ground_truth_final_output == instance.clean_chain.final_result
    assert instance.wrong_branch_final_output == instance.mutated_chain.final_result


def test_prefill_cutoff_equals_error_step_and_lies_near_midpoint():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    x = instance.chain_length_x
    assert abs(instance.prefill_error_step - x // 2) <= 1
    assert 1 <= instance.prefill_error_step <= x - 1


def test_prefill_string_does_not_end_on_terminal_marker():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    prefill = instance.turn_1_assistant_prefill
    assert not prefill.endswith("\n")
    assert "Final Output:" not in prefill
    assert "Result:" not in prefill
    # Last non-empty line must be a numbered step line.
    last_line = prefill.splitlines()[-1]
    assert last_line.startswith(f"Step {instance.prefill_error_step}:")


def test_prefill_step_count_matches_cutoff():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    step_lines = [
        line for line in instance.turn_1_assistant_prefill.splitlines()
        if line.startswith("Step ")
    ]
    assert len(step_lines) == instance.prefill_error_step


def test_turn_1_user_contains_rules_and_expression_but_no_steps():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    turn_1 = instance.turn_1_user
    assert "=== RULES ===" in turn_1
    assert "=== EXPRESSION ===" in turn_1
    assert "=== TASK ===" in turn_1
    assert "Step 1:" not in turn_1
    assert "Final Output:" in turn_1


def test_continuation_record_round_trip():
    system = _easy_system()
    instance = generate_continuation_instance(
        system, length=4, chain_seed=7, error_seed=13,
    )

    record = continuation_record(
        instance=instance,
        example_id="cont-000001",
        base_id="easy-0000",
        split="test",
        difficulty="easy",
        system_json="{}",
        system_seed=1,
        chain_seed=7,
        error_seed=13,
        target_step_count=4,
    )

    assert record["error_type"] == "E-CONTINUE"
    assert record["ground_truth_final_output"] != record["wrong_branch_final_output"]
    assert record["chain_length_x"] >= 2
    for key in (
        "turn_1_user", "turn_1_assistant_prefill",
        "system_seed", "chain_seed", "error_seed",
    ):
        assert key in record
