import json
import math
import subprocess
import sys
from pathlib import Path

from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.dataset.continuation_dataset import (
    DEFAULT_CONTINUATION_TARGET_LENGTHS,
    MIN_REALIZED_X,
    REJECTION_REASONS,
    _try_generate,
    generate_continuation_dataset_records,
)
from emoji_bench.dataset.dataset_io import DIFFICULTY_CONFIGS
from emoji_bench.dataset.rejection_reasons import (
    ContinuationGenerationError,
    R_NO_ELIGIBLE_IN_WINDOW,
)
from emoji_bench.continuation_formatter import format_clean_derivation
from emoji_bench.domain.formatter import system_from_json
from emoji_bench.domain.generator import generate_system


def test_generate_continuation_dataset_records_produces_exact_count():
    split_records, manifest = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-test",
        count=4,
        master_seed=20260413,
    )

    assert split_records["train"] == []
    assert split_records["validation"] == []
    assert len(split_records["test"]) == 4
    assert manifest.total_examples == 4
    assert manifest.split_counts == {"train": 0, "validation": 0, "test": 4}
    assert manifest.error_type_counts == {"E-CONTINUE": 4}


def test_continuation_records_have_full_schema_and_invariants():
    split_records, _ = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-schema",
        count=4,
        master_seed=20260413,
    )

    for record in split_records["test"]:
        # Conversation fields.
        assert record["turn_1_user"].count("=== RULES ===") == 1
        assert record["turn_1_user"].count("=== EXPRESSION ===") == 1
        assert record["turn_1_user"].count("=== TASK ===") == 1
        assert "Step 1:" not in record["turn_1_user"]
        prefill = record["turn_1_assistant_prefill"]
        assert prefill.startswith("Start: ")
        assert not prefill.endswith("\n")
        assert "Final Output:" not in prefill
        assert "Result:" not in prefill
        clean_derivation = record["clean_derivation"]
        assert clean_derivation.startswith("Start: ")
        assert clean_derivation.count("Final Output:") == 1
        assert clean_derivation.endswith(
            f"Final Output: {record['ground_truth_final_output']}"
        )

        # Scoring invariants.
        assert record["ground_truth_final_output"] != record["wrong_branch_final_output"]
        assert record["error_type"] == "E-CONTINUE"

        # Structural invariants.
        x = record["chain_length_x"]
        y = record["prefill_error_step"]
        assert x >= MIN_REALIZED_X
        assert abs(y - x // 2) <= 1
        assert 1 <= y <= x - 1

        # Prefill ends with a Step-N line whose number matches the cutoff.
        last_line = prefill.splitlines()[-1]
        assert last_line.startswith(f"Step {y}:")


def test_runway_floor_is_satisfied_for_every_record():
    split_records, _ = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-runway",
        count=4,
        master_seed=20260413,
    )

    for record in split_records["test"]:
        system = system_from_json(record["system_json"])
        instance = generate_continuation_instance(
            system,
            length=record["target_step_count"],
            chain_seed=record["chain_seed"],
            error_seed=record["error_seed"],
        )
        required = math.ceil(instance.chain_length_x / 2)
        mutated_remaining = len(instance.mutated_chain.steps) - instance.prefill_error_step
        assert mutated_remaining >= required


def test_continuation_records_are_exactly_reproducible_from_stored_metadata():
    split_records, manifest = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-repro",
        count=4,
        master_seed=20260413,
    )

    assert manifest.master_seed == 20260413
    assert manifest.difficulty_configs == DIFFICULTY_CONFIGS

    for record in split_records["test"]:
        system_from_row = system_from_json(record["system_json"])
        system_from_seed = generate_system(
            random_seed=record["system_seed"],
            **DIFFICULTY_CONFIGS[record["difficulty"]],
        )

        assert system_from_seed == system_from_row

        instance = generate_continuation_instance(
            system_from_row,
            length=record["target_step_count"],
            chain_seed=record["chain_seed"],
            error_seed=record["error_seed"],
        )
        assert instance.turn_1_user == record["turn_1_user"]
        assert instance.turn_1_assistant_prefill == record["turn_1_assistant_prefill"]
        assert record["clean_derivation"] == format_clean_derivation(
            instance.clean_chain, system_from_row
        )
        assert instance.ground_truth_final_output.emoji == record["ground_truth_final_output"]
        assert instance.wrong_branch_final_output.emoji == record["wrong_branch_final_output"]
        assert instance.chain_length_x == record["chain_length_x"]
        assert instance.prefill_error_step == record["prefill_error_step"]


def test_rejection_counts_are_populated_for_every_difficulty():
    _, manifest = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-rej",
        count=4,
        master_seed=20260413,
    )

    assert manifest.rejection_counts is not None
    assert set(manifest.rejection_counts.keys()) == {"easy", "medium", "hard", "expert"}
    for difficulty, bucket in manifest.rejection_counts.items():
        assert set(bucket.keys()) == set(REJECTION_REASONS)
        for reason, count in bucket.items():
            assert count >= 0, f"negative rejection count for {difficulty}/{reason}"


def test_try_generate_uses_structured_rejection_reasons(monkeypatch):
    def fake_generate_continuation_instance(*args, **kwargs):
        raise ContinuationGenerationError(
            R_NO_ELIGIBLE_IN_WINDOW,
            "custom message that should not need string parsing",
        )

    monkeypatch.setattr(
        "emoji_bench.dataset.continuation_dataset.generate_continuation_instance",
        fake_generate_continuation_instance,
    )

    instance, reason = _try_generate(
        system=object(),
        target_step_count=6,
        chain_seed=7,
        error_seed=13,
    )

    assert instance is None
    assert reason == R_NO_ELIGIBLE_IN_WINDOW


def test_default_target_lengths_match_locked_values():
    assert DEFAULT_CONTINUATION_TARGET_LENGTHS == {
        "easy": 6,
        "medium": 8,
        "hard": 10,
        "expert": 14,
    }


def test_generate_continuation_dataset_script_supports_exact_count(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_continuation_dataset.py",
            "--dataset-name",
            "emoji-bench-e-continue-cli",
            "--output-dir",
            str(output_dir),
            "--count",
            "4",
            "--master-seed",
            "20260413",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["total_examples"] == 4
    assert summary["master_seed"] == 20260413
    assert summary["target_lengths"] == DEFAULT_CONTINUATION_TARGET_LENGTHS
    assert manifest["total_examples"] == 4
    assert manifest["master_seed"] == 20260413
    assert manifest["difficulty_configs"] == DIFFICULTY_CONFIGS
    assert summary["selected_variants"] == ["E-CONTINUE"]
    assert manifest["error_type_counts"] == {"E-CONTINUE": 4}
    assert manifest["rejection_counts"] is not None
    card_text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "turn_1_user" in card_text
    assert "turn_1_assistant_prefill" in card_text
    assert "clean_derivation" in card_text
    assert "ground_truth_final_output" in card_text
    assert "master_seed" in card_text

    test_jsonl = (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(test_jsonl) == 4
    for line in test_jsonl:
        row = json.loads(line)
        assert row["error_type"] == "E-CONTINUE"
        assert row["clean_derivation"].endswith(
            f"Final Output: {row['ground_truth_final_output']}"
        )
        assert row["ground_truth_final_output"] != row["wrong_branch_final_output"]
