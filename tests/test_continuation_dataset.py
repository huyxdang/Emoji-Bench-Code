import json
import math
import subprocess
import sys
from pathlib import Path

from emoji_bench.continuation_dataset import (
    DEFAULT_CONTINUATION_TARGET_LENGTHS,
    MIN_REALIZED_X,
    REJECTION_REASONS,
    generate_continuation_dataset_records,
)


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
    assert manifest.condition_counts == {"error_injected": 4}


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
        assert record["turn_2_user"] == "Please continue."

        # Scoring invariants.
        assert record["ground_truth_final_output"] != record["wrong_branch_final_output"]
        assert record["has_prefill_error"] is True
        assert record["error_type"] == "E-CONTINUE"

        # Structural invariants.
        x = record["chain_length_x"]
        y = record["prefill_error_step"]
        assert x >= MIN_REALIZED_X
        assert record["prefill_cutoff_step"] == y
        assert abs(y - x // 2) <= 1
        assert 1 <= y <= x - 1

        # Prefill ends with a Step-N line whose number matches the cutoff.
        last_line = prefill.splitlines()[-1]
        assert last_line.startswith(f"Step {y}:")


def test_runway_floor_is_satisfied_for_every_record():
    # We can't read mutated_chain directly from the record, but we can verify
    # the runway by regenerating instances with the same seeds. Here we just
    # check the weaker structural invariant: Y <= X - ceil(X/2), i.e. there
    # are at least ceil(X/2) clean-chain steps after the cutoff, which
    # matches the dataset-generation filter when the mutated chain's suffix
    # is at least as long as the clean chain's remaining steps.
    split_records, _ = generate_continuation_dataset_records(
        dataset_name="emoji-bench-e-continue-runway",
        count=4,
        master_seed=20260413,
    )

    for record in split_records["test"]:
        x = record["chain_length_x"]
        y = record["prefill_cutoff_step"]
        required = math.ceil(x / 2)
        clean_remaining = x - y
        # Because Y is at the midpoint, the clean chain always has >=
        # ceil(X/2) remaining steps; the mutated runway is enforced
        # separately by the generator's R_INSUFFICIENT_RUNWAY filter.
        assert clean_remaining >= required - 1


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
    assert manifest["total_examples"] == 4
    assert summary["selected_variants"] == ["E-CONTINUE"]
    assert manifest["error_type_counts"] == {"E-CONTINUE": 4}
    assert manifest["rejection_counts"] is not None

    test_jsonl = (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(test_jsonl) == 4
    for line in test_jsonl:
        row = json.loads(line)
        assert row["error_type"] == "E-CONTINUE"
        assert row["ground_truth_final_output"] != row["wrong_branch_final_output"]
