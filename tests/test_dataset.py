import json
import subprocess
import sys
from pathlib import Path

from emoji_bench.benchmark_types import Condition, ErrorType
from emoji_bench.dataset import (
    DEFAULT_VARIANTS,
    DatasetVariant,
    generate_dataset_records,
    write_dataset,
)


def test_generate_dataset_records_produces_balanced_variants():
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-test",
        bases_per_difficulty=1,
        master_seed=123,
        train_ratio=0.0,
        validation_ratio=0.0,
        target_lengths={
            "easy": 3,
            "medium": 3,
            "hard": 3,
            "expert": 3,
        },
    )

    assert split_records["train"] == []
    assert split_records["validation"] == []
    assert len(split_records["test"]) == 15
    assert manifest.total_examples == 15
    assert manifest.split_counts == {"train": 0, "validation": 0, "test": 15}
    assert manifest.error_type_counts["clean"] == 4
    assert manifest.error_type_counts["E-RES"] == 4
    assert manifest.error_type_counts["E-INV"] == 4
    assert manifest.error_type_counts["E-CASC"] == 3


def test_generate_dataset_records_include_expected_schema():
    split_records, _ = generate_dataset_records(
        dataset_name="emoji-bench-test",
        bases_per_difficulty=1,
        master_seed=123,
        train_ratio=0.0,
        validation_ratio=0.0,
        target_lengths={
            "easy": 3,
            "medium": 3,
            "hard": 3,
            "expert": 3,
        },
    )

    records = split_records["test"]
    easy_base = [record for record in records if record["base_id"] == "easy-0000"]
    medium_base = [record for record in records if record["base_id"] == "medium-0000"]
    clean = next(record for record in medium_base if record["condition"] == "clean")
    e_inv = next(record for record in medium_base if record["error_type"] == "E-INV")

    assert clean["error_type"] is None
    assert clean["expected_error_step"] is None
    assert clean["actual_step_count"] >= 1

    assert e_inv["has_error"] is True
    assert e_inv["expected_error_step"] is not None
    assert e_inv["actual_step_count"] >= 1
    assert "expected_correct_result" not in e_inv
    assert "expected_correct_rule" not in e_inv

    assert len(easy_base) == len(DEFAULT_VARIANTS) - 1
    assert {record["error_type"] for record in easy_base} == {None, "E-RES", "E-INV"}
    assert len(medium_base) == len(DEFAULT_VARIANTS)
    assert len({record["system_json"] for record in medium_base}) == 1
    assert len({record["chain_seed"] for record in medium_base}) == 1
    assert {record["split"] for record in medium_base} == {"test"}


def test_write_dataset_outputs_jsonl_and_manifest(tmp_path):
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-test",
        bases_per_difficulty=2,
        master_seed=123,
        train_ratio=0.5,
        validation_ratio=0.0,
        target_lengths={
            "easy": 3,
            "medium": 3,
            "hard": 3,
            "expert": 3,
        },
    )
    output_dir = write_dataset(
        tmp_path / "dataset",
        split_records,
        manifest,
        repo_id="huyxdang/emoji-bench-test",
    )

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    assert train_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()
    assert readme_path.exists()

    first_line = train_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert record["split"] == "train"
    assert "prompt" in record

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["dataset_name"] == "emoji-bench-test"
    assert "generator_commit" in manifest_data

    card_text = readme_path.read_text(encoding="utf-8")
    assert "huyxdang/emoji-bench-test" in card_text


def test_generate_dataset_records_supports_e_casc_only():
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-e-casc",
        bases_per_difficulty=1,
        master_seed=123,
        train_ratio=0.0,
        validation_ratio=0.0,
        target_lengths={
            "easy": 4,
            "medium": 4,
            "hard": 4,
            "expert": 4,
        },
        variants=(
            DatasetVariant(
                name="e_casc",
                condition=Condition.ERROR_INJECTED,
                error_type=ErrorType.E_CASC,
                has_error=True,
            ),
        ),
    )

    assert len(split_records["test"]) == 4
    assert manifest.total_examples == 4
    assert manifest.error_type_counts == {"E-CASC": 4}


def test_generate_dataset_script_supports_error_type_and_count(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--dataset-name",
            "emoji-bench-e-casc-cli",
            "--output-dir",
            str(output_dir),
            "--bases-per-difficulty",
            "1",
            "--error-type",
            "E-CASC",
            "--count",
            "8",
            "--target-length",
            "4",
            "--train-ratio",
            "0",
            "--validation-ratio",
            "0",
            "--master-seed",
            "123",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["total_examples"] == 8
    assert summary["error_type_counts"] == {"E-CASC": 8}
    assert manifest["total_examples"] == 8
    assert manifest["error_type_counts"] == {"E-CASC": 8}


