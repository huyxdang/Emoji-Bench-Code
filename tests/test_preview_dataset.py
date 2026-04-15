import json
import subprocess
import sys
from pathlib import Path

from emoji_bench.continuation_dataset import generate_continuation_dataset_records
from emoji_bench.dataset_io import write_dataset


def test_preview_dataset_script_renders_metadata_and_prompt(tmp_path):
    split_records, manifest = generate_continuation_dataset_records(
        dataset_name="emoji-bench-preview-test",
        count=4,
        master_seed=123,
    )
    output_dir = write_dataset(tmp_path / "dataset", split_records, manifest)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_dataset.py",
            str(output_dir),
            "--count",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    first_record = json.loads((output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "=== DATASET ===" in result.stdout
    assert "Example 1/1" in result.stdout
    assert f"example_id: {first_record['example_id']}" in result.stdout
    assert first_record["turn_1_user"] in result.stdout
    assert first_record["turn_1_assistant_prefill"] in result.stdout
    assert "=== TURN 2 USER ===" in result.stdout
    assert "Please continue." in result.stdout


def test_preview_dataset_script_can_select_specific_example(tmp_path):
    split_records, manifest = generate_continuation_dataset_records(
        dataset_name="emoji-bench-preview-test",
        count=4,
        master_seed=123,
    )
    output_dir = write_dataset(tmp_path / "dataset", split_records, manifest)
    repo_root = Path(__file__).resolve().parents[1]
    records = [json.loads(line) for line in (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    target = records[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_dataset.py",
            str(output_dir / "test.jsonl"),
            "--example-id",
            target["example_id"],
            "--no-manifest",
            "--prompt-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "=== DATASET ===" not in result.stdout
    assert "example_id:" not in result.stdout
    assert target["turn_1_user"] in result.stdout
    assert target["turn_1_assistant_prefill"] in result.stdout
