from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


DIFFICULTY_CONFIGS: dict[str, dict[str, int]] = {
    "easy": {
        "n_symbols": 3,
        "n_base_ops": 1,
        "n_derived_ops": 0,
        "n_transformations": 0,
    },
    "medium": {
        "n_symbols": 4,
        "n_base_ops": 1,
        "n_derived_ops": 1,
        "n_transformations": 1,
    },
    "hard": {
        "n_symbols": 5,
        "n_base_ops": 2,
        "n_derived_ops": 1,
        "n_transformations": 1,
    },
    "expert": {
        "n_symbols": 6,
        "n_base_ops": 2,
        "n_derived_ops": 2,
        "n_transformations": 2,
    },
}


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    total_examples: int
    bases_per_difficulty: int
    master_seed: int
    target_lengths: dict[str, int]
    difficulty_configs: dict[str, dict[str, int]]
    split_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    error_type_counts: dict[str, int]
    generator_commit: str | None
    rejection_counts: dict[str, dict[str, int]] | None = None


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def build_continuation_dataset_card(
    manifest: DatasetManifest,
    *,
    repo_id: str | None = None,
) -> str:
    repo_ref = repo_id or manifest.dataset_name
    return (
        "---\n"
        "pretty_name: Emoji-Bench\n"
        "task_categories:\n"
        "- text-generation\n"
        "language:\n"
        "- en\n"
        "---\n\n"
        f"# {repo_ref}\n\n"
        "This dataset contains continuation-benchmark rows for Emoji-Bench.\n\n"
        "Each row stores the first user turn, the prefilled assistant prefix ending on the bad step, "
        "and the scoring targets for the continuation task.\n\n"
        "## Schema\n\n"
        "- `example_id`: unique row id\n"
        "- `base_id`: shared id for the underlying generated formal system\n"
        "- `split`: dataset split (`test` in the current release)\n"
        "- `difficulty`: easy / medium / hard / expert\n"
        "- `error_type`: injected error label (`E-CONTINUE` in the current release)\n"
        "- `turn_1_user`: rules + expression + formatting instructions\n"
        "- `turn_1_assistant_prefill`: partial derivation ending on the injected error\n"
        "- `ground_truth_final_output`: correct final symbol from the clean chain\n"
        "- `wrong_branch_final_output`: final symbol reached by blindly continuing from the bad state\n"
        "- `chain_length_x`: realized clean derivation length\n"
        "- `prefill_error_step`: step number of the injected error\n"
        "- `target_step_count`: requested target length used during generation\n"
        "- `system_json`: JSON serialization of the formal system\n"
        "- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility\n\n"
        "The default Turn 2 user message is `Please continue.` and is applied at evaluation time; "
        "prompt-strength variants can be requested without regenerating the dataset.\n\n"
        "## Counts\n\n"
        f"- total_examples: {manifest.total_examples}\n"
        f"- master_seed: {manifest.master_seed}\n"
        f"- split_counts: {json.dumps(manifest.split_counts, ensure_ascii=False)}\n"
        f"- difficulty_counts: {json.dumps(manifest.difficulty_counts, ensure_ascii=False)}\n"
        f"- error_type_counts: {json.dumps(manifest.error_type_counts, ensure_ascii=False)}\n"
        f"- target_lengths: {json.dumps(manifest.target_lengths, ensure_ascii=False)}\n"
        f"- difficulty_configs: {json.dumps(manifest.difficulty_configs, ensure_ascii=False)}\n"
        f"- generator_commit: {manifest.generator_commit}\n\n"
        "## Load\n\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f'ds = load_dataset("{repo_ref}")\n'
        "print(ds)\n"
        "```\n"
    )


def write_dataset(
    output_dir: str | Path,
    split_records: dict[str, list[dict[str, object]]],
    manifest: DatasetManifest,
    *,
    repo_id: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, records in split_records.items():
        if not records:
            continue
        split_path = output_path / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    card_path = output_path / "README.md"
    card_path.write_text(
        build_continuation_dataset_card(manifest, repo_id=repo_id),
        encoding="utf-8",
    )
    return output_path


def push_dataset_to_hub(
    output_dir: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub. "
            'Install with `pip install -e ".[hf]"`.'
        ) from exc

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
    )
