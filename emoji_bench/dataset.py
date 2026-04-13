from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from emoji_bench.benchmark import generate_benchmark_instance
from emoji_bench.benchmark_types import Condition, ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.error_injector import (
    get_cascading_eligible_steps,
    get_invented_rule_eligible_steps,
    get_wrong_result_eligible_steps,
)
from emoji_bench.formatter import system_to_json
from emoji_bench.generator import generate_system


DEFAULT_TARGET_LENGTHS: dict[str, int] = {
    "easy": 3,
    "medium": 5,
    "hard": 7,
    "expert": 10,
}

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
class DatasetVariant:
    name: str
    condition: Condition
    error_type: ErrorType | None
    has_error: bool


DEFAULT_VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant(
        name="clean",
        condition=Condition.CLEAN,
        error_type=None,
        has_error=False,
    ),
    DatasetVariant(
        name="e_res",
        condition=Condition.ERROR_INJECTED,
        error_type=ErrorType.E_RES,
        has_error=True,
    ),
    DatasetVariant(
        name="e_inv",
        condition=Condition.ERROR_INJECTED,
        error_type=ErrorType.E_INV,
        has_error=True,
    ),
    DatasetVariant(
        name="e_casc",
        condition=Condition.ERROR_INJECTED,
        error_type=ErrorType.E_CASC,
        has_error=True,
    ),
)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    total_examples: int
    bases_per_difficulty: int
    target_lengths: dict[str, int]
    split_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    condition_counts: dict[str, int]
    error_type_counts: dict[str, int]
    generator_commit: str | None
    rejection_counts: dict[str, dict[str, int]] | None = None


MAX_CHAIN_SEED_ATTEMPTS = 100


def _seed_root(master_seed: int, difficulty_index: int, base_index: int) -> int:
    return master_seed * 1_000_000 + difficulty_index * 10_000 + base_index * 100


def _split_for_base(
    base_index: int,
    bases_per_difficulty: int,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    train_cutoff = int(bases_per_difficulty * train_ratio)
    validation_cutoff = train_cutoff + int(bases_per_difficulty * validation_ratio)
    if base_index < train_cutoff:
        return "train"
    if base_index < validation_cutoff:
        return "validation"
    return "test"


def _git_commit() -> str | None:
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


def _variant_seed_offsets(
    variants: tuple[DatasetVariant, ...],
) -> dict[DatasetVariant, int]:
    return {
        variant: 101 + index
        for index, variant in enumerate(variants)
        if variant.has_error
    }


def _error_seed_for_variant(
    seed_root: int,
    variant: DatasetVariant,
    seed_offsets: dict[DatasetVariant, int],
) -> int | None:
    if not variant.has_error:
        return None
    return seed_root + seed_offsets[variant]


def _can_generate_variant_instance(
    *,
    system: Any,
    target_step_count: int,
    chain_seed: int,
    error_seed: int | None,
    variant: DatasetVariant,
) -> bool:
    try:
        generate_benchmark_instance(
            system,
            length=target_step_count,
            condition=variant.condition,
            error_type=variant.error_type or ErrorType.E_RES,
            chain_seed=chain_seed,
            error_seed=error_seed,
        )
    except ValueError:
        return False
    return True


def _variant_supported_by_chain(
    system: Any,
    chain: Any,
    variant: DatasetVariant,
) -> bool:
    if variant.condition is Condition.CLEAN:
        return True
    if variant.error_type is ErrorType.E_RES:
        return bool(get_wrong_result_eligible_steps(chain))
    if variant.error_type is ErrorType.E_INV:
        return bool(get_invented_rule_eligible_steps(chain, system))
    if variant.error_type is ErrorType.E_CASC:
        return bool(get_cascading_eligible_steps(chain))
    return False


def _select_chain_seed(
    *,
    system: Any,
    target_step_count: int,
    seed_root: int,
    variants: tuple[DatasetVariant, ...],
    seed_offsets: dict[DatasetVariant, int],
) -> tuple[int, tuple[DatasetVariant, ...]]:
    best_chain_seed = seed_root + 29
    best_supported_variants: tuple[DatasetVariant, ...] = ()

    for attempt in range(MAX_CHAIN_SEED_ATTEMPTS):
        chain_seed = seed_root + 29 + attempt
        chain = generate_chain(system, length=target_step_count, seed=chain_seed)
        supported_variants = tuple(
            variant
            for variant in variants
            if _variant_supported_by_chain(system, chain, variant)
            and _can_generate_variant_instance(
                system=system,
                target_step_count=target_step_count,
                chain_seed=chain_seed,
                error_seed=_error_seed_for_variant(seed_root, variant, seed_offsets),
                variant=variant,
            )
        )
        if len(supported_variants) > len(best_supported_variants):
            best_chain_seed = chain_seed
            best_supported_variants = supported_variants

        if len(supported_variants) == len(variants):
            return chain_seed, supported_variants

    if best_supported_variants:
        return best_chain_seed, best_supported_variants

    raise RuntimeError(
        "Failed to find a chain compatible with any requested dataset variants "
        f"after {MAX_CHAIN_SEED_ATTEMPTS} attempts"
    )


def _example_record(
    *,
    dataset_name: str,
    base_id: str,
    example_index: int,
    split: str,
    difficulty: str,
    target_step_count: int,
    system_seed: int,
    chain_seed: int,
    error_seed: int | None,
    variant: DatasetVariant,
    instance: Any,
    system_json: str,
) -> dict[str, Any]:
    error_info = instance.error_info
    expected_error_step = error_info.step_number if error_info is not None else None

    return {
        "example_id": f"{dataset_name}-{example_index:06d}",
        "base_id": base_id,
        "split": split,
        "difficulty": difficulty,
        "condition": variant.condition.value,
        "error_type": variant.error_type.value if variant.error_type is not None else None,
        "has_error": variant.has_error,
        "prompt": instance.prompt,
        "actual_step_count": len(instance.chain.steps),
        "target_step_count": target_step_count,
        "expected_error_step": expected_error_step,
        "system_json": system_json,
        "system_seed": system_seed,
        "chain_seed": chain_seed,
        "error_seed": error_seed,
    }


def generate_dataset_records(
    *,
    dataset_name: str,
    bases_per_difficulty: int,
    master_seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    target_lengths: dict[str, int] | None = None,
    variants: tuple[DatasetVariant, ...] = DEFAULT_VARIANTS,
) -> tuple[dict[str, list[dict[str, Any]]], DatasetManifest]:
    if bases_per_difficulty < 1:
        raise ValueError("bases_per_difficulty must be >= 1")
    if train_ratio < 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio > 1:
        raise ValueError("train_ratio + validation_ratio must be <= 1")

    resolved_lengths = dict(DEFAULT_TARGET_LENGTHS)
    if target_lengths is not None:
        resolved_lengths.update(target_lengths)
    seed_offsets = _variant_seed_offsets(variants)

    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    example_index = 0

    for difficulty_index, difficulty_name in enumerate(DIFFICULTY_CONFIGS):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        target_step_count = resolved_lengths[difficulty_name]

        for base_index in range(bases_per_difficulty):
            seed_root = _seed_root(master_seed, difficulty_index, base_index)
            system_seed = seed_root + 11
            base_id = f"{difficulty_name}-{base_index:04d}"
            split = _split_for_base(
                base_index,
                bases_per_difficulty,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
            )

            system = generate_system(random_seed=system_seed, **config)
            chain_seed, active_variants = _select_chain_seed(
                system=system,
                target_step_count=target_step_count,
                seed_root=seed_root,
                variants=variants,
                seed_offsets=seed_offsets,
            )
            system_json = system_to_json(system)

            for variant in active_variants:
                error_seed = _error_seed_for_variant(seed_root, variant, seed_offsets)
                instance = generate_benchmark_instance(
                    system,
                    length=target_step_count,
                    condition=variant.condition,
                    error_type=variant.error_type or ErrorType.E_RES,
                    chain_seed=chain_seed,
                    error_seed=error_seed,
                    instance_id=f"{base_id}-{variant.name}",
                )
                record = _example_record(
                    dataset_name=dataset_name,
                    base_id=base_id,
                    example_index=example_index,
                    split=split,
                    difficulty=difficulty_name,
                    target_step_count=target_step_count,
                    system_seed=system_seed,
                    chain_seed=chain_seed,
                    error_seed=error_seed,
                    variant=variant,
                    instance=instance,
                    system_json=system_json,
                )
                split_records[split].append(record)
                example_index += 1

    split_counts = {split: len(records) for split, records in split_records.items()}
    difficulty_counts = Counter(
        record["difficulty"]
        for records in split_records.values()
        for record in records
    )
    condition_counts = Counter(
        record["condition"]
        for records in split_records.values()
        for record in records
    )
    error_type_counts = Counter(
        record["error_type"] or "clean"
        for records in split_records.values()
        for record in records
    )

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        total_examples=sum(split_counts.values()),
        bases_per_difficulty=bases_per_difficulty,
        target_lengths=resolved_lengths,
        split_counts=dict(split_counts),
        difficulty_counts=dict(difficulty_counts),
        condition_counts=dict(condition_counts),
        error_type_counts=dict(error_type_counts),
        generator_commit=_git_commit(),
    )
    return split_records, manifest


def write_dataset(
    output_dir: str | Path,
    split_records: dict[str, list[dict[str, Any]]],
    manifest: DatasetManifest,
    *,
    repo_id: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, records in split_records.items():
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
    card_path.write_text(build_dataset_card(manifest, repo_id=repo_id), encoding="utf-8")
    return output_path


def build_dataset_card(manifest: DatasetManifest, *, repo_id: str | None = None) -> str:
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
        "This dataset contains prompt-only benchmark instances for Emoji-Bench.\n\n"
        "## Schema\n\n"
        "- `example_id`: unique row id\n"
        "- `base_id`: shared id across clean/error variants of the same underlying problem\n"
        "- `split`: train / validation / test\n"
        "- `difficulty`: easy / medium / hard / expert\n"
        "- `condition`: clean or error_injected\n"
        "- `error_type`: null or an injected error label such as E-RES, E-INV, E-CASC, or E-RECONV\n"
        "- `has_error`: whether the prompt contains an injected error\n"
        "- `prompt`: full benchmark prompt\n"
        "- `actual_step_count`: realized number of derivation steps\n"
        "- `target_step_count`: requested target length used during generation\n"
        "- `expected_error_step`: ground-truth step with the injected error, or null on clean rows\n"
        "- `system_json`: JSON serialization of the underlying formal system\n"
        "- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility\n\n"
        "## Counts\n\n"
        f"- total_examples: {manifest.total_examples}\n"
        f"- split_counts: {json.dumps(manifest.split_counts, ensure_ascii=False)}\n"
        f"- difficulty_counts: {json.dumps(manifest.difficulty_counts, ensure_ascii=False)}\n"
        f"- condition_counts: {json.dumps(manifest.condition_counts, ensure_ascii=False)}\n"
        f"- error_type_counts: {json.dumps(manifest.error_type_counts, ensure_ascii=False)}\n"
        f"- generator_commit: {manifest.generator_commit}\n\n"
        "## Load\n\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        f'ds = load_dataset("{repo_ref}")\n'
        "print(ds)\n"
        "```\n"
    )


def push_dataset_to_hub(
    output_dir: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised by runtime usage only
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
