"""Phase 3: dataset-level generation for the E-CONTINUE benchmark.

This module generates exact-count continuation-only datasets, enforces the
realized-length and runway floors, and records per-difficulty rejection
reasons so yield can be tuned without changing the row schema.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from emoji_bench.continuation_benchmark import (
    ContinuationInstance,
    continuation_record,
    generate_continuation_instance,
)
from emoji_bench.dataset_io import (
    DIFFICULTY_CONFIGS,
    DatasetManifest,
    git_commit,
)
from emoji_bench.formatter import system_to_json
from emoji_bench.generator import generate_system


# Locked defaults from codex_plan.md: targets chosen so the midpoint policy
# always leaves a meaningful prefix and runway. See also MIN_REALIZED_X.
DEFAULT_CONTINUATION_TARGET_LENGTHS: dict[str, int] = {
    "easy": 6,
    "medium": 8,
    "hard": 10,
    "expert": 14,
}
# Realized-X floor. generate_chain is a ±2 best-effort sampler, so a target
# of 6 can return as low as 4. Anything shorter would collapse the midpoint
# policy to Y = 1, which is not a meaningful "bad prefix" signal.
MIN_REALIZED_X: int = 4

MAX_CHAIN_SEED_ATTEMPTS: int = 250
MAX_BASE_ATTEMPTS_PER_DIFFICULTY: int = 20_000
ERROR_SEED_OFFSET: int = 101


# Rejection reasons (kept as strings so they serialize cleanly to JSON).
R_CHAIN_TOO_SHORT = "chain_too_short"
R_INSUFFICIENT_RUNWAY = "insufficient_runway"
R_NO_ELIGIBLE_IN_CHAIN = "no_eligible_in_chain"
R_NO_ELIGIBLE_IN_WINDOW = "no_eligible_in_window"
R_CASCADE_CONVERGENT = "cascade_convergent"
R_OTHER_INJECTOR_ERROR = "other_injector_error"

REJECTION_REASONS: tuple[str, ...] = (
    R_CHAIN_TOO_SHORT,
    R_INSUFFICIENT_RUNWAY,
    R_NO_ELIGIBLE_IN_CHAIN,
    R_NO_ELIGIBLE_IN_WINDOW,
    R_CASCADE_CONVERGENT,
    R_OTHER_INJECTOR_ERROR,
)


def _seed_root(master_seed: int, difficulty_index: int, base_index: int) -> int:
    return master_seed * 1_000_000 + difficulty_index * 10_000 + base_index * 100


def _per_difficulty_targets(count: int) -> dict[str, int]:
    difficulty_names = tuple(DIFFICULTY_CONFIGS)
    n = len(difficulty_names)
    return {
        difficulty: count // n + (1 if index < count % n else 0)
        for index, difficulty in enumerate(difficulty_names)
    }


def _classify_injector_error(msg: str) -> str:
    # Mirrors the error strings raised by generate_continuation_instance and
    # inject_cascading_wrong_result. Update both sides together if either
    # message text changes.
    if "no cascading-eligible steps" in msg:
        return R_NO_ELIGIBLE_IN_CHAIN
    if "No eligible error step in the midpoint" in msg:
        return R_NO_ELIGIBLE_IN_WINDOW
    if "non-convergent cascading error" in msg:
        return R_CASCADE_CONVERGENT
    if "change the final result" in msg:
        return R_CASCADE_CONVERGENT
    return R_OTHER_INJECTOR_ERROR


def _try_generate(
    *,
    system: Any,
    target_step_count: int,
    chain_seed: int,
    error_seed: int,
) -> tuple[ContinuationInstance | None, str | None]:
    """Run the single-instance generator and apply the dataset-level filters.

    Returns ``(instance, None)`` on success or ``(None, reason)`` on rejection.
    """
    try:
        instance = generate_continuation_instance(
            system,
            length=target_step_count,
            chain_seed=chain_seed,
            error_seed=error_seed,
        )
    except ValueError as exc:
        return None, _classify_injector_error(str(exc))

    if instance.chain_length_x < MIN_REALIZED_X:
        return None, R_CHAIN_TOO_SHORT

    remaining_steps = len(instance.mutated_chain.steps) - instance.prefill_error_step
    required_runway = (instance.chain_length_x + 1) // 2  # ceil(X / 2)
    if remaining_steps < required_runway:
        return None, R_INSUFFICIENT_RUNWAY

    return instance, None


def _select_instance(
    *,
    system: Any,
    target_step_count: int,
    seed_root: int,
    rejection_bucket: dict[str, int],
) -> tuple[int, int, ContinuationInstance]:
    """Iterate chain seeds until one produces a usable instance."""
    error_seed = seed_root + ERROR_SEED_OFFSET
    for attempt in range(MAX_CHAIN_SEED_ATTEMPTS):
        chain_seed = seed_root + 29 + attempt
        instance, reason = _try_generate(
            system=system,
            target_step_count=target_step_count,
            chain_seed=chain_seed,
            error_seed=error_seed,
        )
        if instance is not None:
            return chain_seed, error_seed, instance
        assert reason is not None
        rejection_bucket[reason] = rejection_bucket.get(reason, 0) + 1

    raise RuntimeError(
        f"Failed to find a usable continuation instance after "
        f"{MAX_CHAIN_SEED_ATTEMPTS} chain-seed attempts"
    )


def _resolve_target_lengths(
    overrides: dict[str, int] | None,
) -> dict[str, int]:
    resolved = dict(DEFAULT_CONTINUATION_TARGET_LENGTHS)
    if overrides:
        for key, value in overrides.items():
            if key not in DIFFICULTY_CONFIGS:
                raise ValueError(f"unknown difficulty in target lengths: {key}")
            resolved[key] = int(value)
    return resolved


def generate_continuation_dataset_records(
    *,
    dataset_name: str,
    count: int,
    master_seed: int,
    target_lengths: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], DatasetManifest]:
    """Generate an exact-count E-CONTINUE dataset with per-difficulty rejection logging."""
    if count < 1:
        raise ValueError("count must be >= 1")

    resolved_lengths = _resolve_target_lengths(target_lengths)
    targets = _per_difficulty_targets(count)

    rejection_counts: dict[str, dict[str, int]] = {
        difficulty: {reason: 0 for reason in REJECTION_REASONS}
        for difficulty in DIFFICULTY_CONFIGS
    }
    produced_per_difficulty = {difficulty: 0 for difficulty in DIFFICULTY_CONFIGS}
    bases_used_per_difficulty = {difficulty: 0 for difficulty in DIFFICULTY_CONFIGS}
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    example_index = 0

    for difficulty_index, difficulty_name in enumerate(DIFFICULTY_CONFIGS):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        target_step_count = resolved_lengths[difficulty_name]
        rejection_bucket = rejection_counts[difficulty_name]
        base_index = 0

        while produced_per_difficulty[difficulty_name] < targets[difficulty_name]:
            if base_index >= MAX_BASE_ATTEMPTS_PER_DIFFICULTY:
                raise RuntimeError(
                    "Failed to generate enough continuation bases for "
                    f"{difficulty_name} after {MAX_BASE_ATTEMPTS_PER_DIFFICULTY} attempts. "
                    f"Rejection counts so far: {rejection_bucket}"
                )

            seed_root = _seed_root(master_seed, difficulty_index, base_index)
            system_seed = seed_root + 11
            base_id = f"{difficulty_name}-{base_index:04d}"
            system = generate_system(random_seed=system_seed, **config)

            try:
                chain_seed, error_seed, instance = _select_instance(
                    system=system,
                    target_step_count=target_step_count,
                    seed_root=seed_root,
                    rejection_bucket=rejection_bucket,
                )
            except RuntimeError:
                base_index += 1
                continue

            record = continuation_record(
                instance=instance,
                example_id=f"{dataset_name}-{example_index:06d}",
                base_id=base_id,
                split="test",
                difficulty=difficulty_name,
                system_json=system_to_json(system),
                system_seed=system_seed,
                chain_seed=chain_seed,
                error_seed=error_seed,
                target_step_count=target_step_count,
            )
            split_records["test"].append(record)

            example_index += 1
            produced_per_difficulty[difficulty_name] += 1
            base_index += 1
            bases_used_per_difficulty[difficulty_name] = base_index

    split_counts = {split: len(records) for split, records in split_records.items()}
    difficulty_counts = Counter(
        record["difficulty"]
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
        bases_per_difficulty=max(bases_used_per_difficulty.values(), default=0),
        target_lengths=resolved_lengths,
        split_counts=dict(split_counts),
        difficulty_counts=dict(difficulty_counts),
        error_type_counts=dict(error_type_counts),
        generator_commit=git_commit(),
        rejection_counts=rejection_counts,
    )
    return split_records, manifest
