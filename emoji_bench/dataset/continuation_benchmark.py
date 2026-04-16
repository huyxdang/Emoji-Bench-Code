from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emoji_bench.benchmark_types import ErrorInfo, ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.chain_types import DerivationChain
from emoji_bench.continuation_formatter import (
    format_continuation_prefill,
    format_continuation_turn_1_user,
)
from emoji_bench.error_injector import (
    get_cascading_eligible_steps,
    inject_cascading_wrong_result,
)
from emoji_bench.types import FormalSystem, Symbol


@dataclass(frozen=True)
class ContinuationInstance:
    """A single multi-turn continuation benchmark instance."""

    system: FormalSystem
    clean_chain: DerivationChain
    mutated_chain: DerivationChain
    error_info: ErrorInfo

    # Conversation pieces.
    turn_1_user: str
    turn_1_assistant_prefill: str

    # Structural metadata.
    chain_length_x: int       # len(clean_chain.steps)
    prefill_error_step: int   # 1-indexed step number where the error was injected

    # Scoring axes.
    ground_truth_final_output: Symbol
    wrong_branch_final_output: Symbol

    instance_id: str | None = None


def _preferred_error_steps(chain_length_x: int) -> tuple[int, ...]:
    """Return candidate error steps in preference order: midpoint, then ±1.

    The locked policy places the error at ``⌊X/2⌋`` with a ±1 jitter tolerance
    so the injector has room to find a valid cascading slot. Candidates are
    clamped to the interval ``[1, X-1]`` because the cascading injector
    requires at least one remaining step of runway after the cutoff.
    """
    if chain_length_x < 2:
        return ()
    midpoint = chain_length_x // 2
    candidates: list[int] = []
    for offset in (0, 1, -1):
        step = midpoint + offset
        if 1 <= step <= chain_length_x - 1 and step not in candidates:
            candidates.append(step)
    return tuple(candidates)


def generate_continuation_instance(
    system: FormalSystem,
    *,
    length: int,
    chain_seed: int,
    error_seed: int | None = None,
    instance_id: str | None = None,
) -> ContinuationInstance:
    """Generate a single multi-turn continuation instance.

    Strategy:
      1. Build a clean derivation chain of approximately ``length`` steps.
      2. Pick a target error step ``Y ≈ ⌊X/2⌋`` with a ±1 jitter window.
      3. Inject a cascading wrong result at the first target step that
         (a) is in the cascading-eligible set and (b) the injector can
         successfully make non-convergent.
      4. Set the prefill cutoff to the injected error step ``Y``.

    Non-convergence (``wrong_branch_final_output != ground_truth_final_output``)
    is guaranteed by ``inject_cascading_wrong_result`` — it only returns a
    mutation whose recomputed final symbol differs from the clean final
    symbol, otherwise it raises ``ValueError``.
    """
    if error_seed is None:
        error_seed = chain_seed + 1

    clean_chain = generate_chain(system, length=length, seed=chain_seed)
    chain_length_x = len(clean_chain.steps)

    eligible_step_numbers = {
        step.step_number for step in get_cascading_eligible_steps(clean_chain)
    }
    if not eligible_step_numbers:
        raise ValueError(
            f"Chain of length {chain_length_x} has no cascading-eligible steps"
        )

    candidates = [
        step
        for step in _preferred_error_steps(chain_length_x)
        if step in eligible_step_numbers
    ]
    if not candidates:
        raise ValueError(
            "No eligible error step in the midpoint ±1 window for a chain "
            f"of length {chain_length_x}"
        )

    mutated_chain: DerivationChain | None = None
    error_info: ErrorInfo | None = None
    for target_step in candidates:
        try:
            mutated_chain, error_info = inject_cascading_wrong_result(
                clean_chain,
                system,
                step_number=target_step,
                seed=error_seed,
            )
        except ValueError:
            continue
        break

    if mutated_chain is None or error_info is None:
        raise ValueError(
            "Could not inject a non-convergent cascading error at any candidate "
            f"step (tried {candidates}) for a chain of length {chain_length_x}"
        )

    prefill_error_step = error_info.step_number

    turn_1_user = format_continuation_turn_1_user(system, clean_chain)
    turn_1_assistant_prefill = format_continuation_prefill(
        mutated_chain, prefill_error_step, system
    )

    return ContinuationInstance(
        system=system,
        clean_chain=clean_chain,
        mutated_chain=mutated_chain,
        error_info=error_info,
        turn_1_user=turn_1_user,
        turn_1_assistant_prefill=turn_1_assistant_prefill,
        chain_length_x=chain_length_x,
        prefill_error_step=prefill_error_step,
        ground_truth_final_output=clean_chain.final_result,
        wrong_branch_final_output=mutated_chain.final_result,
        instance_id=instance_id,
    )


def continuation_record(
    *,
    instance: ContinuationInstance,
    example_id: str,
    base_id: str,
    split: str,
    difficulty: str,
    system_json: str,
    system_seed: int,
    chain_seed: int,
    error_seed: int,
    target_step_count: int,
) -> dict[str, Any]:
    """Serialize a ``ContinuationInstance`` into the on-disk record shape.

    The record is a flat dict ready for ``jsonl`` emission. It keeps the
    turn-structured conversation pieces, the scoring ground truth, the
    structural metadata, and the reproducibility fields needed by the
    continuation benchmark.
    """
    return {
        # Identity.
        "example_id": example_id,
        "base_id": base_id,
        "split": split,
        "difficulty": difficulty,
        "error_type": ErrorType.E_CONTINUE.value,

        # Conversation.
        "turn_1_user": instance.turn_1_user,
        "turn_1_assistant_prefill": instance.turn_1_assistant_prefill,

        # Scoring.
        "ground_truth_final_output": instance.ground_truth_final_output.emoji,
        "wrong_branch_final_output": instance.wrong_branch_final_output.emoji,

        # Structural metadata.
        "chain_length_x": instance.chain_length_x,
        "prefill_error_step": instance.prefill_error_step,
        "target_step_count": target_step_count,

        # Repro.
        "system_json": system_json,
        "system_seed": system_seed,
        "chain_seed": chain_seed,
        "error_seed": error_seed,
    }
