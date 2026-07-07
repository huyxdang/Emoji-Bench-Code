from __future__ import annotations

import re

from emoji_bench.dataset.continuation_benchmark import (
    continuation_record,
    generate_continuation_instance,
)
from emoji_bench.domain.formatter import system_to_json
from emoji_bench.domain.generator import generate_system
from emoji_bench.scoring.behavior_classifier import (
    BEHAVIOR_MODES,
    build_reference_states,
    classify_behavior,
)

_STEP_LINE = re.compile(r"^Step\s+(\d+)\s*:\s*(.+?)\s*=\s*(.+?)(?:\s{2,}\[by.*)?$")


def _make_record() -> dict:
    system = generate_system(
        n_symbols=3,
        n_base_ops=1,
        n_derived_ops=0,
        n_transformations=0,
        random_seed=11,
    )
    instance = generate_continuation_instance(system, length=6, chain_seed=7, error_seed=13)
    return continuation_record(
        instance=instance,
        example_id="behavior-test-000",
        base_id="easy-0000",
        split="test",
        difficulty="easy",
        system_json=system_to_json(system),
        system_seed=11,
        chain_seed=7,
        error_seed=13,
        target_step_count=6,
    )


def _row_states(record: dict) -> dict[str, str]:
    """Extract the rendered reference states straight from the record text."""
    error_step = record["prefill_error_step"]
    prefill_lines = record["turn_1_assistant_prefill"].splitlines()
    start = prefill_lines[0][len("Start: "):]
    last = _STEP_LINE.match(prefill_lines[-1])
    assert last is not None and int(last.group(1)) == error_step
    clean_after = None
    for line in record["clean_derivation"].splitlines():
        match = _STEP_LINE.match(line)
        if match and int(match.group(1)) == error_step:
            clean_after = match.group(3)
    assert clean_after is not None
    return {
        "start": start,
        "before_error": last.group(2),
        "corrupted_after": last.group(3),
        "clean_after": clean_after,
        "error_step": error_step,
    }


def test_reference_states_match_record_text():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    assert refs.error_step == states["error_step"]
    # The corrupted and clean post-error states must differ (injected error).
    assert refs.corrupted_after != refs.clean_after


def test_classify_inplace_correction():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    text = (
        f"Step {states['error_step']}: {states['before_error']} = "
        f"{states['clean_after']}    [by ⊕ table]\n"
        f"Final Output: {record['ground_truth_final_output']}"
    )
    result = classify_behavior(text, refs)
    assert result.mode == "inplace_correction"
    assert result.first_step_number == states["error_step"]


def test_classify_continue_from_corrupted():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    next_step = states["error_step"] + 1
    text = (
        f"Step {next_step}: {states['corrupted_after']} = "
        f"{states['corrupted_after']}    [by ⊕ table]"
    )
    result = classify_behavior(text, refs)
    assert result.mode == "continue_from_corrupted"
    assert result.first_step_number == next_step


def test_classify_continue_from_corrected():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    next_step = states["error_step"] + 1
    text = f"Step {next_step}: {states['clean_after']} = {states['clean_after']}"
    result = classify_behavior(text, refs)
    assert result.mode == "continue_from_corrected"


def test_classify_full_restart():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    text = f"Step 1: {states['start']} = {states['start']}    [by ⊕ table]"
    result = classify_behavior(text, refs)
    assert result.mode == "full_restart"
    assert result.first_step_number == 1


def test_classify_no_steps_for_prose_only():
    record = _make_record()
    refs = build_reference_states(record)
    result = classify_behavior("I believe there is an error in the work above.", refs)
    assert result.mode == "no_steps"
    assert result.first_step_number is None


def test_classify_other_for_unrecognized_state():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    # A bare-symbol LHS is parseable but matches no reference state
    # (all references have at least one operator at the error step).
    symbol = record["ground_truth_final_output"]
    text = f"Step 3: {symbol} = {symbol}"
    result = classify_behavior(text, refs)
    assert result.mode == "other"
    assert states["start"] != symbol


def test_prose_step_mention_is_skipped():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    text = (
        f"There was an error in the previous Step {states['error_step']}: "
        "the lookup was wrong.\n"
        f"Step {states['error_step']}: {states['before_error']} = {states['clean_after']}"
    )
    result = classify_behavior(text, refs)
    assert result.mode == "inplace_correction"


def test_markdown_wrapped_step_line_still_classifies():
    record = _make_record()
    refs = build_reference_states(record)
    states = _row_states(record)
    text = f"**Step 1:** {states['start']} = {states['start']}"
    result = classify_behavior(text, refs)
    assert result.mode == "full_restart"


def test_behavior_modes_constant_is_exhaustive():
    record = _make_record()
    refs = build_reference_states(record)
    result = classify_behavior("Step 1: nonsense text", refs)
    # Unparseable step-shaped line falls through to no_steps.
    assert result.mode in BEHAVIOR_MODES
