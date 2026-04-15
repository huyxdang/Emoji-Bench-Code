from emoji_bench.continuation_scorer import (
    classify_outcome,
    detects_loose,
    detects_strict,
    extract_final_output,
    score_prediction,
)


# --- Final output extraction ----------------------------------------------


def test_extract_final_output_basic():
    assert extract_final_output("Step 5: x = y\n\nFinal Output: 🪈") == "🪈"


def test_extract_final_output_returns_none_when_marker_missing():
    assert extract_final_output("Step 1: ... Step 2: ... done.") is None


def test_extract_final_output_takes_last_occurrence_when_model_revises():
    text = "Wait, Final Output: 🪵 was wrong.\n\nFinal Output: 🪈"
    assert extract_final_output(text) == "🪈"


def test_extract_final_output_strips_markdown_emphasis():
    assert extract_final_output("Final Output: **🪈**") == "🪈"
    assert extract_final_output("Final Output: `🪈`") == "🪈"


def test_extract_final_output_trims_trailing_commentary():
    assert extract_final_output("Final Output: 🪈 (the correct answer)") == "🪈"


def test_extract_final_output_trims_trailing_punctuation():
    assert extract_final_output("Final Output: 🪈.") == "🪈"


# --- Loose detection ------------------------------------------------------


def test_detects_loose_common_phrases():
    assert detects_loose("Wait, that's wrong.")
    assert detects_loose("let me recheck step 2")
    assert detects_loose("Step 2 should be 🪈, not 🥟")
    assert detects_loose("I made a mistake in step 2")
    assert detects_loose("Hmm, that doesn't match the table.")
    assert detects_loose("Actually, let me reconsider.")
    assert detects_loose("Oops, step 2 is incorrect.")


def test_detects_loose_does_not_fire_on_benign_words():
    # "waiting" should not hit "\bwait\b".
    assert not detects_loose("I am waiting for the next step.")
    # Pure continuation without doubt markers.
    assert not detects_loose(
        "Step 3: (a ⊕ b) = c    [by ⊕ table]\n\nFinal Output: c"
    )


# --- Strict detection -----------------------------------------------------


def test_detects_strict_requires_step_reference_near_doubt():
    # Doubt + step reference within the window → strict.
    assert detects_strict("Wait, step 2 looks wrong.", error_step=2)
    # Doubt but no step reference → not strict.
    assert not detects_strict("Hmm, something feels off.", error_step=2)
    # Doubt far from step reference → not strict.
    far = "wait. " + ("x " * 300) + "step 2"
    assert not detects_strict(far, error_step=2)


def test_detects_strict_matches_specific_error_step_when_requested():
    # Refers to step 3, but error is at 2 → not strict.
    assert not detects_strict("Wait, step 3 looks wrong.", error_step=2)
    # Refers to step 2 specifically → strict.
    assert detects_strict("Wait, step 2 looks wrong.", error_step=2)


def test_detects_strict_with_no_error_step_arg_matches_any_step():
    assert detects_strict("Wait, step 7 is off.", error_step=None)


# --- Outcome classifier ---------------------------------------------------


def test_classify_outcome_all_six_buckets():
    assert classify_outcome(
        final_output="🪈", detected_loose=True,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "detect_recover"

    assert classify_outcome(
        final_output="🪈", detected_loose=False,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "silent_recovery"

    assert classify_outcome(
        final_output="🥟", detected_loose=True,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "detect_only"

    # Detected + landed on wrong_branch → still detect_only, not blind.
    assert classify_outcome(
        final_output="🪵", detected_loose=True,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "detect_only"

    assert classify_outcome(
        final_output="🪵", detected_loose=False,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "blind_wrong_branch"

    assert classify_outcome(
        final_output="🥟", detected_loose=False,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "off_rails"

    assert classify_outcome(
        final_output=None, detected_loose=False,
        ground_truth_final_output="🪈", wrong_branch_final_output="🪵",
    ) == "extraction_failed"


# --- score_prediction end-to-end ------------------------------------------


def _base_row(**overrides):
    row = {
        "example_id": "cont-000001",
        "difficulty": "easy",
        "chain_length_x": 4,
        "prefill_error_step": 2,
        "ground_truth_final_output": "🪈",
        "wrong_branch_final_output": "🪵",
        "raw_continuation_text": "",
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "mode": "prefill",
        "used_native_prefill": True,
    }
    row.update(overrides)
    return row


def test_score_prediction_blind_propagation():
    row = _base_row(raw_continuation_text=(
        "Step 3: (a ⊕ b) = c    [by ⊕ table]\n"
        "Step 4: c = 🪵    [by ⊕ table]\n\n"
        "Final Output: 🪵"
    ))
    scored = score_prediction(row)
    assert scored.final_output == "🪵"
    assert scored.detected_loose is False
    assert scored.outcome_bucket == "blind_wrong_branch"
    assert scored.matches_wrong_branch is True
    assert scored.matches_ground_truth is False


def test_score_prediction_detect_recover():
    row = _base_row(raw_continuation_text=(
        "Wait, step 2 should be 🪈, not 🥟. Let me recompute.\n"
        "Step 3: (🥟 ⊕ 🪈) = 🪈    [by ⊕ table]\n\n"
        "Final Output: 🪈"
    ))
    scored = score_prediction(row)
    assert scored.detected_loose is True
    assert scored.detected_strict is True
    assert scored.outcome_bucket == "detect_recover"
    assert scored.matches_ground_truth is True


def test_score_prediction_extraction_failed():
    row = _base_row(raw_continuation_text="Step 3: ... the model forgot the marker.")
    scored = score_prediction(row)
    assert scored.final_output is None
    assert scored.outcome_bucket == "extraction_failed"


def test_score_prediction_rejects_missing_fields():
    row = _base_row()
    row.pop("ground_truth_final_output")
    try:
        score_prediction(row)
    except ValueError as exc:
        assert "ground_truth_final_output" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing field")
