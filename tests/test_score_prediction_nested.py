"""Tests for the nested-metric combinator (Phase 5b headline scoring).

The combinator turns (judge_verdict + validation_result + final_output) into
three nested booleans. These tests use synthetic verdicts and validation
results — no real LLM, no real interpreter — to verify the boolean algebra
is exactly right.
"""
from __future__ import annotations

from dataclasses import dataclass

from emoji_bench.judge.continuation_scorer import (
    NestedScoredContinuation,
    score_prediction_nested,
    summarize_nested,
)


@dataclass(frozen=True)
class _FakeVerdict:
    detected_error: bool
    corrected_step_y: bool
    reasoning: str = "synthetic"


@dataclass(frozen=True)
class _FakeValidation:
    parseable: bool = True
    derivation_valid: bool = True
    terminal_matches_gt: bool = True
    first_invalid_step: int | None = None
    first_discontinuity_step: int | None = None
    parsed_step_count: int = 2
    reason: str | None = None


def _row(**overrides):
    base = {
        "example_id": "cont-test-000",
        "difficulty": "easy",
        "chain_length_x": 4,
        "prefill_error_step": 2,
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "mode": "prefill",
        "turn_2_level": 0,
    }
    base.update(overrides)
    return base


# --- Boolean algebra ------------------------------------------------------


def test_all_three_metrics_true_when_all_signals_positive():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=True),
        validation_result=_FakeValidation(),
        final_output="🪈",
    )
    assert scored.detected is True
    assert scored.detected_and_fixed is True
    assert scored.detected_fixed_and_right is True


def test_detect_only_does_not_propagate_to_fix_or_right():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=False),
        validation_result=_FakeValidation(),
        final_output="🪈",
    )
    assert scored.detected is True
    assert scored.detected_and_fixed is False
    assert scored.detected_fixed_and_right is False


def test_detect_and_fix_but_invalid_derivation_blocks_metric_3():
    """The compensating-error case the user explicitly called out."""
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=True),
        validation_result=_FakeValidation(
            derivation_valid=False,
            terminal_matches_gt=False,  # validator zeroes this when invalid
            first_invalid_step=3,
            reason="step 3 has wrong reduction",
        ),
        final_output="🪈",  # luck: final happens to match gt
    )
    assert scored.detected is True
    assert scored.detected_and_fixed is True
    # Invalid derivation -> metric (3) False even though final matches.
    assert scored.detected_fixed_and_right is False


def test_detect_and_fix_but_terminal_wrong_blocks_metric_3():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=True),
        validation_result=_FakeValidation(
            derivation_valid=True,
            terminal_matches_gt=False,
        ),
        final_output="🪵",
    )
    assert scored.detected_and_fixed is True
    assert scored.detected_fixed_and_right is False


def test_unparseable_continuation_fails_metric_3():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=True),
        validation_result=_FakeValidation(
            parseable=False,
            derivation_valid=False,
            terminal_matches_gt=False,
            parsed_step_count=0,
            reason="unparseable",
        ),
        final_output=None,
    )
    assert scored.detected is True
    assert scored.detected_and_fixed is True
    assert scored.detected_fixed_and_right is False


def test_no_detection_zeroes_all_three():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=False, corrected_step_y=False),
        validation_result=_FakeValidation(),
        final_output="🪈",
    )
    assert scored.detected is False
    assert scored.detected_and_fixed is False
    assert scored.detected_fixed_and_right is False


# --- Round-trip serialization --------------------------------------------


def test_to_dict_round_trip_keys():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(detected_error=True, corrected_step_y=True),
        validation_result=_FakeValidation(),
        final_output="🪈",
    )
    d = scored.to_dict()
    assert d["detected"] is True
    assert d["detected_and_fixed"] is True
    assert d["detected_fixed_and_right"] is True
    assert d["judge_reasoning"] == "synthetic"
    assert d["validator_parsed_step_count"] == 2
    assert d["final_output"] == "🪈"
    assert d["model"] == "claude-haiku-4-5"


# --- summarize_nested -----------------------------------------------------


def _make_scored(
    *,
    difficulty: str,
    detected: bool,
    fixed: bool,
    right: bool,
) -> NestedScoredContinuation:
    return score_prediction_nested(
        prediction_row=_row(difficulty=difficulty),
        judge_verdict=_FakeVerdict(
            detected_error=detected, corrected_step_y=fixed,
        ),
        validation_result=_FakeValidation(
            derivation_valid=right,
            terminal_matches_gt=right,
        ),
        final_output="🪈" if right else "🪵",
    )


def test_summarize_nested_overall_rates():
    scored = [
        _make_scored(difficulty="easy", detected=True, fixed=True, right=True),
        _make_scored(difficulty="easy", detected=True, fixed=True, right=False),
        _make_scored(difficulty="easy", detected=True, fixed=False, right=False),
        _make_scored(difficulty="easy", detected=False, fixed=False, right=False),
    ]
    summary = summarize_nested(scored)
    assert summary["total"] == 4
    assert summary["detect_rate"] == 0.75
    assert summary["detect_correct_rate"] == 0.5
    assert summary["detect_correct_finaloutput_correct_rate"] == 0.25


def test_summarize_nested_per_difficulty_breakdown():
    scored = [
        _make_scored(difficulty="easy", detected=True, fixed=True, right=True),
        _make_scored(difficulty="hard", detected=False, fixed=False, right=False),
    ]
    summary = summarize_nested(scored)
    assert "easy" in summary["by_difficulty"]
    assert "hard" in summary["by_difficulty"]
    assert summary["by_difficulty"]["easy"]["detect_rate"] == 1.0
    assert summary["by_difficulty"]["hard"]["detect_rate"] == 0.0


def test_summarize_nested_handles_empty_input():
    summary = summarize_nested([])
    assert summary["total"] == 0
    assert summary["detect_rate"] == 0.0
    assert summary["by_difficulty"] == {}
