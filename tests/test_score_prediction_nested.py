"""Tests for the judge-backed headline scorer.

The headline now tracks two independent outcomes:

1. ``error_recovered`` from the LLM judge
2. ``final_answer_correct`` from the extracted ``Final Output:``
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
    error_recovered: bool
    reasoning: str = "synthetic"


def _row(**overrides):
    base = {
        "example_id": "cont-test-000",
        "difficulty": "easy",
        "chain_length_x": 4,
        "prefill_error_step": 2,
        "ground_truth_final_output": "🪈",
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "mode": "prefill",
        "turn_2_level": 0,
    }
    base.update(overrides)
    return base


def test_both_metrics_true_when_recovered_and_final_correct():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(error_recovered=True),
        final_output="🪈",
    )
    assert scored.error_recovered is True
    assert scored.final_answer_correct is True


def test_error_recovery_and_final_answer_are_independent():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(error_recovered=True),
        final_output="🪵",
    )
    assert scored.error_recovered is True
    assert scored.final_answer_correct is False


def test_final_answer_can_be_correct_without_recovery():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(error_recovered=False),
        final_output="🪈",
    )
    assert scored.error_recovered is False
    assert scored.final_answer_correct is True


def test_missing_final_output_marks_final_answer_incorrect():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(error_recovered=False),
        final_output=None,
    )
    assert scored.final_output is None
    assert scored.final_answer_correct is False


def test_to_dict_round_trip_keys():
    scored = score_prediction_nested(
        prediction_row=_row(),
        judge_verdict=_FakeVerdict(error_recovered=True),
        final_output="🪈",
    )
    d = scored.to_dict()
    assert d["error_recovered"] is True
    assert d["final_answer_correct"] is True
    assert d["judge_reasoning"] == "synthetic"
    assert d["final_output"] == "🪈"
    assert d["model"] == "claude-haiku-4-5"


def _make_scored(
    *,
    difficulty: str,
    recovered: bool,
    final_correct: bool,
) -> NestedScoredContinuation:
    return score_prediction_nested(
        prediction_row=_row(difficulty=difficulty),
        judge_verdict=_FakeVerdict(error_recovered=recovered),
        final_output="🪈" if final_correct else "🪵",
    )


def test_summarize_nested_overall_rates():
    scored = [
        _make_scored(difficulty="easy", recovered=True, final_correct=True),
        _make_scored(difficulty="easy", recovered=True, final_correct=False),
        _make_scored(difficulty="easy", recovered=False, final_correct=True),
        _make_scored(difficulty="easy", recovered=False, final_correct=False),
    ]
    summary = summarize_nested(scored)
    assert summary["total"] == 4
    assert summary["error_recovery_rate"] == 0.5
    assert summary["final_answer_correct_rate"] == 0.5


def test_summarize_nested_per_difficulty_breakdown():
    scored = [
        _make_scored(difficulty="easy", recovered=True, final_correct=True),
        _make_scored(difficulty="hard", recovered=False, final_correct=False),
    ]
    summary = summarize_nested(scored)
    assert "easy" in summary["by_difficulty"]
    assert "hard" in summary["by_difficulty"]
    assert summary["by_difficulty"]["easy"]["error_recovery_rate"] == 1.0
    assert summary["by_difficulty"]["hard"]["final_answer_correct_rate"] == 0.0


def test_summarize_nested_handles_empty_input():
    summary = summarize_nested([])
    assert summary["total"] == 0
    assert summary["error_recovery_rate"] == 0.0
    assert summary["final_answer_correct_rate"] == 0.0
    assert summary["by_difficulty"] == {}
