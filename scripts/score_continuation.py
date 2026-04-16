#!/usr/bin/env python3
"""Score an E-CONTINUE predictions.jsonl and emit a summary.

Reads ``predictions.jsonl`` produced by ``evaluate_continuation.py``,
runs the regex-based scorer on every row, optionally folds in the LLM
judge verdicts (``judge.jsonl``) plus the Python derivation validator,
and writes ``scores.jsonl`` + ``score_summary.json`` next to the inputs.

When ``judge.jsonl`` exists alongside the predictions:
    * ``nested_scores.jsonl`` is also written, with one row per prediction
      carrying the three nested booleans (detected, detected_and_fixed,
      detected_fixed_and_right) plus the underlying judge + validator
      payloads for downstream analysis.
    * The script validates that ``judge.jsonl`` covers every prediction
      exactly once and that each row's ``prediction_fingerprint`` matches the
      current ``predictions.jsonl``. Any stale, duplicate, or partial judge
      artifact is rejected instead of being scored silently.
    * The summary's headline ``rates`` block becomes the three nested rates:
      ``detect_rate``, ``detect_correct_rate``,
      ``detect_correct_finaloutput_correct_rate``.
    * The legacy regex bucket counts and rates remain in the summary as a
      ``regex_baseline`` section for diagnostic comparison.

When ``judge.jsonl`` does NOT exist, the script falls back to the original
regex-only behavior and prints a note that nested metrics are unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.domain.continuation_validator import validate_derivation
from emoji_bench.domain.formatter import system_from_json
from emoji_bench.judge.artifacts import (
    build_prediction_fingerprint_map,
    ensure_full_judge_coverage,
    load_validated_judge_rows,
)
from emoji_bench.judge.continuation_scorer import (
    OUTCOME_BUCKETS,
    NestedScoredContinuation,
    ScoredContinuation,
    extract_final_output,
    score_prediction,
    score_prediction_nested,
    summarize_nested,
)
from emoji_bench.jsonl_io import load_jsonl_records
from emoji_bench.eval.paths import (
    build_score_artifact_paths,
    resolve_dataset_path as _resolve_dataset_path,
    resolve_predictions_path as _resolve_predictions_path,
)


# --- Regex baseline summary ----------------------------------------------


def _regex_summary(scored: list[ScoredContinuation]) -> dict[str, Any]:
    total = len(scored)
    overall_buckets = Counter(s.outcome_bucket for s in scored)
    by_difficulty: dict[str, dict[str, int]] = {}
    for s in scored:
        bucket = by_difficulty.setdefault(
            s.difficulty,
            {b: 0 for b in OUTCOME_BUCKETS} | {"_total": 0},
        )
        bucket[s.outcome_bucket] += 1
        bucket["_total"] += 1

    detected_loose = sum(1 for s in scored if s.detected_loose)
    detected_strict = sum(1 for s in scored if s.detected_strict)
    recovered = sum(1 for s in scored if s.matches_ground_truth)
    cascaded = sum(1 for s in scored if s.matches_wrong_branch)
    extraction_ok = sum(1 for s in scored if s.final_output is not None)

    return {
        "total": total,
        "outcome_buckets": {b: overall_buckets.get(b, 0) for b in OUTCOME_BUCKETS},
        "rates": {
            "self_detection_loose": _rate(detected_loose, total),
            "self_detection_strict": _rate(detected_strict, total),
            "final_answer_recovery": _rate(recovered, total),
            "blind_cascade": _rate(cascaded, total),
            "extraction_ok": _rate(extraction_ok, total),
        },
        "by_difficulty": by_difficulty,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# --- Nested-metric scoring -----------------------------------------------


class _ValidationResultLite:
    """Tiny adapter so score_prediction_nested can accept missing-row defaults.

    Used only when a judge row exists for an example_id but the dataset can't
    be located — the nested score for that row should still record the judge
    booleans even if the validator couldn't run.
    """
    parseable = False
    derivation_valid = False
    terminal_matches_gt = False
    first_invalid_step = None
    first_discontinuity_step = None
    parsed_step_count = 0
    reason = "validator_skipped: dataset row not found"


class _JudgeVerdictLite:
    def __init__(self, detected_error: bool, corrected_step_y: bool, reasoning: str):
        self.detected_error = detected_error
        self.corrected_step_y = corrected_step_y
        self.reasoning = reasoning


def _build_nested_scores(
    *,
    predictions: list[dict[str, Any]],
    dataset_rows: dict[str, dict[str, Any]] | None,
    judge_rows: dict[str, dict[str, Any]],
) -> list[NestedScoredContinuation]:
    nested: list[NestedScoredContinuation] = []
    for pred in predictions:
        eid = pred["example_id"]
        judge_row = judge_rows.get(eid)
        if judge_row is None:
            raise RuntimeError(
                f"nested scoring invariant violated: missing judge row for {eid!r}"
            )
        verdict = _JudgeVerdictLite(
            detected_error=bool(judge_row["detected_error"]),
            corrected_step_y=bool(judge_row["corrected_step_y"]),
            reasoning=str(judge_row.get("reasoning", "")),
        )
        ds_row = (dataset_rows or {}).get(eid)
        if ds_row is None or "system_json" not in ds_row:
            validation = _ValidationResultLite()
        else:
            try:
                system = system_from_json(ds_row["system_json"])
                validation = validate_derivation(
                    pred["raw_continuation_text"],
                    system,
                    pred["ground_truth_final_output"],
                )
            except Exception as exc:  # noqa: BLE001 — validator robustness
                validation = type(
                    "ValidationFailure",
                    (),
                    {
                        "parseable": False,
                        "derivation_valid": False,
                        "terminal_matches_gt": False,
                        "first_invalid_step": None,
                        "first_discontinuity_step": None,
                        "parsed_step_count": 0,
                        "reason": f"validator_raised: {exc.__class__.__name__}",
                    },
                )()
        nested.append(
            score_prediction_nested(
                prediction_row=pred,
                judge_verdict=verdict,
                validation_result=validation,
                final_output=extract_final_output(pred["raw_continuation_text"]),
            )
        )
    return nested


# --- Main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a predictions.jsonl from evaluate_continuation.py. Emits a "
            "regex-baseline summary plus, if judge.jsonl is present, a "
            "nested judge+validator headline."
        ),
    )
    parser.add_argument(
        "predictions_path",
        help="Path to predictions.jsonl or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for scores files (default: same dir as predictions).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Path to the source dataset JSONL or directory. Defaults to the "
            "input_path recorded in summary.json. Required for nested-metric "
            "validation when judge.jsonl is present."
        ),
    )
    args = parser.parse_args()

    predictions_path = _resolve_predictions_path(args.predictions_path)
    if not predictions_path.exists():
        parser.error(f"predictions file not found: {predictions_path}")

    artifact_paths = build_score_artifact_paths(predictions_path, output_dir=args.output_dir)
    artifact_paths.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_jsonl_records(predictions_path)
    prediction_fingerprints = build_prediction_fingerprint_map(predictions)
    scored = [score_prediction(row) for row in predictions]

    with artifact_paths.scores_path.open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    regex_summary = _regex_summary(scored)
    summary: dict[str, Any] = {
        "predictions_path": str(predictions_path.resolve()),
        "scores_path": str(artifact_paths.scores_path.resolve()),
    }

    if artifact_paths.judge_path.exists():
        dataset_path = _resolve_dataset_path(
            explicit=args.dataset_path,
            summary_path=artifact_paths.summary_path,
        )
        dataset_rows = (
            {row["example_id"]: row for row in load_jsonl_records(dataset_path)}
            if dataset_path is not None
            else None
        )
        judge_rows = load_validated_judge_rows(
            artifact_paths.judge_path,
            expected_fingerprints=prediction_fingerprints,
        )
        ensure_full_judge_coverage(
            prediction_ids=set(prediction_fingerprints),
            judge_rows=judge_rows,
        )

        nested_scored = _build_nested_scores(
            predictions=predictions,
            dataset_rows=dataset_rows,
            judge_rows=judge_rows,
        )
        with artifact_paths.nested_scores_path.open("w", encoding="utf-8") as fh:
            for ns in nested_scored:
                fh.write(json.dumps(ns.to_dict(), ensure_ascii=False) + "\n")

        summary["headline"] = summarize_nested(nested_scored)
        summary["headline_kind"] = "judge_plus_validator"
        summary["nested_scores_path"] = str(artifact_paths.nested_scores_path.resolve())
        summary["judge_path"] = str(artifact_paths.judge_path.resolve())
        summary["judged_count"] = len(judge_rows)
        summary["judged_coverage"] = 1.0
        if dataset_path is None:
            summary["validator_status"] = (
                "skipped — dataset path could not be located; "
                "metric (3) is False for every row in the nested summary"
            )
        else:
            summary["validator_status"] = "ok"
            summary["dataset_path"] = str(dataset_path.resolve())
        summary["regex_baseline"] = regex_summary
    else:
        summary["headline"] = regex_summary
        summary["headline_kind"] = "regex_only"
        summary["note"] = (
            "judge.jsonl not found alongside predictions; nested metrics are "
            "unavailable. Run scripts/judge_continuation.py first to enable them."
        )

    artifact_paths.score_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
