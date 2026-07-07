#!/usr/bin/env python3
"""Score an E-CONTINUE predictions.jsonl and emit a summary.

Reads ``predictions.jsonl`` produced by ``evaluate_continuation.py``,
runs deterministic final-output and regex-diagnostic scoring on every row,
and writes ``scores.jsonl`` + ``score_summary.json`` next to the inputs.
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

from emoji_bench.scoring.continuation_scorer import (
    OUTCOME_BUCKETS,
    ScoredContinuation,
    score_prediction,
    summarize_behavior,
    summarize_final_answer_only,
)
from emoji_bench.scoring.stats import wilson_interval
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

    counts = {
        "self_detection_loose": detected_loose,
        "self_detection_strict": detected_strict,
        "final_answer_recovery": recovered,
        "blind_cascade": cascaded,
        "extraction_ok": extraction_ok,
    }
    return {
        "total": total,
        "outcome_buckets": {b: overall_buckets.get(b, 0) for b in OUTCOME_BUCKETS},
        "rates": {name: _rate(count, total) for name, count in counts.items()},
        "rates_ci95": {
            name: list(wilson_interval(count, total)) for name, count in counts.items()
        },
        "by_difficulty": by_difficulty,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# --- Main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a predictions.jsonl from evaluate_continuation.py. Emits a "
            "final-answer headline and regex-baseline diagnostics."
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
            "Source dataset (jsonl or directory containing test.jsonl) used "
            "for AST-grounded behavior classification and derivation "
            "validation. Defaults to the input_path recorded in the eval's "
            "summary.json; when neither resolves, only regex scoring runs."
        ),
    )
    args = parser.parse_args()

    predictions_path = _resolve_predictions_path(args.predictions_path)
    if not predictions_path.exists():
        parser.error(f"predictions file not found: {predictions_path}")

    artifact_paths = build_score_artifact_paths(predictions_path, output_dir=args.output_dir)
    artifact_paths.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = _resolve_dataset_path(
        explicit=args.dataset_path,
        summary_path=artifact_paths.summary_path,
    )
    dataset_by_id: dict[str, dict[str, Any]] = {}
    if dataset_path is not None and dataset_path.exists():
        dataset_by_id = {
            row["example_id"]: row for row in load_jsonl_records(dataset_path)
        }
    else:
        print(
            "warning: source dataset not found; scoring without behavior "
            "classification (pass --dataset-path to enable it)",
            file=sys.stderr,
        )

    predictions = load_jsonl_records(predictions_path)
    scored = [
        score_prediction(row, dataset_row=dataset_by_id.get(row["example_id"]))
        for row in predictions
    ]

    with artifact_paths.scores_path.open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    regex_summary = _regex_summary(scored)
    summary: dict[str, Any] = {
        "predictions_path": str(predictions_path),
        "scores_path": str(artifact_paths.scores_path),
        "headline": summarize_final_answer_only(scored),
        "headline_kind": "final_output_only",
        "behavior_mix": summarize_behavior(scored),
        "regex_baseline": regex_summary,
    }

    artifact_paths.score_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
