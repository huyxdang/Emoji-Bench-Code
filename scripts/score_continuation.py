#!/usr/bin/env python3
"""Score an E-CONTINUE predictions.jsonl and emit a summary.

Reads ``predictions.jsonl`` produced by ``evaluate_continuation.py``,
runs ``score_prediction`` on every row, writes ``scores.jsonl`` with the
per-row classification, and prints an aggregate summary grouped by
difficulty plus an overall roll-up.

Scoring is separated from inference so a predictions file can be
rescored as the detection rules evolve, without re-spending API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_scorer import (
    OUTCOME_BUCKETS,
    ScoredContinuation,
    score_prediction,
)
from emoji_bench.evaluation import load_jsonl_records


def _resolve_predictions_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / "predictions.jsonl"
    return path


def _summary(scored: list[ScoredContinuation]) -> dict[str, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a predictions.jsonl from evaluate_continuation.py and "
            "write scores.jsonl + a summary JSON."
        ),
    )
    parser.add_argument(
        "predictions_path",
        help="Path to predictions.jsonl or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for scores.jsonl + summary.json (default: same dir as predictions).",
    )
    args = parser.parse_args()

    predictions_path = _resolve_predictions_path(args.predictions_path)
    if not predictions_path.exists():
        parser.error(f"predictions file not found: {predictions_path}")

    rows = load_jsonl_records(predictions_path)
    scored = [score_prediction(row) for row in rows]

    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.jsonl"
    summary_path = output_dir / "score_summary.json"

    with scores_path.open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    summary = _summary(scored)
    summary["predictions_path"] = str(predictions_path.resolve())
    summary["scores_path"] = str(scores_path.resolve())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
