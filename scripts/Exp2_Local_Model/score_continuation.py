#!/usr/bin/env python3
"""Score E-CONTINUE predictions produced with the \\boxed{} output format.

Identical to scripts/score_continuation.py except that final-output extraction
looks for ``\\boxed{<symbol>}`` (the instruction used in the boxed-ver datasets)
instead of the plain ``Final Output: <symbol>`` line.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emoji_bench.scoring.continuation_scorer import (
    OUTCOME_BUCKETS,
    ScoredContinuation,
    classify_outcome,
    detects_loose,
    detects_strict,
    summarize_final_answer_only,
)
from emoji_bench.jsonl_io import load_jsonl_records
from emoji_bench.eval.paths import (
    build_score_artifact_paths,
    resolve_predictions_path as _resolve_predictions_path,
)

# ---------------------------------------------------------------------------
# Boxed-format extractor
# ---------------------------------------------------------------------------


def _extract_boxed(text: str) -> str:
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else ""


def extract_boxed_output(text: str) -> str | None:
    """Return the symbol inside ``\\boxed{...}``, or None if absent."""
    raw = _extract_boxed(text).strip()
    return raw if raw else None


# ---------------------------------------------------------------------------
# Scoring entry point (replaces score_prediction from the shared module)
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: tuple[str, ...] = (
    "example_id",
    "difficulty",
    "chain_length_x",
    "prefill_error_step",
    "ground_truth_final_output",
    "wrong_branch_final_output",
    "raw_continuation_text",
    "model",
    "provider",
    "mode",
)


def score_prediction_boxed(row: dict[str, Any]) -> ScoredContinuation:
    missing = [f for f in _REQUIRED_FIELDS if f not in row]
    if missing:
        raise ValueError(f"prediction row {row.get('example_id')!r} missing fields: {missing}")

    text = row["raw_continuation_text"]
    final = extract_boxed_output(text)
    detected_loose_flag = detects_loose(text)
    detected_strict_flag = detects_strict(text, error_step=row["prefill_error_step"])
    gt = row["ground_truth_final_output"]
    wb = row["wrong_branch_final_output"]
    bucket = classify_outcome(
        final_output=final,
        detected_loose=detected_loose_flag,
        ground_truth_final_output=gt,
        wrong_branch_final_output=wb,
    )
    return ScoredContinuation(
        example_id=row["example_id"],
        difficulty=row["difficulty"],
        chain_length_x=row["chain_length_x"],
        prefill_error_step=row["prefill_error_step"],
        ground_truth_final_output=gt,
        wrong_branch_final_output=wb,
        final_output=final,
        detected_loose=detected_loose_flag,
        detected_strict=detected_strict_flag,
        outcome_bucket=bucket,
        matches_ground_truth=(final == gt),
        matches_wrong_branch=(final == wb),
        model=row["model"],
        provider=row["provider"],
        mode=row["mode"],
        raw_continuation_text=text,
    )


# ---------------------------------------------------------------------------
# Summary helpers (identical to scripts/score_continuation.py)
# ---------------------------------------------------------------------------


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


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_path",
        help="Path to predictions.jsonl or a directory containing it.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    predictions_path = _resolve_predictions_path(args.predictions_path)
    if not predictions_path.exists():
        parser.error(f"predictions file not found: {predictions_path}")

    artifact_paths = build_score_artifact_paths(predictions_path, output_dir=args.output_dir)
    artifact_paths.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_jsonl_records(predictions_path)
    scored = [score_prediction_boxed(row) for row in predictions]

    with artifact_paths.scores_path.open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "predictions_path": str(predictions_path),
        "scores_path": str(artifact_paths.scores_path),
        "headline": summarize_final_answer_only(scored),
        "headline_kind": "final_output_only",
        "regex_baseline": _regex_summary(scored),
    }
    artifact_paths.score_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
