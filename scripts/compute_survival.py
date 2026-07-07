#!/usr/bin/env python3
"""Compute the normalized survival headline from a paired eval.

Joins a corrupted-prefill eval run with its clean-control counterpart (same
instances, clean prefill cut at the same step) and reports:

- ``survival_rate``: corrupted-condition accuracy / clean-condition accuracy,
  with a Katz 95% log-interval. This is the headline: "of the competence the
  model shows on clean partial derivations, what fraction survives a planted
  error?" It absorbs base task ability into the denominator so weak models
  aren't misread as bad self-correctors.
- ``conditional_recovery_rate``: P(correct under corruption | correct under
  clean control), the per-instance paired variant, with a Wilson interval.
  Its denominator is the model's clean-solved subset, so expect wide
  intervals for weak models — that is honest, not a bug.

Usage:
  python scripts/compute_survival.py \
      artifacts/evals/<model>-B-L0 \
      artifacts/evals/<model>-control-B-L0

Both directories must contain ``scores.jsonl`` (run score_continuation.py
first). Control rows are matched by stripping the ``-control`` suffix from
their example ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.eval.paths import SCORES_FILENAME
from emoji_bench.jsonl_io import load_jsonl_records
from emoji_bench.scoring.stats import katz_ratio_ci, wilson_interval


SURVIVAL_SUMMARY_FILENAME = "survival_summary.json"
CONTROL_ID_SUFFIX = "-control"


def _base_example_id(example_id: str) -> str:
    if example_id.endswith(CONTROL_ID_SUFFIX):
        return example_id[: -len(CONTROL_ID_SUFFIX)]
    return example_id


def _load_scores(eval_dir: Path) -> dict[str, dict[str, Any]]:
    scores_path = eval_dir / SCORES_FILENAME
    if not scores_path.exists():
        raise FileNotFoundError(
            f"missing {scores_path}; run score_continuation.py on {eval_dir} first"
        )
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_records(scores_path):
        rows[_base_example_id(row["example_id"])] = row
    return rows


def _cell(corrupted_correct: int, clean_correct: int, total: int) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "total": total,
        "corrupted_correct": corrupted_correct,
        "clean_correct": clean_correct,
        "corrupted_rate": round(corrupted_correct / total, 4) if total else 0.0,
        "clean_rate": round(clean_correct / total, 4) if total else 0.0,
    }
    if total and clean_correct:
        ratio = (corrupted_correct / total) / (clean_correct / total)
        cell["survival_rate"] = round(ratio, 4)
        lo, hi = katz_ratio_ci(corrupted_correct, total, clean_correct, total)
        cell["survival_ci95"] = None if lo is None else [lo, hi]
    else:
        cell["survival_rate"] = None
        cell["survival_ci95"] = None
    return cell


def compute_survival_summary(
    corrupted: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(corrupted) & set(control))
    if not shared:
        raise ValueError("no shared example ids between the two runs")

    corrupted_correct = clean_correct = 0
    cond_total = cond_correct = 0
    by_difficulty_counts: dict[str, dict[str, int]] = {}
    for example_id in shared:
        c_row = corrupted[example_id]
        k_row = control[example_id]
        difficulty = c_row["difficulty"]
        bucket = by_difficulty_counts.setdefault(
            difficulty, {"total": 0, "corrupted": 0, "clean": 0}
        )
        bucket["total"] += 1
        if c_row["matches_ground_truth"]:
            corrupted_correct += 1
            bucket["corrupted"] += 1
        if k_row["matches_ground_truth"]:
            clean_correct += 1
            bucket["clean"] += 1
            cond_total += 1
            if c_row["matches_ground_truth"]:
                cond_correct += 1

    summary = _cell(corrupted_correct, clean_correct, len(shared))
    summary["conditional_recovery_rate"] = (
        round(cond_correct / cond_total, 4) if cond_total else None
    )
    summary["conditional_recovery_ci95"] = (
        list(wilson_interval(cond_correct, cond_total)) if cond_total else None
    )
    summary["conditional_denominator"] = cond_total
    summary["by_difficulty"] = {
        d: _cell(counts["corrupted"], counts["clean"], counts["total"])
        for d, counts in sorted(by_difficulty_counts.items())
    }

    dropped_corrupted = len(corrupted) - len(shared)
    dropped_control = len(control) - len(shared)
    if dropped_corrupted or dropped_control:
        summary["unmatched_rows"] = {
            "corrupted_only": dropped_corrupted,
            "control_only": dropped_control,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corrupted_dir",
        type=Path,
        help="Eval dir for the error-injected condition (contains scores.jsonl).",
    )
    parser.add_argument(
        "control_dir",
        type=Path,
        help="Eval dir for the clean-control condition (contains scores.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the summary JSON "
            f"(default: <corrupted_dir>/{SURVIVAL_SUMMARY_FILENAME})."
        ),
    )
    args = parser.parse_args()

    corrupted = _load_scores(args.corrupted_dir)
    control = _load_scores(args.control_dir)
    summary = compute_survival_summary(corrupted, control)
    summary["corrupted_dir"] = str(args.corrupted_dir)
    summary["control_dir"] = str(args.control_dir)

    output_path = args.output or (args.corrupted_dir / SURVIVAL_SUMMARY_FILENAME)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
