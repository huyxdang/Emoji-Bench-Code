"""Refresh judge-backed headline metrics in existing score_summary.json files.

Walks every directory under artifacts/evals/ that contains both
`nested_scores.jsonl` and `score_summary.json`, rehydrates the nested scores
into NestedScoredContinuation rows, recomputes the headline via
summarize_nested, and writes the patched score_summary.json back in place.

The filename is a legacy holdover from the earlier mechanical-correctness
iteration of the benchmark.

Usage:
  python scripts/backfill_mechanical_correct.py                 # all dirs under artifacts/evals
  python scripts/backfill_mechanical_correct.py <dir> [<dir>..] # specific dirs
  python scripts/backfill_mechanical_correct.py --dry-run       # print changes, don't write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emoji_bench.judge.continuation_scorer import (  # noqa: E402
    NestedScoredContinuation,
    summarize_nested,
)

EVALS_DIR = REPO_ROOT / "artifacts" / "evals"


def _load_nested(path: Path) -> list[NestedScoredContinuation]:
    rows: list[NestedScoredContinuation] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append(
            NestedScoredContinuation(
                example_id=d["example_id"],
                difficulty=d["difficulty"],
                chain_length_x=d["chain_length_x"],
                prefill_error_step=d["prefill_error_step"],
                error_recovered=d["error_recovered"],
                judge_reasoning=d.get("judge_reasoning", ""),
                final_output=d.get("final_output"),
                final_answer_correct=d["final_answer_correct"],
                model=d.get("model", ""),
                provider=d.get("provider", ""),
                mode=d.get("mode", ""),
                turn_2_level=d.get("turn_2_level"),
            )
        )
    return rows


def _backfill_one(eval_dir: Path, *, dry_run: bool) -> tuple[str, float] | None:
    nested_path = eval_dir / "nested_scores.jsonl"
    summary_path = eval_dir / "score_summary.json"
    if not nested_path.exists() or not summary_path.exists():
        return None

    rows = _load_nested(nested_path)
    headline = summarize_nested(rows)
    summary = json.loads(summary_path.read_text())

    # Only rewrite judge-backed summaries; final-output-only and regex-only
    # outputs do not have a judge-backed headline to refresh.
    if summary.get("headline_kind") != "judge_plus_final_output":
        return None

    summary["headline"] = headline

    if not dry_run:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return (eval_dir.name, headline.get("final_answer_correct_rate", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="*", type=Path,
                        help="Specific eval dirs; default: all under artifacts/evals/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    args = parser.parse_args()

    if args.dirs:
        targets = [d if d.is_absolute() else REPO_ROOT / d for d in args.dirs]
    else:
        targets = sorted(p for p in EVALS_DIR.iterdir() if p.is_dir())

    results: list[tuple[str, float]] = []
    skipped: list[str] = []
    for t in targets:
        out = _backfill_one(t, dry_run=args.dry_run)
        if out is None:
            skipped.append(t.name)
        else:
            results.append(out)

    results.sort(key=lambda kv: -kv[1])
    verb = "would refresh" if args.dry_run else "refreshed"
    print(f"{verb} headline metrics for {len(results)} dirs:")
    for name, rate in results:
        print(f"  {rate:6.2%}  {name}")
    if skipped:
        print(
            f"\nskipped {len(skipped)} dirs "
            f"(no nested score_summary or not judge_plus_final_output):"
        )
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
