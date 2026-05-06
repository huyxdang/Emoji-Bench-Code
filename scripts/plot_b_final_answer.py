"""Bar-chart visualizer for B-variant final-answer correctness.

Produces two PNGs in artifacts/plots/:
  * b_final_answer_l0.png — Final answer correct rate (without hint)
  * b_final_answer_l1.png — Final answer correct rate (with hint)

Reads ``final_answer_correct_rate`` from each cell's ``score_summary.json``
under ``headline``. Cells without that field (e.g. fully empty) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "artifacts" / "evals"

LEVEL_TITLES = {
    0: "Final-answer correctness (without hint)",
    1: "Final-answer correctness (with hint)",
}

EXCLUDED_MODELS: set[str] = set()


def collect_b_results(level: int, evals_dir: Path) -> list[tuple[str, float]]:
    suffix = f"-B-L{level}"
    results: list[tuple[str, float]] = []
    for cell_dir in sorted(evals_dir.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.endswith(suffix):
            continue
        if "4096" in cell_dir.name:
            continue
        model_name = cell_dir.name[: -len(suffix)]
        if model_name in EXCLUDED_MODELS:
            continue
        summary_path = cell_dir / "score_summary.json"
        if not summary_path.exists():
            print(f"skip {cell_dir.name}: no score_summary.json", file=sys.stderr)
            continue
        with summary_path.open() as fh:
            summary = json.load(fh)
        headline = summary.get("headline") or {}
        rate = headline.get("final_answer_correct_rate")
        if rate is None:
            print(
                f"skip {cell_dir.name}: no final_answer_correct_rate in headline",
                file=sys.stderr,
            )
            continue
        results.append((model_name, float(rate)))
        # Sort from highest correct rate to lowest
        results.sort(key=lambda x: x[1], reverse=True)
    return results


def plot_level(
    level: int,
    results: list[tuple[str, float]],
    output_path: Path,
) -> None:
    if not results:
        print(f"no scored cells for B-L{level}", file=sys.stderr)
        return

    models = [name for name, _ in results]
    values = [rate * 100 for _, rate in results]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models) + 2), 6))
    bars = ax.bar(range(len(models)), values, color="#4C78A8")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(LEVEL_TITLES[level])
    ax.set_ylabel("Final answer correct (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=EVALS_DIR,
        help=f"Directory containing <model>-B-L{{0,1}} eval cells (default: {EVALS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "plots",
        help="Directory to write the two PNGs into.",
    )
    args = parser.parse_args()

    for level in (0, 1):
        results = collect_b_results(level, args.evals_dir)
        output_path = args.output_dir / f"b_final_answer_l{level}.png"
        plot_level(level, results, output_path)


if __name__ == "__main__":
    main()
