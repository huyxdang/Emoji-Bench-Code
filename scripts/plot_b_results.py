"""Bar-chart visualizer for the B-variant eval results.

Produces two PNGs:
  * L0: "Performance when prompted w/out hint"
  * L1: "Performance when prompted with hint"

Each chart puts models on the x-axis with three grouped bars per model:
  * "error detection %"                        (detect_rate)
  * "correct error detection %"               (detect_correct_rate)
  * "correct error detection + final output %" (detect_correct_finaloutput_correct_rate)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "artifacts" / "evals"

LEVELS = {
    0: "Performance (w/out hint)",
    1: "Performance (with hint)",
}

METRICS = [
    ("detect_rate", "error detection %"),
    ("detect_correct_rate", "correct error detection %"),
    ("detect_correct_finaloutput_correct_rate", "correct error detection + final output %"),
]


UNCAPPED_MODELS = {
    "claude-opus-4-7-reasoning-max",
    "claude-sonnet-4-6-reasoning-max",
    "gemini-3-flash-preview-thinking-high",
    "gemini-3.1-pro-preview-thinking-high",
    "gpt-5.4-mini-reasoning-xhigh",
    "gpt-5.4-reasoning-xhigh",
}

EXCLUDED_MODELS: set[str] = set()


def collect_b_results(level: int, evals_dir: Path) -> list[tuple[str, dict[str, float]]]:
    suffix = f"-B-L{level}"
    results: list[tuple[str, dict[str, float]]] = []
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
        display_name = model_name if model_name in UNCAPPED_MODELS else f"{model_name} (4096)"
        results.append((display_name, headline))
    return results


def plot_level(
    level: int,
    results: list[tuple[str, dict[str, float]]],
    output_path: Path,
) -> None:
    if not results:
        print(f"no scored cells for L{level}", file=sys.stderr)
        return

    models = [name for name, _ in results]
    series: dict[str, list[float]] = {label: [] for _, label in METRICS}
    for _, headline in results:
        for key, label in METRICS:
            value = headline.get(key)
            series[label].append(float(value) * 100 if value is not None else 0.0)

    fig, ax = plt.subplots(figsize=(max(8, 2.0 * len(models) + 2), 6))
    bar_width = 0.26
    x_positions = range(len(models))
    offsets = [-bar_width, 0, bar_width]
    colors = ["#72B7B2", "#4C78A8", "#F58518"]

    for (label, values), offset, color in zip(series.items(), offsets, colors):
        xs = [x + offset for x in x_positions]
        bars = ax.bar(xs, values, width=bar_width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(LEVELS[level])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(models, rotation=25, ha="right")
    for tick_label, name in zip(ax.get_xticklabels(), models):
        if "(4096)" in name:
            tick_label.set_color("#4C78A8")
    ax.legend(loc="upper right", frameon=False)
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
        output_path = args.output_dir / f"b_level{level}.png"
        plot_level(level, results, output_path)


if __name__ == "__main__":
    main()
