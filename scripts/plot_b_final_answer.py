"""Bar-chart visualizer for B-variant final-answer correctness.

Produces PNGs in artifacts/plots/:
  * b_final_answer_l0.png — Final answer correct rate (without hint), 95% CI
  * b_final_answer_l1.png — Final answer correct rate (with hint), 95% CI
  * b_behavior_mix_l0.png — Behavior-mode mix per model (without hint)
  * b_behavior_mix_l1.png — Behavior-mode mix per model (with hint)

Reads ``final_answer_correct_rate`` (plus ``final_answer_correct_ci95`` and
``behavior_mix`` when present) from each cell's ``score_summary.json``.
Cells without the needed fields are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "artifacts" / "evals"

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from emoji_bench.scoring.behavior_classifier import BEHAVIOR_MODES
from emoji_bench.scoring.stats import wilson_interval

LEVEL_TITLES = {
    0: "Final-answer correctness (without hint)",
    1: "Final-answer correctness (with hint)",
}

BEHAVIOR_TITLES = {
    0: "Recovery behavior mix (without hint)",
    1: "Recovery behavior mix (with hint)",
}

EXCLUDED_MODELS: set[str] = set()

BAR_COLOR = "#2a78d6"

# Fixed color per behavior mode (color follows the entity, never its rank).
# Recovery modes get the cool identity hues; the blind cascade gets red.
BEHAVIOR_COLORS: dict[str, str] = {
    "inplace_correction": "#2a78d6",
    "continue_from_corrected": "#1baf7a",
    "full_restart": "#4a3aa7",
    "continue_from_corrupted": "#e34948",
    "other": "#eda100",
    "no_steps": "#9a998e",
}

BEHAVIOR_LABELS: dict[str, str] = {
    "inplace_correction": "In-place correction",
    "continue_from_corrected": "Continue from corrected state",
    "full_restart": "Full restart",
    "continue_from_corrupted": "Continue from corrupted state",
    "other": "Other",
    "no_steps": "No parseable steps",
}

# Stack order: recovery behaviors from the baseline, cascade on top.
BEHAVIOR_STACK_ORDER: tuple[str, ...] = (
    "inplace_correction",
    "continue_from_corrected",
    "full_restart",
    "continue_from_corrupted",
    "other",
    "no_steps",
)
assert set(BEHAVIOR_STACK_ORDER) == set(BEHAVIOR_MODES)


def _load_cell_summaries(level: int, evals_dir: Path) -> list[tuple[str, dict]]:
    suffix = f"-B-L{level}"
    cells: list[tuple[str, dict]] = []
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
            cells.append((model_name, json.load(fh)))
    return cells


def collect_b_results(
    level: int, evals_dir: Path
) -> list[tuple[str, float, tuple[float, float]]]:
    """Return (model, rate, (ci_lo, ci_hi)) per scored cell, best first."""
    results: list[tuple[str, float, tuple[float, float]]] = []
    for model_name, summary in _load_cell_summaries(level, evals_dir):
        headline = summary.get("headline") or {}
        rate = headline.get("final_answer_correct_rate")
        if rate is None:
            print(
                f"skip {model_name}-B-L{level}: no final_answer_correct_rate in headline",
                file=sys.stderr,
            )
            continue
        ci = headline.get("final_answer_correct_ci95")
        if ci is None:
            total = int(headline.get("total") or 0)
            ci = wilson_interval(round(float(rate) * total), total) if total else (0.0, 1.0)
        results.append((model_name, float(rate), (float(ci[0]), float(ci[1]))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def plot_level(
    level: int,
    results: list[tuple[str, float, tuple[float, float]]],
    output_path: Path,
) -> None:
    if not results:
        print(f"no scored cells for B-L{level}", file=sys.stderr)
        return

    models = [name for name, _, _ in results]
    values = [rate * 100 for _, rate, _ in results]
    err_lo = [max(0.0, rate - lo) * 100 for _, rate, (lo, _) in results]
    err_hi = [max(0.0, hi - rate) * 100 for _, rate, (_, hi) in results]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models) + 2), 6))
    bars = ax.bar(range(len(models)), values, color=BAR_COLOR, width=0.6)
    ax.errorbar(
        range(len(models)),
        values,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=3,
    )
    for index, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + err_hi[index] + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{LEVEL_TITLES[level]} — 95% CI")
    ax.set_ylabel("Final answer correct (%)")
    ax.set_ylim(0, 112)
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"wrote {output_path}")


def collect_behavior_results(
    level: int, evals_dir: Path
) -> list[tuple[str, dict[str, float]]]:
    """Return (model, {mode: rate}) per cell that carries behavior_mix."""
    results: list[tuple[str, dict[str, float]]] = []
    for model_name, summary in _load_cell_summaries(level, evals_dir):
        behavior = summary.get("behavior_mix")
        if not behavior:
            print(
                f"skip {model_name}-B-L{level}: no behavior_mix "
                "(rescore with the source dataset available)",
                file=sys.stderr,
            )
            continue
        results.append((model_name, behavior.get("behavior_rates") or {}))
    # Sort by recovery-style behavior share, most self-correcting first.
    results.sort(
        key=lambda x: x[1].get("inplace_correction", 0.0)
        + x[1].get("continue_from_corrected", 0.0),
        reverse=True,
    )
    return results


def plot_behavior_level(
    level: int,
    results: list[tuple[str, dict[str, float]]],
    output_path: Path,
) -> None:
    if not results:
        print(f"no behavior_mix cells for B-L{level}", file=sys.stderr)
        return

    models = [name for name, _ in results]
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models) + 2), 6.5))

    bottoms = [0.0] * len(models)
    for mode in BEHAVIOR_STACK_ORDER:
        heights = [rates.get(mode, 0.0) * 100 for _, rates in results]
        ax.bar(
            range(len(models)),
            heights,
            bottom=bottoms,
            width=0.6,
            color=BEHAVIOR_COLORS[mode],
            edgecolor="white",
            linewidth=1.0,
            label=BEHAVIOR_LABELS[mode],
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_title(BEHAVIOR_TITLES[level])
    ax.set_ylabel("Share of episodes (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )
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
        help="Directory to write the PNGs into.",
    )
    args = parser.parse_args()

    for level in (0, 1):
        results = collect_b_results(level, args.evals_dir)
        plot_level(level, results, args.output_dir / f"b_final_answer_l{level}.png")

        behavior_results = collect_behavior_results(level, args.evals_dir)
        plot_behavior_level(
            level, behavior_results, args.output_dir / f"b_behavior_mix_l{level}.png"
        )


if __name__ == "__main__":
    main()
