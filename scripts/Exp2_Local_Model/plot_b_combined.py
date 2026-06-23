"""Horizontal paired bar chart: L0 (red, no hint) vs L1 (black, with hint).

Produces one PNG in the output directory:
  * b_combined_l0_l1.png

Reads ``final_answer_correct_rate`` from each cell's ``score_summary.json``
under ``headline``. Models missing either L0 or L1 are skipped.
Sorted by L0 rate descending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR / "artifacts" / "evals"
OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "plots"

COLOR_L0 = "#C0392B"   # red  — without hint
COLOR_L1 = "#2C2C2C"   # near-black — with hint

EXCLUDED_MODELS: set[str] = set()


def _read_rate(cell_dir: Path) -> float | None:
    summary_path = cell_dir / "score_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as fh:
        summary = json.load(fh)
    headline = summary.get("headline") or {}
    rate = headline.get("final_answer_correct_rate")
    return float(rate) if rate is not None else None


def collect_results(evals_dir: Path) -> list[tuple[str, float, float]]:
    """Return list of (model_name, l0_rate, l1_rate) sorted by l0 descending."""
    l0: dict[str, float] = {}
    l1: dict[str, float] = {}

    for cell_dir in sorted(evals_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name
        if "4096" in name:
            continue

        if name.endswith("-B-L0"):
            model = name[: -len("-B-L0")]
            if model in EXCLUDED_MODELS:
                continue
            rate = _read_rate(cell_dir)
            if rate is not None:
                l0[model] = rate
        elif name.endswith("-B-L1"):
            model = name[: -len("-B-L1")]
            if model in EXCLUDED_MODELS:
                continue
            rate = _read_rate(cell_dir)
            if rate is not None:
                l1[model] = rate

    common = sorted(
        l0.keys() & l1.keys(),
        key=lambda m: l0[m],
        reverse=True,
    )

    missing_l1 = set(l0) - set(l1)
    missing_l0 = set(l1) - set(l0)
    for m in sorted(missing_l1):
        print(f"skip {m}: no L1 score", file=sys.stderr)
    for m in sorted(missing_l0):
        print(f"skip {m}: no L0 score", file=sys.stderr)

    return [(m, l0[m], l1[m]) for m in common]


def plot(
    results: list[tuple[str, float, float]],
    output_path: Path,
) -> None:
    if not results:
        print("no scored models with both L0 and L1", file=sys.stderr)
        return

    n = len(results)
    row_height = 0.9          # height allocated per model row
    bar_h = 0.32              # height of each individual bar
    gap = 0.06                # gap between the two bars of one model

    fig_height = max(4, n * row_height + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(n))         # one integer y-position per model

    for i, (model, r0, r1) in enumerate(results):
        y = n - 1 - i                    # top model at highest y
        v0 = r0 * 100
        v1 = r1 * 100

        # L0 bar (top, red)
        y0 = y + gap / 2
        ax.barh(y0, v0, height=bar_h, color=COLOR_L0, align="edge")
        ax.text(v0 + 0.8, y0 + bar_h / 2, f"{v0:.0f}%",
                va="center", ha="left", fontsize=8.5, color=COLOR_L0, fontweight="bold")

        # L1 bar (bottom, black)
        y1 = y - bar_h - gap / 2
        ax.barh(y1, v1, height=bar_h, color=COLOR_L1, align="edge")
        ax.text(v1 + 0.8, y1 + bar_h / 2, f"{v1:.0f}%",
                va="center", ha="left", fontsize=8.5, color=COLOR_L1, fontweight="bold")

        # Diff annotation on the right
        diff = v1 - v0
        diff_str = f"+{diff:.0f}%" if diff >= 0 else f"{diff:.0f}%"
        diff_color = "#27AE60" if diff >= 0 else COLOR_L0
        ax.text(103, y - bar_h / 2, diff_str,
                va="center", ha="left", fontsize=9, color=diff_color, fontweight="bold")

    # Y-axis: model names centred between the two bars
    ax.set_yticks([n - 1 - i - bar_h / 2 for i in range(n)])
    ax.set_yticklabels([m for m, *_ in results], fontsize=10)
    ax.set_ylim(-1, n)

    ax.set_xlabel("Final answer correct (%)", fontsize=10)
    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_L0, label="L0 — without hint"),
        mpatches.Patch(color=COLOR_L1, label="L1 — with hint"),
    ]
    # ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.7)

    ax.set_title("Final-answer correctness: L0 vs L1 (B-variant)", fontsize=12, pad=10)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
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
        default=OUTPUT_DIR,
        help=f"Directory to write the PNG into (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    results = collect_results(args.evals_dir)

    summary = {
        model: {"L0": round(r0 * 100, 1), "L1": round(r1 * 100, 1), "Diff": round((r1 - r0) * 100, 1)}
        for model, r0, r1 in results
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    output_path = args.output_dir / "b_combined_l0_l1.png"
    plot(results, output_path)


if __name__ == "__main__":
    main()
