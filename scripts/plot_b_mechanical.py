"""Bar chart of `mechanical_correct_rate` for every B-variant run.

Produces one combined PNG under artifacts/plots/:
  * b_mechanical.png — two bars per model (L0 on top, L1 below), sorted by
    L0 rate descending.

No exclusions — every `*-B-L{0,1}/score_summary.json` with a
`judge_plus_validator` headline is included, including 4096-capped variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "artifacts" / "evals"
PLOTS_DIR = REPO_ROOT / "artifacts" / "plots"

L0_COLOR = "#3a6ea5"  # blue
L1_COLOR = "#f4a261"  # orange


def _collect() -> dict[str, dict[int, float]]:
    """Return {model_name: {level: rate}} for B-slice judge_plus_validator runs."""
    out: dict[str, dict[int, float]] = {}
    for level in (0, 1):
        suffix = f"-B-L{level}"
        for d in sorted(EVALS_DIR.iterdir()):
            if not d.is_dir() or not d.name.endswith(suffix):
                continue
            summary_path = d / "score_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("headline_kind") != "judge_plus_validator":
                continue
            rate = summary.get("headline", {}).get("mechanical_correct_rate")
            if rate is None:
                continue
            model = d.name[: -len(suffix)]
            out.setdefault(model, {})[level] = float(rate)
    return out


def _plot(data: dict[str, dict[int, float]], out_path: Path) -> None:
    if not data:
        print("no rows; skipping")
        return

    # Sort by L0 rate desc; models missing L0 go last, tie-break by L1 then name.
    def sort_key(item: tuple[str, dict[int, float]]):
        _, rates = item
        l0 = rates.get(0, -1.0)
        l1 = rates.get(1, -1.0)
        return (-l0, -l1)

    ordered = sorted(data.items(), key=sort_key)
    models = [m for m, _ in ordered]
    l0_rates = [rates.get(0, 0.0) for _, rates in ordered]
    l1_rates = [rates.get(1, 0.0) for _, rates in ordered]
    l0_present = [0 in rates for _, rates in ordered]
    l1_present = [1 in rates for _, rates in ordered]

    y = np.arange(len(models))
    bar_height = 0.4

    fig_height = max(5.0, 0.55 * len(models) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars_l0 = ax.barh(
        y - bar_height / 2, l0_rates, bar_height,
        color=L0_COLOR, label="L0 (no hint)",
    )
    bars_l1 = ax.barh(
        y + bar_height / 2, l1_rates, bar_height,
        color=L1_COLOR, label="L1 (with hint)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # best model at top
    ax.set_xlim(0, 1)
    ax.set_xlabel("mechanical_correct_rate")
    ax.set_title("Mechanical correct  —  B slice  (L0 top, L1 bottom per model)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right")

    for bar, rate, present in zip(bars_l0, l0_rates, l0_present):
        if not present:
            continue
        ax.text(rate + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}", va="center", fontsize=8)
    for bar, rate, present in zip(bars_l1, l1_rates, l1_present):
        if not present:
            continue
        ax.text(rate + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    data = _collect()
    _plot(data, PLOTS_DIR / "b_mechanical.png")


if __name__ == "__main__":
    main()
