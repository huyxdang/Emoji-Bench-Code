"""Small statistics helpers for scoring summaries.

Pure-stdlib implementations so the scoring pipeline stays dependency-free:

- ``wilson_interval``: 95% (by default) Wilson score interval for a
  binomial proportion. Preferred over the normal approximation because it
  behaves sensibly at the extremes (k=0, k=n) and at the small sample
  sizes this benchmark runs at (n=25 per difficulty cell).
- ``katz_ratio_ci``: Katz log-interval for a ratio of two independent
  binomial proportions. Used for the survival-rate headline
  (corrupted-condition accuracy / clean-condition accuracy).
"""
from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(lo, hi)`` rounded to 4 decimals. ``(0.0, 1.0)`` when
    ``total`` is 0 — no data means no information.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if not 0 <= successes <= max(total, 0):
        raise ValueError(f"successes must be in [0, {total}], got {successes}")
    if total == 0:
        return (0.0, 1.0)

    n = float(total)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (round(lo, 4), round(hi, 4))


def katz_ratio_ci(
    successes_num: int,
    total_num: int,
    successes_den: int,
    total_den: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Katz log-interval for the ratio of two binomial proportions.

    Numerator proportion: ``successes_num / total_num`` (e.g. corrupted
    condition). Denominator proportion: ``successes_den / total_den``
    (e.g. clean control). Returns ``(lo, hi)`` rounded to 4 decimals, or
    ``(None, None)`` when the interval is undefined (either count is 0 —
    the log-ratio variance is infinite there).
    """
    for label, value in (
        ("total_num", total_num),
        ("total_den", total_den),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be > 0, got {value}")
    if not 0 <= successes_num <= total_num:
        raise ValueError(f"successes_num must be in [0, {total_num}]")
    if not 0 <= successes_den <= total_den:
        raise ValueError(f"successes_den must be in [0, {total_den}]")

    if successes_num == 0 or successes_den == 0:
        return (None, None)

    ratio = (successes_num / total_num) / (successes_den / total_den)
    se = math.sqrt(
        1.0 / successes_num
        - 1.0 / total_num
        + 1.0 / successes_den
        - 1.0 / total_den
    )
    lo = ratio * math.exp(-z * se)
    hi = ratio * math.exp(z * se)
    return (round(lo, 4), round(hi, 4))
