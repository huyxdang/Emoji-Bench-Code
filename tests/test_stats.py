import math

import pytest

from emoji_bench.scoring.stats import katz_ratio_ci, wilson_interval


def test_wilson_interval_no_data_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_half_at_n_100():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_extremes_stay_in_bounds():
    lo, hi = wilson_interval(100, 100)
    assert hi == 1.0
    assert lo == pytest.approx(0.9630, abs=1e-3)

    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert hi == pytest.approx(0.0370, abs=1e-3)


def test_wilson_interval_narrows_with_sample_size():
    lo_small, hi_small = wilson_interval(5, 10)
    lo_big, hi_big = wilson_interval(500, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 4)


def test_katz_ratio_ci_unit_ratio():
    lo, hi = katz_ratio_ci(50, 100, 50, 100)
    assert lo is not None and hi is not None
    se = math.sqrt(0.02)
    assert lo == pytest.approx(math.exp(-1.96 * se), abs=1e-3)
    assert hi == pytest.approx(math.exp(1.96 * se), abs=1e-3)
    assert lo < 1.0 < hi


def test_katz_ratio_ci_undefined_at_zero_counts():
    assert katz_ratio_ci(0, 100, 50, 100) == (None, None)
    assert katz_ratio_ci(50, 100, 0, 100) == (None, None)


def test_katz_ratio_ci_rejects_bad_inputs():
    with pytest.raises(ValueError):
        katz_ratio_ci(1, 0, 1, 10)
    with pytest.raises(ValueError):
        katz_ratio_ci(11, 10, 1, 10)
