from __future__ import annotations

import json

import pytest

from tests.script_helpers import load_script_module, write_jsonl


def _score_row(example_id: str, difficulty: str, correct: bool) -> dict:
    return {
        "example_id": example_id,
        "difficulty": difficulty,
        "matches_ground_truth": correct,
    }


def _write_run(tmp_path, name: str, rows: list[dict]):
    run_dir = tmp_path / name
    run_dir.mkdir()
    write_jsonl(run_dir / "scores.jsonl", rows)
    return run_dir


def test_compute_survival_joins_and_reports(tmp_path, capsys):
    module = load_script_module("compute_survival")

    # 4 instances: clean solves 3 of 4; corruption keeps 2 of those 3.
    corrupted = [
        _score_row("ex-0", "easy", True),
        _score_row("ex-1", "easy", True),
        _score_row("ex-2", "medium", False),
        _score_row("ex-3", "medium", False),
    ]
    control = [
        _score_row("ex-0-control", "easy", True),
        _score_row("ex-1-control", "easy", True),
        _score_row("ex-2-control", "medium", True),
        _score_row("ex-3-control", "medium", False),
    ]
    corrupted_dir = _write_run(tmp_path, "corrupted", corrupted)
    control_dir = _write_run(tmp_path, "control", control)

    module.sys.argv = ["compute_survival.py", str(corrupted_dir), str(control_dir)]
    module.main()
    capsys.readouterr()

    summary = json.loads(
        (corrupted_dir / "survival_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total"] == 4
    assert summary["corrupted_correct"] == 2
    assert summary["clean_correct"] == 3
    assert summary["corrupted_rate"] == 0.5
    assert summary["clean_rate"] == 0.75
    assert summary["survival_rate"] == pytest.approx(0.6667, abs=1e-3)
    lo, hi = summary["survival_ci95"]
    assert lo < summary["survival_rate"] < hi

    # Conditional variant: of the 3 clean-solved, 2 survived corruption.
    assert summary["conditional_denominator"] == 3
    assert summary["conditional_recovery_rate"] == pytest.approx(0.6667, abs=1e-3)
    c_lo, c_hi = summary["conditional_recovery_ci95"]
    assert c_lo < summary["conditional_recovery_rate"] < c_hi

    by_diff = summary["by_difficulty"]
    assert by_diff["easy"]["survival_rate"] == 1.0
    assert by_diff["medium"]["corrupted_correct"] == 0
    assert by_diff["medium"]["survival_rate"] == 0.0
    assert by_diff["medium"]["survival_ci95"] is None


def test_compute_survival_flags_unmatched_rows(tmp_path, capsys):
    module = load_script_module("compute_survival")
    corrupted_dir = _write_run(
        tmp_path,
        "corrupted",
        [_score_row("ex-0", "easy", True), _score_row("ex-9", "easy", True)],
    )
    control_dir = _write_run(
        tmp_path, "control", [_score_row("ex-0-control", "easy", True)]
    )

    module.sys.argv = ["compute_survival.py", str(corrupted_dir), str(control_dir)]
    module.main()
    capsys.readouterr()

    summary = json.loads(
        (corrupted_dir / "survival_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total"] == 1
    assert summary["unmatched_rows"] == {"corrupted_only": 1, "control_only": 0}


def test_compute_survival_errors_when_disjoint(tmp_path):
    module = load_script_module("compute_survival")
    corrupted = module._load_scores(
        _write_run(tmp_path, "corrupted", [_score_row("ex-0", "easy", True)])
    )
    control = module._load_scores(
        _write_run(tmp_path, "control", [_score_row("zz-1-control", "easy", True)])
    )
    with pytest.raises(ValueError):
        module.compute_survival_summary(corrupted, control)
