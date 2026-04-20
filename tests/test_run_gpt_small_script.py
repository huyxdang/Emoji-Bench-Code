from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "calls.jsonl"
    shim_path = tmp_path / "fake_python.py"
    shim_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["FAKE_PYTHON_LOG"])
args = sys.argv[1:]
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(args, ensure_ascii=False) + "\\n")

joined = " ".join(args)
fail_eval = os.environ.get("FAKE_FAIL_EVAL_SUBSTRING")
fail_judge = os.environ.get("FAKE_FAIL_JUDGE_SUBSTRING")
fail_score = os.environ.get("FAKE_FAIL_SCORE_SUBSTRING")

if args and args[0] == "scripts/evaluate_continuation.py" and fail_eval and fail_eval in joined:
    raise SystemExit(1)
if args and args[0] == "scripts/judge_continuation.py" and fail_judge and fail_judge in joined:
    raise SystemExit(1)
if args and args[0] == "scripts/score_continuation.py" and fail_score and fail_score in joined:
    raise SystemExit(1)
""",
        encoding="utf-8",
    )
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IEXEC)
    return shim_path, log_path


def _run_script(args: list[str], *, tmp_path: Path, extra_env: dict[str, str] | None = None):
    shim_path, log_path = _make_fake_python(tmp_path)
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(shim_path)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["./run_gpt_small.sh", *args],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    calls = []
    if log_path.exists():
        calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return result, calls


def test_run_gpt_small_sh_help_mentions_small_model_defaults(tmp_path):
    result = subprocess.run(
        ["./run_gpt_small.sh", "--help"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Runs only gpt-5.4-mini-reasoning-xhigh on the B slice (L0 and L1)" in result.stdout
    assert "gpt-5.4-mini-no-reasoning" in result.stdout


def test_run_gpt_small_sh_rejects_forwarded_output_dir_and_model(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--output-dir", "custom-out"],
        tmp_path=tmp_path,
    )
    assert result.returncode == 2
    assert "does not support forwarding --output-dir" in result.stderr
    assert calls == []

    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--model", "gpt-5.4-mini"],
        tmp_path=tmp_path,
    )
    assert result.returncode == 2
    assert "does not support forwarding --model" in result.stderr
    assert calls == []


def test_run_gpt_small_sh_runs_b_l0_and_b_l1_for_gpt54_mini_then_judge_then_score(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100"],
        tmp_path=tmp_path,
    )

    assert result.returncode == 0
    assert "All GPT-5.4 small-model eval, judge, and score runs completed successfully." in result.stdout
    assert len(calls) == 6

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    judge_calls = [call for call in calls if call[0] == "scripts/judge_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]

    assert len(eval_calls) == 2
    assert len(judge_calls) == 2
    assert len(score_calls) == 2
    assert sorted({call[3] for call in eval_calls}) == [
        "gpt-5.4-mini-reasoning-xhigh",
    ]
    assert all(call[5] == "prefill" for call in eval_calls)
    assert sorted((call[3], call[7]) for call in eval_calls) == [
        ("gpt-5.4-mini-reasoning-xhigh", "0"),
        ("gpt-5.4-mini-reasoning-xhigh", "1"),
    ]
    assert all("--max-output-tokens" not in call for call in eval_calls)
    assert [call[1] for call in judge_calls] == [
        "artifacts/evals/gpt-5.4-mini-reasoning-xhigh-B-L0",
        "artifacts/evals/gpt-5.4-mini-reasoning-xhigh-B-L1",
    ]
    assert [call[1] for call in score_calls] == [
        "artifacts/evals/gpt-5.4-mini-reasoning-xhigh-B-L0",
        "artifacts/evals/gpt-5.4-mini-reasoning-xhigh-B-L1",
    ]
