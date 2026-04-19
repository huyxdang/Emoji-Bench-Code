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
        ["./run_gpt.sh", *args],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    calls = []
    if log_path.exists():
        calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return result, calls


def test_run_gpt_sh_help_mentions_gpt54_defaults(tmp_path):
    result = subprocess.run(
        ["./run_gpt.sh", "--help"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "GPT-5.4 reasoning-xhigh matrix" in result.stdout
    assert "gpt-5.4-mini-no-reasoning" in result.stdout


def test_run_gpt_sh_rejects_forwarded_output_dir_and_model(tmp_path):
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


def test_run_gpt_sh_runs_eval_then_judge_then_score_for_successful_cells(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--max-concurrent", "8"],
        tmp_path=tmp_path,
    )

    assert result.returncode == 0
    assert "All GPT-5.4 eval, judge, and score runs completed successfully." in result.stdout
    assert len(calls) == 12

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    judge_calls = [call for call in calls if call[0] == "scripts/judge_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]

    assert len(eval_calls) == 4
    assert len(judge_calls) == 4
    assert len(score_calls) == 4
    assert all(call[3] == "gpt-5.4-reasoning-xhigh" for call in eval_calls)

    first_eval = eval_calls[0]
    assert first_eval[:8] == [
        "scripts/evaluate_continuation.py",
        "artifacts/emoji-bench-dataset-100",
        "--model",
        "gpt-5.4-reasoning-xhigh",
        "--mode",
        "prefill",
        "--turn-2-prompt-level",
        "0",
    ]
    assert judge_calls[0][:5] == [
        "scripts/judge_continuation.py",
        "artifacts/evals/gpt-5.4-reasoning-xhigh-B-L0",
        "--judge-model",
        "gpt-5.4-mini-no-reasoning",
        "--max-concurrent",
    ]
    assert score_calls[0] == [
        "scripts/score_continuation.py",
        "artifacts/evals/gpt-5.4-reasoning-xhigh-B-L0",
    ]
