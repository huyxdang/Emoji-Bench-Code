from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
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
fail_score = os.environ.get("FAKE_FAIL_SCORE_SUBSTRING")

if args and args[0] == "scripts/evaluate_continuation.py" and fail_eval and fail_eval in joined:
    raise SystemExit(1)
if args and args[0] == "scripts/score_continuation.py" and fail_score and fail_score in joined:
    raise SystemExit(1)
""",
        encoding="utf-8",
    )
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IEXEC)
    return shim_path, log_path


def _run_script(
    args: list[str],
    *,
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    script_name: str = "run.sh",
):
    shim_path, log_path = _make_fake_python(tmp_path)
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(shim_path)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [f"./{script_name}", *args],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    calls = []
    if log_path.exists():
        calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return result, calls


def test_run_sh_help_mentions_final_answer_defaults(tmp_path):
    result = subprocess.run(
        ["./run.sh", "--help"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Scores final-answer-only after the eval phase finishes" in result.stdout
    assert "Runs the B headline slices" in result.stdout
    assert "Generates B-variant final-answer plots" in result.stdout


def test_run_sh_rejects_forwarded_output_dir(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--output-dir", "custom-out"],
        tmp_path=tmp_path,
    )

    assert result.returncode == 2
    assert "does not support forwarding --output-dir" in result.stderr
    assert calls == []


def test_run_sh_runs_eval_then_score_for_successful_cells(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--max-concurrent", "8"],
        tmp_path=tmp_path,
    )

    assert result.returncode == 0
    assert "All eval, score, and plot steps completed successfully." in result.stdout
    assert len(calls) == 49

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]
    plot_calls = [call for call in calls if call[0] == "scripts/plot_b_final_answer.py"]

    assert len(eval_calls) == 24
    assert len(score_calls) == 24
    assert len(plot_calls) == 1

    expected_models = [
        "claude-opus-4-7-reasoning-max",
        "claude-opus-4-6-reasoning-max",
        "claude-sonnet-4-6-reasoning-max",
        "gpt-5.5-reasoning-max",
        "gpt-5.2-reasoning-xhigh",
        "gpt-5.4-reasoning-xhigh",
        "gpt-5.4-mini-reasoning-xhigh",
        "gpt-5.4-nano-reasoning-xhigh",
        "gemini-3.1-pro-preview-thinking-high",
        "gemini-3-flash-preview-thinking-high",
        "mistral-large-2512",
        "magistral-medium-2509",
    ]
    assert [call[call.index("--model") + 1] for call in eval_calls[:12]] == expected_models
    assert [call[call.index("--model") + 1] for call in eval_calls[12:]] == expected_models

    first_eval = eval_calls[0]
    assert first_eval[:8] == [
        "scripts/evaluate_continuation.py",
        "artifacts/emoji-bench-dataset-100",
        "--model",
        "claude-opus-4-7-reasoning-max",
        "--mode",
        "prefill",
        "--turn-2-prompt-level",
        "0",
    ]
    assert score_calls[0] == [
        "scripts/score_continuation.py",
        "artifacts/evals/claude-opus-4-7-reasoning-max-B-L0",
    ]
    assert all(call[call.index("--mode") + 1] == "prefill" for call in eval_calls)
    assert [call[call.index("--turn-2-prompt-level") + 1] for call in eval_calls] == ["0"] * 12 + ["1"] * 12
    assert score_calls[12] == [
        "scripts/score_continuation.py",
        "artifacts/evals/claude-opus-4-7-reasoning-max-B-L1",
    ]


def test_run_sh_continues_past_failed_eval_and_skips_scoring_failed_cell(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100"],
        tmp_path=tmp_path,
        extra_env={"FAKE_FAIL_EVAL_SUBSTRING": "gpt-5.4-reasoning-xhigh --mode prefill --turn-2-prompt-level 0"},
    )

    assert result.returncode == 1
    assert "FAILED: model=gpt-5.4-reasoning-xhigh mode=prefill turn_2_level=0" in result.stderr
    assert "Failed eval runs:" in result.stdout

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]

    assert len(eval_calls) == 24
    assert len(score_calls) == 23

    failed_output_dir = "artifacts/evals/gpt-5.4-reasoning-xhigh-B-L0"
    assert all(call[1] != failed_output_dir for call in score_calls)


def test_run_sh_reports_failed_scores(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100"],
        tmp_path=tmp_path,
        extra_env={"FAKE_FAIL_SCORE_SUBSTRING": "claude-opus-4-6-reasoning-max-B-L0"},
    )

    assert result.returncode == 1
    assert "SCORE FAILED: artifacts/evals/claude-opus-4-6-reasoning-max-B-L0" in result.stderr
    assert "Failed score runs:" in result.stdout

    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]

    assert len(score_calls) == 24


def test_run_gpt55_l0_l1_runs_both_prompt_levels_then_scores(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--max-concurrent", "4"],
        tmp_path=tmp_path,
        script_name="run_gpt55_l0_l1.sh",
    )

    assert result.returncode == 0
    assert "All GPT-5.5 L0/L1 eval, score, and plot steps completed successfully." in result.stdout
    assert len(calls) == 6
    assert calls[0] == ["-c", "import openai, matplotlib"]

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]
    plot_calls = [call for call in calls if call[0] == "scripts/plot_b_final_answer.py"]

    assert len(eval_calls) == 2
    assert len(score_calls) == 2
    assert len(plot_calls) == 1

    assert [call[call.index("--turn-2-prompt-level") + 1] for call in eval_calls] == ["0", "1"]
    assert all(call[call.index("--model") + 1] == "gpt-5.5-reasoning-max" for call in eval_calls)
    assert all(call[call.index("--mode") + 1] == "prefill" for call in eval_calls)
    assert score_calls == [
        ["scripts/score_continuation.py", "artifacts/evals/gpt-5.5-reasoning-max-B-L0"],
        ["scripts/score_continuation.py", "artifacts/evals/gpt-5.5-reasoning-max-B-L1"],
    ]


def test_run_gpt55_l0_l1_rejects_forwarded_output_dir(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--output-dir", "custom-out"],
        tmp_path=tmp_path,
        script_name="run_gpt55_l0_l1.sh",
    )

    assert result.returncode == 2
    assert "run_gpt55_l0_l1.sh does not support forwarding --output-dir" in result.stderr
    assert calls == []


def test_run_l1_mistral_runs_expected_models_then_scores(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--max-concurrent", "4"],
        tmp_path=tmp_path,
        script_name="run_l1_mistral.sh",
    )

    assert result.returncode == 0
    assert "All Mistral L1 eval, score, and plot steps completed successfully." in result.stdout
    assert len(calls) == 5

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]
    plot_calls = [call for call in calls if call[0] == "scripts/plot_b_final_answer.py"]

    assert [call[call.index("--model") + 1] for call in eval_calls] == [
        "mistral-large-2512",
        "magistral-medium-2509",
    ]
    assert all(call[call.index("--mode") + 1] == "prefill" for call in eval_calls)
    assert all(call[call.index("--turn-2-prompt-level") + 1] == "1" for call in eval_calls)
    assert score_calls == [
        ["scripts/score_continuation.py", "artifacts/evals/mistral-large-2512-B-L1"],
        ["scripts/score_continuation.py", "artifacts/evals/magistral-medium-2509-B-L1"],
    ]
    assert len(plot_calls) == 1


def test_run_l1_mistral_rejects_forwarded_output_dir(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--output-dir", "custom-out"],
        tmp_path=tmp_path,
        script_name="run_l1_mistral.sh",
    )

    assert result.returncode == 2
    assert "run_l1_mistral.sh does not support forwarding --output-dir" in result.stderr
    assert calls == []


def test_run_l1_gemini_runs_expected_models_then_scores(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--max-concurrent", "4"],
        tmp_path=tmp_path,
        script_name="run_l1_gemini.sh",
    )

    assert result.returncode == 0
    assert "All Gemini L1 eval, score, and plot steps completed successfully." in result.stdout
    assert len(calls) == 5

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]
    plot_calls = [call for call in calls if call[0] == "scripts/plot_b_final_answer.py"]

    assert [call[call.index("--model") + 1] for call in eval_calls] == [
        "gemini-3.1-pro-preview-thinking-high",
        "gemini-3-flash-preview-thinking-high",
    ]
    assert all(call[call.index("--mode") + 1] == "prefill" for call in eval_calls)
    assert all(call[call.index("--turn-2-prompt-level") + 1] == "1" for call in eval_calls)
    assert score_calls == [
        ["scripts/score_continuation.py", "artifacts/evals/gemini-3.1-pro-preview-thinking-high-B-L1"],
        ["scripts/score_continuation.py", "artifacts/evals/gemini-3-flash-preview-thinking-high-B-L1"],
    ]
    assert len(plot_calls) == 1


def test_run_l1_gemini_rejects_forwarded_output_dir(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100", "--", "--output-dir", "custom-out"],
        tmp_path=tmp_path,
        script_name="run_l1_gemini.sh",
    )

    assert result.returncode == 2
    assert "run_l1_gemini.sh does not support forwarding --output-dir" in result.stderr
    assert calls == []


def test_run_l1_gpt_reports_cleanly_when_all_evals_fail(tmp_path):
    result, calls = _run_script(
        ["artifacts/emoji-bench-dataset-100"],
        tmp_path=tmp_path,
        script_name="run_l1_gpt.sh",
        extra_env={"FAKE_FAIL_EVAL_SUBSTRING": "scripts/evaluate_continuation.py"},
    )

    assert result.returncode == 1
    assert "Eval phase completed: 0/4 runs successful." in result.stdout
    assert "No successful eval runs to score." in result.stdout
    assert "Score phase completed: 0/0 runs successful." in result.stdout
    assert "unbound variable" not in result.stderr

    eval_calls = [call for call in calls if call[0] == "scripts/evaluate_continuation.py"]
    score_calls = [call for call in calls if call[0] == "scripts/score_continuation.py"]
    assert len(eval_calls) == 4
    assert score_calls == []
