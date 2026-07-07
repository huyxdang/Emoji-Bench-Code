from __future__ import annotations

import json

import pytest

from emoji_bench.continuation_formatter import format_step
from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.dataset.continuation_dataset import continuation_record
from emoji_bench.domain.formatter import system_to_json
from emoji_bench.domain.generator import generate_system
from tests.script_helpers import load_script_module, write_jsonl


def _make_real_example() -> tuple[dict[str, object], dict[str, object]]:
    system = generate_system(
        n_symbols=3,
        n_base_ops=1,
        n_derived_ops=0,
        n_transformations=0,
        random_seed=11,
    )
    instance = generate_continuation_instance(system, length=6, chain_seed=7, error_seed=13)
    record = continuation_record(
        instance=instance,
        example_id="cont-test-000",
        base_id="easy-0000",
        split="test",
        difficulty="easy",
        system_json=system_to_json(system),
        system_seed=11,
        chain_seed=7,
        error_seed=13,
        target_step_count=6,
    )
    continuation_lines = [
        format_step(step, system)
        for step in instance.clean_chain.steps[record["prefill_error_step"] - 1 :]
    ]
    continuation_lines.append(f"Final Output: {record['ground_truth_final_output']}")
    prediction = {
        "example_id": record["example_id"],
        "difficulty": record["difficulty"],
        "chain_length_x": record["chain_length_x"],
        "prefill_error_step": record["prefill_error_step"],
        "ground_truth_final_output": record["ground_truth_final_output"],
        "wrong_branch_final_output": record["wrong_branch_final_output"],
        "raw_continuation_text": "\n".join(continuation_lines),
        "model": "claude-opus-4-6-reasoning-high",
        "provider": "anthropic",
        "mode": "prefill",
        "turn_2_level": 0,
    }
    return record, prediction


def test_score_continuation_writes_final_output_only_summary(tmp_path, capsys):
    module = load_script_module("score_continuation")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _, prediction = _make_real_example()
    write_jsonl(eval_dir / "predictions.jsonl", [prediction])

    module.sys.argv = ["score_continuation.py", str(eval_dir)]
    module.main()
    capsys.readouterr()

    summary = json.loads((eval_dir / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["headline_kind"] == "final_output_only"
    assert "regex_baseline" in summary
    assert summary["headline"]["total"] == 1
    assert summary["headline"]["final_answer_correct_rate"] == 1.0
    assert summary["headline"]["final_answer_correct_ci95"][0] <= 1.0
    # No dataset available: behavior classification degrades to None.
    assert summary["behavior_mix"] is None
    scores = [
        json.loads(line)
        for line in (eval_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert scores[0]["behavior_mode"] is None
    assert "note" not in summary


def test_score_continuation_with_dataset_classifies_behavior(tmp_path, capsys):
    module = load_script_module("score_continuation")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    record, prediction = _make_real_example()
    write_jsonl(eval_dir / "predictions.jsonl", [prediction])
    dataset_path = tmp_path / "test.jsonl"
    write_jsonl(dataset_path, [record])

    module.sys.argv = [
        "score_continuation.py",
        str(eval_dir),
        "--dataset-path",
        str(dataset_path),
    ]
    module.main()
    capsys.readouterr()

    summary = json.loads((eval_dir / "score_summary.json").read_text(encoding="utf-8"))
    behavior = summary["behavior_mix"]
    assert behavior is not None
    assert behavior["total_classified"] == 1
    # The synthesized continuation re-derives the erroneous step correctly.
    assert behavior["behavior_counts"]["inplace_correction"] == 1
    assert behavior["derivation_valid_rate"] == 1.0
    assert summary["headline"]["chance_rate"] == pytest.approx(1 / 3, abs=1e-3)

    scores = [
        json.loads(line)
        for line in (eval_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert scores[0]["behavior_mode"] == "inplace_correction"
    assert scores[0]["derivation_valid"] is True
    assert scores[0]["n_symbols"] == 3
