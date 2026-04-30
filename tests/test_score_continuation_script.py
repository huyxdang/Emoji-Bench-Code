from __future__ import annotations

import json

import pytest

from emoji_bench.continuation_formatter import format_step
from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.dataset.continuation_dataset import continuation_record
from emoji_bench.domain.formatter import system_to_json
from emoji_bench.domain.generator import generate_system
from emoji_bench.judge.artifacts import prediction_fingerprint
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


def test_score_continuation_final_output_only_when_judge_is_missing(tmp_path, capsys):
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
    assert "error_recovery_rate is unavailable" in summary["note"]


def test_score_continuation_writes_judge_plus_final_output_summary(tmp_path, capsys):
    module = load_script_module("score_continuation")
    eval_dir = tmp_path / "eval"
    dataset_dir = tmp_path / "dataset"
    eval_dir.mkdir()
    dataset_dir.mkdir()

    record, prediction = _make_real_example()
    write_jsonl(eval_dir / "predictions.jsonl", [prediction])
    write_jsonl(dataset_dir / "test.jsonl", [record])
    write_jsonl(
        eval_dir / "judge.jsonl",
        [
            {
                "example_id": prediction["example_id"],
                "prediction_fingerprint": prediction_fingerprint(prediction),
                "error_recovered": True,
                "reasoning": "continuation corrected the seeded error",
                "raw_response_text": "{}",
                "judge_model": "gpt-5.4-mini-no-reasoning",
                "judge_api_model": "gpt-5.4-mini",
            }
        ],
    )
    (eval_dir / "summary.json").write_text(
        json.dumps({"input_path": str(dataset_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    module.sys.argv = ["score_continuation.py", str(eval_dir)]
    module.main()
    capsys.readouterr()

    summary = json.loads((eval_dir / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["headline_kind"] == "judge_plus_final_output"
    assert summary["judged_count"] == 1
    assert summary["judged_coverage"] == 1.0
    assert summary["headline"]["error_recovery_rate"] == 1.0
    assert summary["headline"]["final_answer_correct_rate"] == 1.0
    assert "regex_baseline" in summary
    assert (eval_dir / "nested_scores.jsonl").exists()


def test_score_continuation_rejects_partial_judge_coverage(tmp_path):
    module = load_script_module("score_continuation")
    eval_dir = tmp_path / "eval"
    dataset_dir = tmp_path / "dataset"
    eval_dir.mkdir()
    dataset_dir.mkdir()

    record_1, prediction_1 = _make_real_example()
    record_2 = dict(record_1)
    record_2["example_id"] = "cont-test-001"
    prediction_2 = dict(prediction_1)
    prediction_2["example_id"] = "cont-test-001"
    prediction_2["raw_continuation_text"] = prediction_1["raw_continuation_text"] + "\n"

    write_jsonl(eval_dir / "predictions.jsonl", [prediction_1, prediction_2])
    write_jsonl(dataset_dir / "test.jsonl", [record_1, record_2])
    write_jsonl(
        eval_dir / "judge.jsonl",
        [
            {
                "example_id": prediction_1["example_id"],
                "prediction_fingerprint": prediction_fingerprint(prediction_1),
                "error_recovered": True,
                "reasoning": "only one row judged",
                "raw_response_text": "{}",
                "judge_model": "gpt-5.4-mini-no-reasoning",
                "judge_api_model": "gpt-5.4-mini",
            }
        ],
    )
    (eval_dir / "summary.json").write_text(
        json.dumps({"input_path": str(dataset_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    module.sys.argv = ["score_continuation.py", str(eval_dir)]
    with pytest.raises(ValueError, match="coverage incomplete: 1/2"):
        module.main()
