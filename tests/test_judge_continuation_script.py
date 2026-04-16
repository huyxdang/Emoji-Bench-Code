from __future__ import annotations

import json
from types import SimpleNamespace

from emoji_bench.judge_artifacts import prediction_fingerprint
from tests.script_helpers import load_script_module, write_jsonl


def _prediction(example_id: str, raw_text: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "difficulty": "easy",
        "chain_length_x": 4,
        "prefill_error_step": 2,
        "ground_truth_final_output": "🪈",
        "wrong_branch_final_output": "🪵",
        "raw_continuation_text": raw_text,
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "mode": "prefill",
    }


def _dataset_row(example_id: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "system_json": "{}",
        "chain_seed": 7,
        "error_seed": 13,
        "target_step_count": 6,
        "prefill_error_step": 2,
        "chain_length_x": 4,
    }


def test_judge_continuation_writes_rows_with_prediction_fingerprints(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = load_script_module("judge_continuation")
    eval_dir = tmp_path / "eval"
    dataset_dir = tmp_path / "dataset"
    eval_dir.mkdir()
    dataset_dir.mkdir()

    prediction = _prediction("cont-000001", "Wait, step 2 is wrong.\nFinal Output: 🪈")
    write_jsonl(eval_dir / "predictions.jsonl", [prediction])
    write_jsonl(dataset_dir / "test.jsonl", [_dataset_row("cont-000001")])
    (eval_dir / "summary.json").write_text(
        json.dumps({"input_path": str(dataset_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    seen_examples: list[str] = []

    def fake_judge_continuation(**kwargs):
        seen_examples.append(kwargs["prediction_row"]["example_id"])
        return SimpleNamespace(
            detected_error=True,
            corrected_step_y=False,
            reasoning="noticed the error but did not restate the step",
            raw_response_text='{"detected_error": true, "corrected_step_y": false}',
        )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "judge_continuation", fake_judge_continuation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "judge_continuation.py",
            str(eval_dir),
            "--judge-model",
            "gpt-5.4-mini-no-reasoning",
        ],
    )

    module.main()
    capsys.readouterr()

    assert seen_examples == ["cont-000001"]
    rows = [
        json.loads(line)
        for line in (eval_dir / "judge.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["example_id"] == "cont-000001"
    assert rows[0]["prediction_fingerprint"] == prediction_fingerprint(prediction)
    assert rows[0]["judge_model"] == "gpt-5.4-mini-no-reasoning"
    assert rows[0]["judge_api_model"] == "gpt-5.4-mini"


def test_judge_continuation_resume_skips_existing_valid_rows(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = load_script_module("judge_continuation")
    eval_dir = tmp_path / "eval"
    dataset_dir = tmp_path / "dataset"
    eval_dir.mkdir()
    dataset_dir.mkdir()

    prediction_1 = _prediction("cont-000001", "Final Output: 🪈")
    prediction_2 = _prediction("cont-000002", "Wait, let me fix it.\nFinal Output: 🪈")
    write_jsonl(eval_dir / "predictions.jsonl", [prediction_1, prediction_2])
    write_jsonl(
        dataset_dir / "test.jsonl",
        [_dataset_row("cont-000001"), _dataset_row("cont-000002")],
    )
    (eval_dir / "summary.json").write_text(
        json.dumps({"input_path": str(dataset_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(
        eval_dir / "judge.jsonl",
        [
            {
                "example_id": "cont-000001",
                "prediction_fingerprint": prediction_fingerprint(prediction_1),
                "detected_error": False,
                "corrected_step_y": False,
                "reasoning": "already judged",
                "raw_response_text": "{}",
                "judge_model": "gpt-4.1-mini",
                "judge_api_model": "gpt-4.1-mini",
            }
        ],
    )

    seen_examples: list[str] = []

    def fake_judge_continuation(**kwargs):
        seen_examples.append(kwargs["prediction_row"]["example_id"])
        return SimpleNamespace(
            detected_error=True,
            corrected_step_y=True,
            reasoning="explicit correction",
            raw_response_text='{"detected_error": true, "corrected_step_y": true}',
        )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "judge_continuation", fake_judge_continuation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "judge_continuation.py",
            str(eval_dir),
            "--judge-model",
            "gpt-4.1-mini",
        ],
    )

    module.main()
    capsys.readouterr()

    assert seen_examples == ["cont-000002"]
    rows = [
        json.loads(line)
        for line in (eval_dir / "judge.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["example_id"] for row in rows] == ["cont-000001", "cont-000002"]


def test_judge_continuation_rejects_non_openai_judge_model(monkeypatch, tmp_path):
    module = load_script_module("judge_continuation")
    eval_dir = tmp_path / "eval"
    dataset_dir = tmp_path / "dataset"
    eval_dir.mkdir()
    dataset_dir.mkdir()
    write_jsonl(eval_dir / "predictions.jsonl", [_prediction("cont-000001", "Final Output: 🪈")])
    write_jsonl(dataset_dir / "test.jsonl", [_dataset_row("cont-000001")])
    (eval_dir / "summary.json").write_text(
        json.dumps({"input_path": str(dataset_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "judge_continuation.py",
            str(eval_dir),
            "--judge-model",
            "claude-haiku-4-5",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser.error for non-openai judge model")
