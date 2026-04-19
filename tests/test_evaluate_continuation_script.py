from __future__ import annotations

import json
from pathlib import Path

from emoji_bench.continuation_formatter import get_turn_2_prompt
from emoji_bench.providers.clients import ProviderUsage
from emoji_bench.providers.transport import ContinuationResponse
from tests.script_helpers import load_script_module, write_jsonl


def _dataset_record(example_id: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "turn_1_user": f"prompt for {example_id}",
        "turn_1_assistant_prefill": f"Start: {example_id}\nStep 1: a = b    [by table]",
        "ground_truth_final_output": "🪈",
        "wrong_branch_final_output": "🪵",
        "chain_length_x": 4,
        "prefill_error_step": 2,
        "difficulty": "easy",
        "error_type": "E_CONTINUE",
    }


def test_evaluate_continuation_writes_predictions_and_summary(monkeypatch, tmp_path, capsys):
    module = load_script_module("evaluate_continuation")
    dataset_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "eval-out"
    write_jsonl(dataset_path, [_dataset_record("cont-000001")])

    calls: list[dict[str, object]] = []

    def fake_request_continuation(**kwargs):
        calls.append(kwargs)
        return ContinuationResponse(
            raw_continuation_text="Step 2: a = b    [by table]\nFinal Output: 🪈",
            response_id="resp_123",
            usage=ProviderUsage(
                input_tokens=11,
                output_tokens=7,
                reasoning_tokens=3,
                total_tokens=21,
            ),
            mode="single_turn",
        )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "request_continuation", fake_request_continuation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "evaluate_continuation.py",
            str(dataset_path),
            "--model",
            "gpt-5.4-mini-no-reasoning",
            "--mode",
            "single_turn",
            "--turn-2-prompt-level",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()
    capsys.readouterr()

    assert len(calls) == 1
    assert calls[0]["mode"] == "single_turn"
    assert calls[0]["turn_2_user"] == get_turn_2_prompt(1)
    assert calls[0]["max_output_tokens"] == 4096

    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 1
    assert predictions[0]["example_id"] == "cont-000001"
    assert predictions[0]["turn_2_level"] == 1
    assert predictions[0]["turn_2_user_sent"] == get_turn_2_prompt(1)
    assert predictions[0]["model"] == "gpt-5.4-mini-no-reasoning"
    assert predictions[0]["provider"] == "openai"
    assert predictions[0]["input_tokens"] == 11
    assert predictions[0]["reasoning_tokens"] == 3

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["matrix_variant"] == "C"
    assert summary["matrix_cell"] == "C-L1"
    assert summary["turn_2_level"] == 1
    assert summary["turn_2_user_sent"] == get_turn_2_prompt(1)
    assert summary["openai_reasoning_effort"] == "none"
    assert summary["max_output_tokens"] == 4096
    assert summary["completed_examples"] == 1
    assert summary["total_examples"] == 1


def test_evaluate_continuation_summary_records_max_output_tokens_override(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = load_script_module("evaluate_continuation")
    dataset_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "eval-out"
    write_jsonl(dataset_path, [_dataset_record("cont-000001")])

    calls: list[dict[str, object]] = []

    def fake_request_continuation(**kwargs):
        calls.append(kwargs)
        return ContinuationResponse(
            raw_continuation_text="Final Output: 🪈",
            response_id="resp_123",
            usage=None,
            mode="prefill",
        )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "request_continuation", fake_request_continuation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "evaluate_continuation.py",
            str(dataset_path),
            "--model",
            "gpt-5.4-mini-no-reasoning",
            "--max-output-tokens",
            "777",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()
    capsys.readouterr()

    assert len(calls) == 1
    assert calls[0]["max_output_tokens"] == 777

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["max_output_tokens"] == 777


def test_evaluate_continuation_defaults_max_concurrent_to_10(monkeypatch, tmp_path, capsys):
    module = load_script_module("evaluate_continuation")
    dataset_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "eval-out"
    write_jsonl(dataset_path, [_dataset_record("cont-000001")])

    captured: dict[str, object] = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model_config"].key,
            "provider": kwargs["model_config"].provider,
            "api_model": kwargs["model_config"].api_model,
            "mode": kwargs["options"].mode,
            "completed_examples": len(kwargs["records"]),
            "total_examples": len(kwargs["records"]),
        }

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "evaluate_continuation.py",
            str(dataset_path),
            "--model",
            "gpt-5.4-mini-no-reasoning",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()
    capsys.readouterr()

    assert captured["options"].max_concurrent == 10


def test_evaluate_continuation_resume_skips_already_written_examples(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = load_script_module("evaluate_continuation")
    dataset_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "eval-out"
    output_dir.mkdir()
    write_jsonl(
        dataset_path,
        [
            _dataset_record("cont-000001"),
            _dataset_record("cont-000002"),
        ],
    )
    write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {
                "example_id": "cont-000001",
                "difficulty": "easy",
                "chain_length_x": 4,
                "prefill_error_step": 2,
                "ground_truth_final_output": "🪈",
                "wrong_branch_final_output": "🪵",
                "raw_continuation_text": "Final Output: 🪈",
                "mode": "prefill",
                "turn_2_user_sent": get_turn_2_prompt(0),
                "turn_2_level": 0,
                "model": "gpt-4.1-mini",
                "provider": "openai",
                "api_model": "gpt-4.1-mini",
            }
        ],
    )

    seen_turn_1_prompts: list[str] = []

    def fake_request_continuation(**kwargs):
        seen_turn_1_prompts.append(kwargs["turn_1_user"])
        return ContinuationResponse(
            raw_continuation_text="Final Output: 🪈",
            response_id="resp_resume",
            usage=None,
            mode="prefill",
        )

    monkeypatch.setattr(module, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(module, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(module, "make_client", lambda provider, api_key: object())
    monkeypatch.setattr(module, "request_continuation", fake_request_continuation)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "evaluate_continuation.py",
            str(dataset_path),
            "--model",
            "gpt-4.1-mini",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()
    capsys.readouterr()

    assert seen_turn_1_prompts == ["prompt for cont-000002"]
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["example_id"] for row in predictions] == ["cont-000001", "cont-000002"]


def test_default_output_dir_uses_matrix_cell_and_reasoning_suffix():
    module = load_script_module("evaluate_continuation")

    assert module._default_output_dir("gpt-5.4-mini", "prefill", turn_2_level=0) == Path(
        "artifacts/evals/gpt-5.4-mini-B-L0"
    )
    assert module._default_output_dir(
        "gpt-5.4-mini",
        "single_turn",
        reasoning_effort="xhigh",
        turn_2_level=1,
    ) == Path("artifacts/evals/gpt-5.4-mini-reasoning-xhigh-C-L1")
