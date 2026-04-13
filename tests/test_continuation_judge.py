import json
from types import SimpleNamespace

import pytest

from emoji_bench.continuation_benchmark import generate_continuation_instance
from emoji_bench.continuation_dataset import continuation_record
from emoji_bench.continuation_judge import (
    JudgeVerdict,
    build_judge_prompt,
    compute_step_values,
    judge_continuation,
)
from emoji_bench.formatter import system_to_json
from emoji_bench.generator import generate_system
from emoji_bench.model_registry import get_model_config


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def pilot_pair():
    """Build a real (dataset_row, prediction_row_scaffold) pair deterministically."""
    system = generate_system(
        n_symbols=3, n_base_ops=1, n_derived_ops=0, n_transformations=0,
        random_seed=11,
    )
    instance = generate_continuation_instance(
        system, length=6, chain_seed=7, error_seed=13,
    )
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
    pred_scaffold = {
        "example_id": record["example_id"],
        "prefill_error_step": record["prefill_error_step"],
        "prefill_cutoff_step": record["prefill_cutoff_step"],
        "chain_length_x": record["chain_length_x"],
        "ground_truth_final_output": record["ground_truth_final_output"],
        "wrong_branch_final_output": record["wrong_branch_final_output"],
        "raw_continuation_text": "",  # filled by individual tests
    }
    return record, pred_scaffold


# --- compute_step_values ---------------------------------------------------


def test_compute_step_values_round_trip(pilot_pair):
    dataset_row, _ = pilot_pair
    values = compute_step_values(dataset_row=dataset_row)
    assert values.correct_value != values.injected_value
    # The injected result must match what the record says was wrong.
    assert values.injected_value.emoji != values.correct_value.emoji
    # The reconstructed "correct" step string matches a plausible expression
    # render (non-empty, different from the injected one).
    assert values.after_correct_str != values.after_injected_str


def test_compute_step_values_rejects_missing_fields(pilot_pair):
    dataset_row, _ = pilot_pair
    pruned = dict(dataset_row)
    pruned.pop("system_json")
    with pytest.raises(ValueError, match="missing fields"):
        compute_step_values(dataset_row=pruned)


def test_compute_step_values_detects_seed_drift(pilot_pair):
    dataset_row, _ = pilot_pair
    drifted = dict(dataset_row)
    drifted["chain_length_x"] = 999  # lies about what was stored
    with pytest.raises(RuntimeError, match="regeneration drift"):
        compute_step_values(dataset_row=drifted)


# --- build_judge_prompt ---------------------------------------------------


def test_build_judge_prompt_mentions_step_number_and_values(pilot_pair):
    dataset_row, pred = pilot_pair
    pred["raw_continuation_text"] = "Step X: ..."
    step_values = compute_step_values(dataset_row=dataset_row)
    prompt = build_judge_prompt(prediction_row=pred, step_values=step_values)
    assert f"Step {pred['prefill_error_step']}" in prompt
    assert step_values.correct_value.emoji in prompt
    assert step_values.injected_value.emoji in prompt
    assert "detected_error" in prompt
    assert "corrected_step_y" in prompt
    # Must warn the judge off grading the math.
    assert "Do not grade the math" in prompt
    # Must include the model's continuation verbatim.
    assert "Step X: ..." in prompt


def test_build_judge_prompt_instructs_narrow_metric_2(pilot_pair):
    dataset_row, pred = pilot_pair
    pred["raw_continuation_text"] = ""
    step_values = compute_step_values(dataset_row=dataset_row)
    prompt = build_judge_prompt(prediction_row=pred, step_values=step_values)
    # The narrow definition must be present (explicit restatement required).
    assert "Implicit corrections" in prompt
    assert "do NOT count" in prompt


# --- judge_continuation (mocked client) ------------------------------------


class _FakeParsed:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return dict(self._payload)


class _FakeResponsesAPI:
    def __init__(self, *, payload: dict, raw_text: str):
        self.payload = payload
        self.raw_text = raw_text
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="judge_test",
            output_parsed=_FakeParsed(self.payload),
            output_text=self.raw_text,
            output=[],
        )


class _FakeOpenAIClient:
    def __init__(self, *, payload: dict, raw_text: str):
        self.responses = _FakeResponsesAPI(payload=payload, raw_text=raw_text)


def test_judge_continuation_returns_structured_verdict(pilot_pair):
    dataset_row, pred = pilot_pair
    pred["raw_continuation_text"] = "Step 3: ... Final Output: 🪈"

    client = _FakeOpenAIClient(
        payload={
            "detected_error": True,
            "corrected_step_y": False,
            "reasoning": "Model hedged but did not restate step Y.",
        },
        raw_text='{"detected_error": true, "corrected_step_y": false, "reasoning": "..."}',
    )
    verdict = judge_continuation(
        client=client,
        judge_model_config=get_model_config("gpt-4.1-mini"),
        prediction_row=pred,
        dataset_row=dataset_row,
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.detected_error is True
    assert verdict.corrected_step_y is False
    assert "step Y" in verdict.reasoning or "hedged" in verdict.reasoning


def test_judge_continuation_passes_prompt_and_schema_to_api(pilot_pair):
    dataset_row, pred = pilot_pair
    pred["raw_continuation_text"] = "(ignored in mock)"

    client = _FakeOpenAIClient(
        payload={
            "detected_error": False,
            "corrected_step_y": False,
            "reasoning": "no hedge",
        },
        raw_text="{}",
    )
    judge_continuation(
        client=client,
        judge_model_config=get_model_config("gpt-4.1-mini"),
        prediction_row=pred,
        dataset_row=dataset_row,
    )
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["input"][0]["role"] == "system"
    assert call["input"][1]["role"] == "user"
    assert "detected_error" in call["input"][1]["content"]
    # text_format must be passed so the Responses API enforces the JSON schema.
    assert "text_format" in call


def test_judge_continuation_rejects_non_openai_provider(pilot_pair):
    dataset_row, pred = pilot_pair
    with pytest.raises(NotImplementedError, match="openai-provider"):
        judge_continuation(
            client=object(),
            judge_model_config=get_model_config("claude-haiku-4-5"),
            prediction_row=pred,
            dataset_row=dataset_row,
        )
