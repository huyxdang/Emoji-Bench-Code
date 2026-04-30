"""Phase 5b: LLM-as-judge for broad seeded-error recovery.

The judge answers one yes/no question about the model's raw continuation:

``error_recovered`` — did the continuation recover from the seeded prefill
error, either explicitly (e.g. "wait, step Y should be ...") or implicitly
(e.g. later steps clearly continue from the corrected state)?

The judge never has to recompute the formal-system math. We feed it the
correct and injected values at prompt time, reconstructed deterministically
from the dataset's seeds via ``generate_continuation_instance``. That keeps
the task focused on reading comprehension instead of symbolic evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.domain.expressions import expr_to_str_with_system
from emoji_bench.domain.formatter import system_from_json
from emoji_bench.model_registry import ModelConfig
from emoji_bench.domain.types import Symbol


@dataclass(frozen=True)
class JudgeVerdict:
    error_recovered: bool
    reasoning: str
    raw_response_text: str


@dataclass(frozen=True)
class StepValues:
    """Correct and injected values for the bad step, plus context strings."""
    correct_value: Symbol
    injected_value: Symbol
    before_str: str   # the expression right before the bad step
    after_correct_str: str  # how the step SHOULD render if done correctly
    after_injected_str: str  # how the prefill actually rendered it


def compute_step_values(
    *,
    dataset_row: dict[str, Any],
) -> StepValues:
    """Reconstruct correct/injected values for the bad step from dataset seeds.

    Regenerates the ``ContinuationInstance`` deterministically from
    ``(system_seed, chain_seed, error_seed, target_step_count)`` and pulls the
    reference step values off ``instance.error_info``. Asserts that the
    regenerated instance's structural metadata matches the stored row so a
    seed drift is caught loudly instead of silently feeding the judge wrong
    reference values.
    """
    required = (
        "system_json", "chain_seed", "error_seed", "target_step_count",
        "prefill_error_step", "chain_length_x",
    )
    missing = [f for f in required if f not in dataset_row]
    if missing:
        raise ValueError(
            f"dataset row {dataset_row.get('example_id')!r} missing fields "
            f"for judge step-value reconstruction: {missing}"
        )

    system = system_from_json(dataset_row["system_json"])
    instance = generate_continuation_instance(
        system,
        length=dataset_row["target_step_count"],
        chain_seed=dataset_row["chain_seed"],
        error_seed=dataset_row["error_seed"],
    )

    # Safety checks: the regenerated instance must agree with the stored row.
    if instance.chain_length_x != dataset_row["chain_length_x"]:
        raise RuntimeError(
            f"regeneration drift: stored chain_length_x="
            f"{dataset_row['chain_length_x']} but regenerated "
            f"{instance.chain_length_x}"
        )
    if instance.prefill_error_step != dataset_row["prefill_error_step"]:
        raise RuntimeError(
            f"regeneration drift: stored prefill_error_step="
            f"{dataset_row['prefill_error_step']} but regenerated "
            f"{instance.prefill_error_step}"
        )

    error_info = instance.error_info
    assert error_info.correct_result is not None
    assert error_info.injected_result is not None

    # Render the before/after expressions for the judge's prompt.
    bad_step = instance.mutated_chain.steps[error_info.step_number - 1]
    before_str = expr_to_str_with_system(bad_step.before, system)
    after_injected_str = expr_to_str_with_system(bad_step.after, system)
    # The "correct after" is what the step should have rendered as: same
    # before, but the reduced subexpression replaced with the correct result
    # — the original ``step.after`` from the clean chain holds this.
    clean_step = instance.clean_chain.steps[error_info.step_number - 1]
    after_correct_str = expr_to_str_with_system(clean_step.after, system)

    return StepValues(
        correct_value=error_info.correct_result,
        injected_value=error_info.injected_result,
        before_str=before_str,
        after_correct_str=after_correct_str,
        after_injected_str=after_injected_str,
    )


def build_judge_prompt(
    *,
    prediction_row: dict[str, Any],
    step_values: StepValues,
) -> str:
    """Reading-only judge prompt for broad error recovery."""
    error_step = prediction_row["prefill_error_step"]
    return (
        f"A formal-system derivation was pre-filled with a deliberate error at Step {error_step}.\n"
        f"\n"
        f"- The pre-filled Step {error_step} claims: "
        f"{step_values.before_str} = {step_values.after_injected_str}\n"
        f"- The correct value of the reducible subexpression is "
        f"{step_values.correct_value.emoji} (not {step_values.injected_value.emoji}). "
        f"Written out, Step {error_step} SHOULD read "
        f"{step_values.before_str} = {step_values.after_correct_str}.\n"
        f"\n"
        f"A model was asked to continue from that pre-filled state. Here is the model's continuation:\n"
        f"\n"
        f"---\n"
        f"{prediction_row['raw_continuation_text']}\n"
        f"---\n"
        f"\n"
        f"Answer one yes/no question about the CONTINUATION TEXT ONLY. Do not grade the math.\n"
        f"\n"
        f"Question: error_recovered\n"
        f"Answer yes if the continuation identifies and corrects the seeded error, either:\n"
        f"- explicitly, by saying the earlier step was wrong and fixing it, or\n"
        f"- implicitly, by clearly continuing from the corrected value/state instead of the "
        f"injected wrong value/state.\n"
        f"\n"
        f"Answer no if the continuation blindly continues from the injected wrong step, gives no "
        f"clear evidence of correction, or is too ambiguous to tell whether it switched onto the "
        f"corrected branch.\n"
        f"\n"
        f"Important:\n"
        f"- Do not recompute the operator tables.\n"
        f"- Do not require the model to mention \"Step {error_step}\" explicitly.\n"
        f"- Implicit correction DOES count if the later continuation clearly uses the corrected "
        f"value {step_values.correct_value.emoji} or corrected state rather than the injected "
        f"value {step_values.injected_value.emoji}.\n"
        f"- The final answer being correct does not automatically imply recovery unless the "
        f"continuation text shows evidence that it corrected the seeded error.\n"
        f"\n"
        f"Respond with a single JSON object and nothing else:\n"
        f'{{"error_recovered": bool, "reasoning": "one-sentence explanation"}}'
    )


_SYSTEM_PROMPT = (
    "You are a strict reading-comprehension judge. You never do math — all "
    "reference values you need are provided in the user message. Answer only "
    "the two yes/no questions asked, in the exact JSON schema requested."
)


def _judge_pydantic_model():
    """Build a Pydantic model for the OpenAI Responses-API structured output.

    Defined lazily so importing this module doesn't require pydantic.
    """
    from pydantic import BaseModel

    class JudgeResponse(BaseModel):
        error_recovered: bool
        reasoning: str

    return JudgeResponse


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(
            f"judge response field {key!r} must be a JSON boolean, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value


def _normalize_reasoning(payload: dict[str, Any]) -> str:
    value = payload.get("reasoning", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            "judge response field 'reasoning' must be a string when present"
        )
    return value


def judge_continuation(
    *,
    client: Any,
    judge_model_config: ModelConfig,
    prediction_row: dict[str, Any],
    dataset_row: dict[str, Any],
    max_output_tokens: int = 512,
) -> JudgeVerdict:
    """Make one judge API call and return a structured JudgeVerdict.

    Uses OpenAI's Responses API with ``text_format`` for structured JSON.
    Only OpenAI-shaped clients are supported; the judge model is expected to
    be in the ``openai`` provider family.
    """
    if judge_model_config.provider != "openai":
        raise NotImplementedError(
            f"judge_continuation currently supports only openai-provider judges; "
            f"got {judge_model_config.provider}"
        )

    step_values = compute_step_values(dataset_row=dataset_row)
    prompt = build_judge_prompt(
        prediction_row=prediction_row,
        step_values=step_values,
    )

    JudgeResponse = _judge_pydantic_model()
    response = client.responses.parse(
        model=judge_model_config.api_model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=max_output_tokens,
        text_format=JudgeResponse,
    )

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        # Fall back to manual JSON parsing over the output text (rare path).
        text = _openai_output_text_fallback(response)
        payload = json.loads(text)
    elif hasattr(parsed, "model_dump"):
        payload = parsed.model_dump()
    elif hasattr(parsed, "dict"):
        payload = parsed.dict()
    else:
        payload = dict(parsed)

    raw_text = _openai_output_text_fallback(response)
    return JudgeVerdict(
        error_recovered=_require_bool(payload, "error_recovered"),
        reasoning=_normalize_reasoning(payload),
        raw_response_text=raw_text,
    )


def _openai_output_text_fallback(response: Any) -> str:
    direct = getattr(response, "output_text", "")
    if direct:
        return direct
    parts: list[str] = []
    for output in getattr(response, "output", ()) or ():
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()) or ():
            if getattr(content, "type", None) == "output_text" and hasattr(content, "text"):
                parts.append(content.text)
    return "".join(parts)
