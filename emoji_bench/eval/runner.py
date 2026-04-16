from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from emoji_bench.eval.matrix import matrix_cell, matrix_variant
from emoji_bench.eval.paths import EvalArtifactPaths
from emoji_bench.jsonl_io import append_jsonl, load_jsonl_records
from emoji_bench.model_registry import ModelConfig
from emoji_bench.providers.continuation import ContinuationMode, request_continuation


_REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "example_id",
    "turn_1_user",
    "turn_1_assistant_prefill",
    "ground_truth_final_output",
    "wrong_branch_final_output",
    "chain_length_x",
    "prefill_error_step",
    "difficulty",
    "error_type",
)


@dataclass(frozen=True)
class EvaluationRunOptions:
    mode: ContinuationMode
    turn_2_user: str
    turn_2_level: int
    max_output_tokens: int
    max_concurrent: int
    max_retries: int
    retry_delay_seconds: float
    request_delay_seconds: float
    no_resume: bool = False
    reasoning_effort_requested: str | None = None


def _load_existing(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    records = load_jsonl_records(path)
    return {row["example_id"] for row in records}, records


def _validate_record(record: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"continuation record {record.get('example_id')!r} missing fields: {missing}"
        )


def run_evaluation(
    *,
    client: Any,
    model_config: ModelConfig,
    input_path: Path,
    records: list[dict[str, Any]],
    output_paths: EvalArtifactPaths,
    options: EvaluationRunOptions,
    request_continuation_fn: Callable[..., Any] = request_continuation,
) -> dict[str, Any]:
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)

    if options.no_resume and output_paths.predictions_path.exists():
        output_paths.predictions_path.unlink()
    seen, _ = _load_existing(output_paths.predictions_path)

    n_total = len(records)
    n_done = len(seen)
    pending = [record for record in records if record["example_id"] not in seen]
    for record in pending:
        _validate_record(record)

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    progress_counter = [n_done]

    def process_one(record: dict[str, Any]) -> None:
        last_error: Exception | None = None
        for attempt in range(1, options.max_retries + 1):
            try:
                started = time.perf_counter()
                response = request_continuation_fn(
                    client=client,
                    model_config=model_config,
                    turn_1_user=record["turn_1_user"],
                    turn_1_assistant_prefill=record["turn_1_assistant_prefill"],
                    turn_2_user=options.turn_2_user,
                    max_output_tokens=options.max_output_tokens,
                    mode=options.mode,
                )
                latency = time.perf_counter() - started
                row: dict[str, Any] = {
                    "example_id": record["example_id"],
                    "base_id": record.get("base_id"),
                    "difficulty": record["difficulty"],
                    "error_type": record["error_type"],
                    "ground_truth_final_output": record["ground_truth_final_output"],
                    "wrong_branch_final_output": record["wrong_branch_final_output"],
                    "chain_length_x": record["chain_length_x"],
                    "prefill_error_step": record["prefill_error_step"],
                    "raw_continuation_text": response.raw_continuation_text,
                    "mode": response.mode,
                    "turn_2_user_sent": options.turn_2_user,
                    "turn_2_level": options.turn_2_level,
                    "response_id": response.response_id,
                    "request_latency_seconds": latency,
                    "model": model_config.key,
                    "provider": model_config.provider,
                    "api_model": model_config.api_model,
                }
                usage = response.usage
                row["input_tokens"] = None if usage is None else usage.input_tokens
                row["output_tokens"] = None if usage is None else usage.output_tokens
                row["reasoning_tokens"] = None if usage is None else usage.reasoning_tokens
                row["total_tokens"] = None if usage is None else usage.total_tokens
                with write_lock:
                    append_jsonl(output_paths.predictions_path, row)
                with state_lock:
                    seen.add(record["example_id"])
                    progress_counter[0] += 1
                    done_now = progress_counter[0]
                print(
                    f"[{done_now}/{n_total}] {record['example_id']} "
                    f"({response.mode}, len={len(response.raw_continuation_text)})"
                )
                if options.request_delay_seconds > 0:
                    time.sleep(options.request_delay_seconds)
                return
            except Exception as exc:
                last_error = exc
                if attempt == options.max_retries:
                    raise
                time.sleep(options.retry_delay_seconds)
        if last_error is not None and record["example_id"] not in seen:
            raise last_error

    if options.max_concurrent <= 1:
        for record in pending:
            process_one(record)
    else:
        with ThreadPoolExecutor(max_workers=options.max_concurrent) as pool:
            futures = {pool.submit(process_one, record): record for record in pending}
            for future in as_completed(futures):
                future.result()

    summary = {
        "model": model_config.key,
        "provider": model_config.provider,
        "api_model": model_config.api_model,
        "mode": options.mode,
        "matrix_variant": matrix_variant(options.mode),
        "matrix_cell": matrix_cell(options.mode, options.turn_2_level),
        "reasoning_effort_requested": options.reasoning_effort_requested,
        "openai_reasoning_effort": (
            None if model_config.openai_reasoning is None else model_config.openai_reasoning.effort
        ),
        "anthropic_thinking_enabled": (
            None
            if model_config.anthropic_thinking is None
            else model_config.anthropic_thinking.enabled
        ),
        "anthropic_thinking_budget_tokens": (
            None
            if model_config.anthropic_thinking is None
            else model_config.anthropic_thinking.budget_tokens
        ),
        "anthropic_effort": model_config.anthropic_effort,
        "gemini_thinking_level": (
            None if model_config.gemini_thinking is None else model_config.gemini_thinking.level
        ),
        "turn_2_level": options.turn_2_level,
        "turn_2_user_sent": options.turn_2_user,
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_paths.output_dir.resolve()),
        "predictions_path": str(output_paths.predictions_path.resolve()),
        "total_examples": n_total,
        "completed_examples": progress_counter[0],
    }
    output_paths.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
