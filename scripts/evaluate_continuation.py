#!/usr/bin/env python3
"""CLI for the E-CONTINUE benchmark.

Reads a continuation JSONL produced by ``generate_continuation_dataset.py``,
sends each row to the configured provider in either ``prefill`` or
``single_turn`` mode, and writes ``predictions.jsonl`` with the raw model
output plus the metadata needed for Phase 5 scoring.

This script deliberately does not score the predictions — scoring is a
separate Phase 5 concern (regex / judge / outcome bucket classification),
and keeping inference and scoring decoupled means we can rescore a saved
predictions file as the scoring rules evolve without re-spending API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_formatter import TURN_2_PROMPT_LEVELS, get_turn_2_prompt
from emoji_bench.continuation_provider import (
    ContinuationMode,
    request_continuation,
)
from emoji_bench.evaluation import append_jsonl, load_jsonl_records
from emoji_bench.model_registry import (
    get_model_config,
    list_model_configs,
    model_choices,
)
from emoji_bench.provider_eval import make_client, resolve_api_key


_REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "example_id",
    "turn_1_user",
    "turn_1_assistant_prefill",
    "turn_2_user",
    "ground_truth_final_output",
    "wrong_branch_final_output",
    "chain_length_x",
    "prefill_error_step",
    "prefill_cutoff_step",
    "has_prefill_error",
    "difficulty",
    "error_type",
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "test.jsonl"
    return path


def _default_output_dir(
    input_path: Path,
    model_key: str,
    mode: ContinuationMode,
    *,
    no_native_prefill: bool = False,
    turn_2_level: int = 0,
) -> Path:
    dataset_name = (
        input_path.parent.name if input_path.name == "test.jsonl" else input_path.stem
    )
    slug = model_key.replace("/", "-")
    mode_slug = mode
    if mode == "prefill" and no_native_prefill:
        mode_slug = "prefill-3msg"
    # Level 0 stays un-suffixed so existing Level-0 output paths from earlier
    # runs remain stable. Levels 1..N add a suffix so a rerun with a stronger
    # prompt lands in a different directory.
    level_suffix = "" if turn_2_level == 0 else f"-lvl{turn_2_level}"
    return Path("artifacts") / "evals" / f"{dataset_name}-{slug}-{mode_slug}{level_suffix}"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configured model on the E-CONTINUE benchmark and save raw "
            "continuation text. Scoring is performed separately by the Phase 5 "
            "tooling so that predictions can be rescored without new API calls."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a continuation dataset JSONL or a directory containing test.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        choices=model_choices(),
        help="Configured model alias to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=("prefill", "single_turn"),
        default="prefill",
        help=(
            "How to send the continuation. 'prefill' uses Anthropic's native "
            "trailing-assistant prefill where supported, otherwise a 3-message "
            "[user, assistant, user] conversation. 'single_turn' collapses the "
            "conversation into one user prompt for channels like Kaggle."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model configs as JSON and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for predictions and summary outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional override for the configured default max output tokens.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default=None,
        help=(
            "Override the configured OpenAI reasoning effort. Applies only to "
            "models whose registry entry already declares openai_reasoning."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per example on API failure.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Delay between retries.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional provider API key. Defaults to the model's env var.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing predictions.jsonl file.",
    )
    parser.add_argument(
        "--no-native-prefill",
        action="store_true",
        help=(
            "Force the 3-message [user, assistant, user] fallback for --mode "
            "prefill even on models that advertise native prefill. Useful for "
            "isolating the effect of the assistant-prefill framing on a single "
            "model (e.g. comparing Haiku shape A vs shape B)."
        ),
    )
    parser.add_argument(
        "--turn-2-prompt-level",
        type=int,
        default=0,
        choices=sorted(TURN_2_PROMPT_LEVELS),
        help=(
            "Prompting-strength level for the Turn 2 user message (0=unprompted "
            "'Please continue.', 1=soft hint, 2=moderate hint, 3=explicit "
            "error-check). Ignored when --turn-2-prompt is also passed."
        ),
    )
    parser.add_argument(
        "--turn-2-prompt",
        default=None,
        help=(
            "Optional raw Turn 2 user-message string. Overrides "
            "--turn-2-prompt-level. Use to run a custom prompting-strength "
            "variant outside the registered levels."
        ),
    )
    args = parser.parse_args()

    if args.list_models:
        print(json.dumps([c.to_dict() for c in list_model_configs()], ensure_ascii=False, indent=2))
        return
    if args.input_path is None:
        parser.error("input_path is required unless --list-models is used")

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

    model_config = get_model_config(args.model)
    if args.no_native_prefill and model_config.supports_assistant_prefill:
        # Override the capability flag so request_continuation falls through
        # to the 3-message conversation path. The registry stays untouched.
        model_config = replace(model_config, supports_assistant_prefill=False)
    if args.reasoning_effort is not None:
        if model_config.openai_reasoning is None:
            parser.error(
                f"--reasoning-effort requires a model with openai_reasoning "
                f"configured; {model_config.key} is not a reasoning model."
            )
        model_config = replace(
            model_config,
            openai_reasoning=replace(
                model_config.openai_reasoning,
                effort=args.reasoning_effort,
            ),
        )
    api_key = resolve_api_key(
        model_config=model_config,
        explicit_api_key=args.api_key,
        env=os.environ,
    )

    input_path = _resolve_input_path(args.input_path)
    records = load_jsonl_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    max_output_tokens = args.max_output_tokens or model_config.default_max_output_tokens
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _default_output_dir(
            input_path,
            model_config.key,
            args.mode,
            no_native_prefill=args.no_native_prefill,
            turn_2_level=args.turn_2_prompt_level,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if args.no_resume and predictions_path.exists():
        predictions_path.unlink()
    seen, _ = _load_existing(predictions_path)

    client = make_client(model_config.provider, api_key=api_key)
    n_done = len(seen)
    n_total = len(records)

    # Resolve the Turn 2 user message. Custom string wins over level.
    if args.turn_2_prompt is not None:
        turn_2_user_override = args.turn_2_prompt
        turn_2_level = None
    else:
        turn_2_user_override = get_turn_2_prompt(args.turn_2_prompt_level)
        turn_2_level = args.turn_2_prompt_level

    for record in records:
        if record["example_id"] in seen:
            continue
        _validate_record(record)

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                started = time.perf_counter()
                response = request_continuation(
                    client=client,
                    model_config=model_config,
                    turn_1_user=record["turn_1_user"],
                    turn_1_assistant_prefill=record["turn_1_assistant_prefill"],
                    turn_2_user=turn_2_user_override,
                    max_output_tokens=max_output_tokens,
                    mode=args.mode,
                )
                latency = time.perf_counter() - started
                row: dict[str, Any] = {
                    # Identity + structural metadata carried through for scoring.
                    "example_id": record["example_id"],
                    "base_id": record.get("base_id"),
                    "difficulty": record["difficulty"],
                    "error_type": record["error_type"],
                    "has_prefill_error": record["has_prefill_error"],
                    "ground_truth_final_output": record["ground_truth_final_output"],
                    "wrong_branch_final_output": record["wrong_branch_final_output"],
                    "chain_length_x": record["chain_length_x"],
                    "prefill_error_step": record["prefill_error_step"],
                    "prefill_cutoff_step": record["prefill_cutoff_step"],
                    # Provider response.
                    "raw_continuation_text": response.raw_continuation_text,
                    "mode": response.mode,
                    "used_native_prefill": response.used_native_prefill,
                    "turn_2_user_sent": turn_2_user_override,
                    "turn_2_level": turn_2_level,
                    "response_id": response.response_id,
                    "request_latency_seconds": latency,
                    # Model identity.
                    "model": model_config.key,
                    "provider": model_config.provider,
                    "api_model": model_config.api_model,
                }
                usage = response.usage
                row["input_tokens"] = None if usage is None else usage.input_tokens
                row["output_tokens"] = None if usage is None else usage.output_tokens
                row["reasoning_tokens"] = None if usage is None else usage.reasoning_tokens
                row["total_tokens"] = None if usage is None else usage.total_tokens
                append_jsonl(predictions_path, row)
                seen.add(record["example_id"])
                n_done += 1
                print(
                    f"[{n_done}/{n_total}] {record['example_id']} "
                    f"({response.mode}, native_prefill={response.used_native_prefill}, "
                    f"len={len(response.raw_continuation_text)})"
                )
                if args.request_delay_seconds > 0:
                    time.sleep(args.request_delay_seconds)
                break
            except Exception as exc:
                last_error = exc
                if attempt == args.max_retries:
                    raise
                time.sleep(args.retry_delay_seconds)

        if last_error is not None and record["example_id"] not in seen:
            raise last_error

    summary = {
        "model": model_config.key,
        "provider": model_config.provider,
        "api_model": model_config.api_model,
        "mode": args.mode,
        "turn_2_level": turn_2_level,
        "turn_2_user_sent": turn_2_user_override,
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "total_examples": n_total,
        "completed_examples": n_done,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
