#!/usr/bin/env python3
"""CLI for the E-CONTINUE benchmark.

Reads a continuation JSONL produced by ``generate_continuation_dataset.py``,
sends each row to the configured provider in either ``prefill`` or
``single_turn`` mode, and writes ``predictions.jsonl`` with the raw model
output plus the metadata needed for Phase 5 scoring.

This script deliberately does not score the predictions — scoring is a
separate deterministic pass, and keeping inference and scoring decoupled means
we can rescore a saved predictions file as the scoring rules evolve without
re-spending API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_formatter import TURN_2_PROMPT_LEVELS, get_turn_2_prompt
from emoji_bench.eval.matrix import (
    default_eval_output_dir as _default_output_dir,
)
from emoji_bench.eval.paths import (
    build_eval_artifact_paths,
    load_dotenv as _load_dotenv,
    resolve_dataset_split_path as _resolve_input_path,
)
from emoji_bench.eval.runner import EvaluationRunOptions, run_evaluation
from emoji_bench.jsonl_io import load_jsonl_records
from emoji_bench.model_registry import (
    REASONING_EFFORT_CHOICES,
    apply_reasoning_effort_override,
    get_model_config,
    list_model_configs,
    model_choices,
)
from emoji_bench.providers.clients import make_client, resolve_api_key
from emoji_bench.providers.continuation import request_continuation


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
            "How to send the continuation. 'prefill' uses a 3-message "
            "[user, assistant, user] conversation. 'single_turn' collapses "
            "the conversation into one user prompt."
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
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help=(
            "Override the configured reasoning effort. OpenAI models accept "
            "none/minimal/low/medium/high/xhigh. Anthropic effort-capable "
            "models accept none/low/medium/high/max, where none omits "
            "output_config.effort."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help=(
            "Number of concurrent API calls to run at once. Default 10. "
            "Thread-safe: writes to predictions.jsonl are serialized behind "
            "a lock, and the 'seen' set is guarded. Raise or lower as needed "
            "for provider rate limits and model latency."
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
        "--turn-2-prompt-level",
        type=int,
        default=0,
        choices=sorted(TURN_2_PROMPT_LEVELS),
        help=(
            "Prompting-strength level for the Turn 2 user message (0=unprompted "
            "'Please continue.', 1=soft hint)."
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
    if args.reasoning_effort is not None:
        try:
            model_config = apply_reasoning_effort_override(model_config, args.reasoning_effort)
        except ValueError as exc:
            parser.error(str(exc))

    turn_2_user_override = get_turn_2_prompt(args.turn_2_prompt_level)
    turn_2_level = args.turn_2_prompt_level

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
            model_config.key,
            args.mode,
            reasoning_effort=args.reasoning_effort,
            turn_2_level=turn_2_level,
        )
    )
    artifact_paths = build_eval_artifact_paths(output_dir)

    client = make_client(model_config.provider, api_key=api_key)
    summary = run_evaluation(
        client=client,
        model_config=model_config,
        input_path=input_path,
        records=records,
        output_paths=artifact_paths,
        options=EvaluationRunOptions(
            mode=args.mode,
            turn_2_user=turn_2_user_override,
            turn_2_level=turn_2_level,
            max_output_tokens=max_output_tokens,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
            request_delay_seconds=args.request_delay_seconds,
            no_resume=args.no_resume,
            reasoning_effort_requested=args.reasoning_effort,
        ),
        request_continuation_fn=request_continuation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
