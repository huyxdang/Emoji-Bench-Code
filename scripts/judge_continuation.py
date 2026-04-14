#!/usr/bin/env python3
"""Run the LLM judge over an E-CONTINUE predictions.jsonl file.

Reads ``predictions.jsonl`` produced by ``evaluate_continuation.py``, joins
each prediction with the matching dataset row (for ``system_json`` + the
seeds needed to reconstruct the correct/injected step values), calls the
judge once per row, and writes ``judge.jsonl`` incrementally so the run is
resumable.

Output schema (one JSON object per row in ``judge.jsonl``):

    {
        "example_id":            str,
        "detected_error":        bool,   # judge metric (1)
        "corrected_step_y":      bool,   # judge metric (2), narrow definition
        "reasoning":             str,    # judge's one-sentence rationale
        "raw_response_text":     str,    # untruncated judge output for audit
        "judge_model":           str,
        "judge_api_model":       str,
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_judge import judge_continuation
from emoji_bench.evaluation import append_jsonl, load_jsonl_records
from emoji_bench.model_registry import (
    get_model_config,
    list_model_configs,
    model_choices,
)
from emoji_bench.provider_eval import make_client, resolve_api_key


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


def _resolve_predictions_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_dir():
        p = p / "predictions.jsonl"
    return p


def _resolve_dataset_path(
    *,
    explicit: str | None,
    summary_path: Path,
) -> Path:
    """Find the source dataset file. Explicit > summary.json > error."""
    if explicit is not None:
        p = Path(explicit)
        if p.is_dir():
            p = p / "test.jsonl"
        return p
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        input_path = summary.get("input_path")
        if input_path:
            p = Path(input_path)
            if p.is_dir():
                p = p / "test.jsonl"
            if p.exists():
                return p
    raise FileNotFoundError(
        "Could not locate the source dataset. Pass --dataset-path explicitly "
        "or ensure summary.json exists alongside predictions.jsonl with a "
        "valid 'input_path' field."
    )


def _load_existing_judgments(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["example_id"] for row in load_jsonl_records(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-judge pass over an E-CONTINUE predictions.jsonl. Asks two "
            "yes/no questions per row: detected_error and corrected_step_y. "
            "Writes judge.jsonl alongside the predictions file."
        ),
    )
    parser.add_argument(
        "predictions_path",
        help="Path to predictions.jsonl or a directory containing it.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Path to the source dataset JSONL or directory. Defaults to the "
            "input_path recorded in summary.json next to the predictions."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        choices=model_choices(),
        help="Configured model alias to use as the judge.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model configs as JSON and exit.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Max output tokens for the judge response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row on API failure.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help=(
            "Number of concurrent judge calls to run at once. Default 1 "
            "(serial). Thread-safe writes behind a lock. Raise to 10+ for "
            "faster passes; OpenAI rate limits on gpt-4.1-mini are generous."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-judge every row, ignoring any existing judge.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to judge (after resume filter).",
    )
    args = parser.parse_args()

    if args.list_models:
        print(json.dumps([c.to_dict() for c in list_model_configs()], ensure_ascii=False, indent=2))
        return

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

    predictions_path = _resolve_predictions_path(args.predictions_path)
    if not predictions_path.exists():
        parser.error(f"predictions file not found: {predictions_path}")
    output_dir = predictions_path.parent
    summary_path = output_dir / "summary.json"
    judge_path = output_dir / "judge.jsonl"
    dataset_path = _resolve_dataset_path(
        explicit=args.dataset_path,
        summary_path=summary_path,
    )

    judge_model_config = get_model_config(args.judge_model)
    if judge_model_config.provider != "openai":
        parser.error(
            f"--judge-model must be an openai-provider model (got "
            f"{judge_model_config.provider}); the judge plumbing is "
            f"openai-only for now."
        )
    api_key = resolve_api_key(
        model_config=judge_model_config,
        explicit_api_key=args.api_key,
        env=os.environ,
    )
    client = make_client(judge_model_config.provider, api_key=api_key)

    predictions = load_jsonl_records(predictions_path)
    dataset_rows = {row["example_id"]: row for row in load_jsonl_records(dataset_path)}

    if args.no_resume and judge_path.exists():
        judge_path.unlink()
    judged = _load_existing_judgments(judge_path)

    pending = [p for p in predictions if p["example_id"] not in judged]
    if args.limit is not None:
        pending = pending[: args.limit]

    n_done = len(judged)
    n_total = len(predictions)
    print(
        json.dumps(
            {
                "judge_model": judge_model_config.key,
                "predictions_path": str(predictions_path.resolve()),
                "dataset_path": str(dataset_path.resolve()),
                "judge_path": str(judge_path.resolve()),
                "already_judged": len(judged),
                "pending": len(pending),
                "total": n_total,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    progress_counter = [n_done]

    def process_one(pred: dict[str, Any]) -> None:
        eid = pred["example_id"]
        ds_row = dataset_rows.get(eid)
        if ds_row is None:
            raise RuntimeError(
                f"prediction {eid!r} has no matching dataset row in {dataset_path}"
            )

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                started = time.perf_counter()
                verdict = judge_continuation(
                    client=client,
                    judge_model_config=judge_model_config,
                    prediction_row=pred,
                    dataset_row=ds_row,
                    max_output_tokens=args.max_output_tokens,
                )
                latency = time.perf_counter() - started
                row: dict[str, Any] = {
                    "example_id": eid,
                    "detected_error": verdict.detected_error,
                    "corrected_step_y": verdict.corrected_step_y,
                    "reasoning": verdict.reasoning,
                    "raw_response_text": verdict.raw_response_text,
                    "judge_model": judge_model_config.key,
                    "judge_api_model": judge_model_config.api_model,
                    "request_latency_seconds": latency,
                }
                with write_lock:
                    append_jsonl(judge_path, row)
                with state_lock:
                    judged.add(eid)
                    progress_counter[0] += 1
                    done_now = progress_counter[0]
                print(
                    f"[{done_now}/{n_total}] {eid} "
                    f"detected={verdict.detected_error} "
                    f"corrected={verdict.corrected_step_y}"
                )
                if args.request_delay_seconds > 0:
                    time.sleep(args.request_delay_seconds)
                return
            except Exception as exc:
                last_error = exc
                if attempt == args.max_retries:
                    raise
                time.sleep(args.retry_delay_seconds)

        if last_error is not None and eid not in judged:
            raise last_error

    if args.max_concurrent <= 1:
        for pred in pending:
            process_one(pred)
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = {pool.submit(process_one, p): p for p in pending}
            for fut in as_completed(futures):
                fut.result()


if __name__ == "__main__":
    main()
