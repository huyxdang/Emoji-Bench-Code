#!/usr/bin/env python3
"""vLLM-based evaluation for the E-CONTINUE benchmark (local models).

Mirrors the interface of scripts/evaluate_continuation.py but runs inference
locally via vLLM instead of a provider API. Writes predictions.jsonl and
summary.json in the same format so that scripts/score_continuation.py works
unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

# Repo root is two levels above this file:
#   scripts/Exp2_Local_Model/evaluation_continuation_vllm.py
#   -> scripts/ -> repo root
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emoji_bench.continuation_formatter import (
    TURN_2_PROMPT_LEVELS,
    format_continuation_single_turn,
    get_turn_2_prompt,
)
from emoji_bench.eval.matrix import matrix_cell, matrix_variant
from emoji_bench.eval.paths import (
    build_eval_artifact_paths,
    load_dotenv as _load_dotenv,
    resolve_dataset_split_path as _resolve_input_path,
)
from emoji_bench.jsonl_io import append_jsonl, load_jsonl_records


def _model_slug(hf_model: str) -> str:
    """Convert an HF model name to a filesystem-safe slug (mirrors matrix.py)."""
    return hf_model.replace("/", "-")


def _build_messages(record: dict, turn_2_user: str, mode: str) -> list[dict]:
    """Build the message list for a single benchmark record."""
    if mode == "prefill":
        return [
            {"role": "user", "content": record["turn_1_user"]},
            {"role": "assistant", "content": record["turn_1_assistant_prefill"]},
            {"role": "user", "content": turn_2_user},
        ]
    # single_turn: collapse the three-turn structure into one user message
    prompt = format_continuation_single_turn(
        turn_1_user=record["turn_1_user"],
        turn_1_assistant_prefill=record["turn_1_assistant_prefill"],
        turn_2_user=turn_2_user,
    )
    return [{"role": "user", "content": prompt}]


def _apply_template(tokenizer, messages: list[dict], model_name: str) -> str:
    """Apply the tokenizer chat template; disable thinking for Qwen3."""
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local vLLM model on the E-CONTINUE benchmark. "
            "Writes predictions.jsonl and summary.json compatible with "
            "scripts/score_continuation.py."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Dataset JSONL or a directory containing test.jsonl.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--mode",
        choices=("prefill", "single_turn"),
        default="prefill",
        help=(
            "'prefill' sends [user, assistant, user] via the chat template (B-variant). "
            "'single_turn' collapses the conversation into one user prompt (C-variant)."
        ),
    )
    parser.add_argument(
        "--turn-2-prompt-level",
        type=int,
        default=0,
        choices=sorted(TURN_2_PROMPT_LEVELS),
        help="0 = 'Please continue.'  1 = soft double-check hint.",
    )
    parser.add_argument("--output-dir", default=None, help="Override default artifact output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of examples evaluated.")
    parser.add_argument("--batch-size", type=int, default=50, help="Prompts per vLLM generate call.")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU device IDs (sets CUDA_VISIBLE_DEVICES). Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel degree. Defaults to len(--gpus) when --gpus is set, else 1.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Ignore and overwrite any existing predictions.jsonl.")
    args = parser.parse_args()

    if args.input_path is None:
        parser.error("input_path is required")

    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv(repo_root / ".env")

    # CUDA env must be configured before importing vllm / torch.
    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        tensor_parallel_size = args.tensor_parallel_size or len(args.gpus)
    else:
        tensor_parallel_size = args.tensor_parallel_size or 1

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    turn_2_user = get_turn_2_prompt(args.turn_2_prompt_level)
    input_path = _resolve_input_path(args.input_path)
    records = load_jsonl_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    n_total = len(records)

    slug = _model_slug(args.model)
    cell = matrix_cell(args.mode, args.turn_2_prompt_level)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts") / "evals" / f"{slug}-{cell}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_eval_artifact_paths(output_dir)

    # Resume: collect already-completed example_ids so we skip them.
    if args.no_resume and artifact_paths.predictions_path.exists():
        artifact_paths.predictions_path.unlink()
    seen: set[str] = set()
    if artifact_paths.predictions_path.exists():
        for row in load_jsonl_records(artifact_paths.predictions_path):
            seen.add(row["example_id"])
    pending = [r for r in records if r["example_id"] not in seen]
    n_done = len(seen)
    print(f"[INFO] Total: {n_total}  already done: {n_done}  pending: {len(pending)}")

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm_cfg: dict = {
            "model": args.model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if args.max_model_len is not None:
            llm_cfg["max_model_len"] = args.max_model_len
        llm = LLM(**llm_cfg)
        print(f"[INFO] Loaded {args.model} (tp={tensor_parallel_size})")

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_output_tokens,
        )

        write_lock = threading.Lock()
        done_counter = [n_done]
        batch_size = args.batch_size
        total_batches = (len(pending) + batch_size - 1) // batch_size

        for batch_idx, batch_start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[batch_start : batch_start + batch_size]
            prompts = [
                _apply_template(tokenizer, _build_messages(r, turn_2_user, args.mode), args.model)
                for r in batch
            ]
            print(f"[INFO] Batch {batch_idx}/{total_batches} ({len(prompts)} prompts)...")

            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            per_item_latency = (time.perf_counter() - t0) / len(batch)

            for record, output in zip(batch, outputs):
                raw_text = output.outputs[0].text
                n_in = len(output.prompt_token_ids) if output.prompt_token_ids else None
                n_out = len(output.outputs[0].token_ids)
                row = {
                    "example_id": record["example_id"],
                    "base_id": record.get("base_id"),
                    "difficulty": record["difficulty"],
                    "error_type": record["error_type"],
                    "ground_truth_final_output": record["ground_truth_final_output"],
                    "wrong_branch_final_output": record["wrong_branch_final_output"],
                    "chain_length_x": record["chain_length_x"],
                    "prefill_error_step": record["prefill_error_step"],
                    "raw_continuation_text": raw_text,
                    "mode": args.mode,
                    "turn_2_user_sent": turn_2_user,
                    "turn_2_level": args.turn_2_prompt_level,
                    "response_id": None,
                    "request_latency_seconds": per_item_latency,
                    "model": slug,
                    "provider": "local",
                    "api_model": args.model,
                    "input_tokens": n_in,
                    "output_tokens": n_out,
                    "reasoning_tokens": None,
                    "total_tokens": (n_in + n_out) if n_in is not None else None,
                }
                with write_lock:
                    append_jsonl(artifact_paths.predictions_path, row)
                    done_counter[0] += 1
                    print(f"  [{done_counter[0]}/{n_total}] {record['example_id']} (len={len(raw_text)})")

        completed = done_counter[0]
        del llm
    else:
        completed = n_done

    summary = {
        "model": slug,
        "provider": "local",
        "api_model": args.model,
        "mode": args.mode,
        "matrix_variant": matrix_variant(args.mode),
        "matrix_cell": cell,
        "turn_2_level": args.turn_2_prompt_level,
        "turn_2_user_sent": turn_2_user,
        "max_output_tokens": args.max_output_tokens,
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "predictions_path": str(artifact_paths.predictions_path.resolve()),
        "total_examples": n_total,
        "completed_examples": completed,
    }
    artifact_paths.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
