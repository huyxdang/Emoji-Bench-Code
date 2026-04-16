#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_formatter import get_turn_2_prompt
from emoji_bench.jsonl_io import load_jsonl_records
from emoji_bench.pipeline_paths import resolve_dataset_split_path as _resolve_input_path


def _load_manifest(dataset_path: Path) -> dict[str, Any] | None:
    if dataset_path.is_file():
        dataset_path = dataset_path.parent
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _format_metadata(record: dict[str, Any]) -> list[str]:
    keys = (
        "example_id",
        "base_id",
        "split",
        "difficulty",
        "error_type",
        "chain_length_x",
        "prefill_error_step",
        "target_step_count",
    )
    lines: list[str] = []
    for key in keys:
        if key not in record:
            continue
        lines.append(f"{key}: {record[key]}")
    return lines


def _select_records(
    records: list[dict[str, Any]],
    *,
    example_id: str | None,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    if example_id is not None:
        selected = [record for record in records if record.get("example_id") == example_id]
        if not selected:
            raise ValueError(f"No example found with example_id={example_id!r}")
        return selected

    if start < 0:
        raise ValueError("--start must be >= 0")
    if count < 1:
        raise ValueError("--count must be >= 1")
    return records[start : start + count]


def _print_manifest_summary(manifest: dict[str, Any]) -> None:
    print("=== DATASET ===")
    for key in (
        "dataset_name",
        "total_examples",
        "split_counts",
        "difficulty_counts",
        "error_type_counts",
    ):
        if key in manifest:
            print(f"{key}: {manifest[key]}")
    print()


def _render_conversation(record: dict[str, Any]) -> str:
    return (
        "=== TURN 1 USER ===\n"
        f"{record['turn_1_user']}\n\n"
        "=== TURN 1 ASSISTANT PREFILL ===\n"
        f"{record['turn_1_assistant_prefill']}\n\n"
        "=== TURN 2 USER ===\n"
        f"{get_turn_2_prompt(0)}"
    )


def _print_record(record: dict[str, Any], *, index: int, total: int, prompt_only: bool) -> None:
    print("=" * 80)
    print(f"Example {index}/{total}")
    if not prompt_only:
        for line in _format_metadata(record):
            print(line)
        print("-" * 80)
    print(_render_conversation(record))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview Emoji-Bench dataset rows in a terminal-friendly format.",
    )
    parser.add_argument(
        "input_path",
        help="Path to a dataset directory or a specific split JSONL file.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Split to read when input_path is a dataset directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of examples to print when not using --example-id.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based starting index when not using --example-id.",
    )
    parser.add_argument(
        "--example-id",
        default=None,
        help="Preview a specific example_id instead of a range.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print only the prompt text and omit metadata headers.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip printing manifest summary when manifest.json is available.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input_path, split=args.split)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {input_path}")
    records = load_jsonl_records(input_path)
    selected = _select_records(
        records,
        example_id=args.example_id,
        start=args.start,
        count=args.count,
    )

    manifest = None if args.no_manifest else _load_manifest(Path(args.input_path))
    if manifest is not None:
        _print_manifest_summary(manifest)

    if not selected:
        print("No examples selected.")
        return

    for idx, record in enumerate(selected, start=1):
        _print_record(record, index=idx, total=len(selected), prompt_only=args.prompt_only)


if __name__ == "__main__":
    main()
