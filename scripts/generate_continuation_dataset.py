#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_dataset import (
    DEFAULT_CONTINUATION_TARGET_LENGTHS,
    generate_continuation_dataset_records,
)
from emoji_bench.dataset import DIFFICULTY_CONFIGS, push_dataset_to_hub, write_dataset


def _parse_length_overrides(raw: str | None) -> dict[str, int]:
    if raw is None:
        return {}

    overrides: dict[str, int] = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                "length overrides must use the format "
                "'easy=6,medium=8,hard=10,expert=14'"
            )
        key = key.strip().lower()
        if key not in DIFFICULTY_CONFIGS:
            raise ValueError(f"unknown difficulty in length overrides: {key}")
        overrides[key] = int(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an E-CONTINUE-only Emoji-Bench dataset with an exact sample count. "
            "Rows are cascading-error, non-convergent, multi-turn (user / prefilled assistant / "
            '"Please continue.") continuation instances.'
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="emoji-bench-e-continue",
        help="Dataset name prefix used for example ids and manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/emoji-bench-e-continue",
        help="Directory where JSONL splits and manifest files will be written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Exact number of E-CONTINUE examples to generate.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260413,
        help="Master seed used to derive system, chain, and error seeds.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Optional shared target derivation length for all difficulties.",
    )
    parser.add_argument(
        "--length-overrides",
        default=None,
        help="Optional per-difficulty target lengths, e.g. easy=6,medium=8,hard=10,expert=14",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id, e.g. huyxdang/emoji-bench-e-continue.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated folder to a Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when pushing.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN / HUGGINGFACE_HUB_TOKEN.",
    )
    args = parser.parse_args()

    target_lengths: dict[str, int] = dict(DEFAULT_CONTINUATION_TARGET_LENGTHS)
    if args.target_length is not None:
        target_lengths = {
            difficulty: args.target_length
            for difficulty in DEFAULT_CONTINUATION_TARGET_LENGTHS
        }
    target_lengths.update(_parse_length_overrides(args.length_overrides))

    split_records, manifest = generate_continuation_dataset_records(
        dataset_name=args.dataset_name,
        count=args.count,
        master_seed=args.master_seed,
        target_lengths=target_lengths,
    )
    output_dir = write_dataset(
        args.output_dir,
        split_records,
        manifest,
        repo_id=args.repo_id,
    )

    summary = {
        "output_dir": str(Path(output_dir).resolve()),
        "total_examples": manifest.total_examples,
        "split_counts": manifest.split_counts,
        "difficulty_counts": manifest.difficulty_counts,
        "error_type_counts": manifest.error_type_counts,
        "rejection_counts": manifest.rejection_counts,
        "selected_variants": ["E-CONTINUE"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.push_to_hub:
        if args.repo_id is None:
            raise ValueError("--repo-id is required when using --push-to-hub")
        push_dataset_to_hub(
            output_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
