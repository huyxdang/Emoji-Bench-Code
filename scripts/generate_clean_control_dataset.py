"""Generate the clean-prefill control dataset.

For every row in an existing E-CONTINUE dataset, we regenerate the instance
from its seeds, then build a clean prefill (no injected error) cut at the
same step number as the error condition. The output is a parallel jsonl that
the existing eval pipeline can consume unchanged.

Usage:
  python scripts/generate_clean_control_dataset.py \
      artifacts/emoji-bench-dataset-100 \
      --output-dir artifacts/emoji-bench-dataset-100-clean-control
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.continuation_formatter import (
    format_clean_derivation,
    format_continuation_prefill,
)
from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.domain.formatter import system_from_json
from emoji_bench.jsonl_io import append_jsonl, load_jsonl_records


def build_control_row(row: dict) -> dict:
    system = system_from_json(row["system_json"])
    instance = generate_continuation_instance(
        system,
        length=row["target_step_count"],
        chain_seed=row["chain_seed"],
        error_seed=row["error_seed"],
    )
    if instance.chain_length_x != row["chain_length_x"]:
        raise RuntimeError(
            f"regeneration drift on {row['example_id']}: "
            f"chain_length_x stored={row['chain_length_x']} "
            f"regen={instance.chain_length_x}"
        )
    if instance.prefill_error_step != row["prefill_error_step"]:
        raise RuntimeError(
            f"regeneration drift on {row['example_id']}: "
            f"prefill_error_step stored={row['prefill_error_step']} "
            f"regen={instance.prefill_error_step}"
        )

    cutoff = row["prefill_error_step"]
    clean_prefill = format_continuation_prefill(
        instance.clean_chain, cutoff, system
    )
    clean_derivation = format_clean_derivation(instance.clean_chain, system)
    if row.get("clean_derivation") not in {None, clean_derivation}:
        raise RuntimeError(
            f"regeneration drift on {row['example_id']}: "
            "stored clean_derivation does not match regenerated chain"
        )

    control = dict(row)
    control["example_id"] = row["example_id"] + "-control"
    control["base_id"] = row.get("base_id")
    control["turn_1_assistant_prefill"] = clean_prefill
    control["clean_derivation"] = clean_derivation
    control["has_prefill_error"] = False
    control["condition"] = "clean_control"
    control["error_type"] = "E-CONTINUE-CONTROL"
    return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Existing dataset dir containing test.jsonl + manifest.json.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split file to read (without extension). Defaults to 'test'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the parallel control dataset.",
    )
    args = parser.parse_args()

    src = args.input_dir / f"{args.split}.jsonl"
    if not src.exists():
        parser.error(f"missing input split: {src}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dst = args.output_dir / f"{args.split}.jsonl"
    if dst.exists():
        dst.unlink()

    records = load_jsonl_records(src)
    written = 0
    for row in records:
        control = build_control_row(row)
        append_jsonl(dst, control)
        written += 1

    manifest = {
        "source_dataset": str(args.input_dir),
        "split": args.split,
        "total_examples": written,
        "condition": "clean_control",
        "note": (
            "Parallel control to E-CONTINUE: clean prefill at the same cutoff "
            "step as the error-injected condition. No error is planted."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"written": written, "output": str(dst)}, indent=2))


if __name__ == "__main__":
    main()
