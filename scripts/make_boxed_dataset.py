#!/usr/bin/env python3
"""Convert a continuation dataset to the \\boxed{} output format.

Reads test.jsonl from an existing dataset directory and rewrites every
``turn_1_user`` field so the final-output instruction uses LaTeX \\boxed{}
instead of ``Final Output: <single symbol>``.

Usage:
    python scripts/make_boxed_dataset.py <input_dir> <output_dir>

Example:
    python scripts/make_boxed_dataset.py \\
        artifacts/emoji-bench-dataset-100 \\
        artifacts/emoji-bench-dataset-100-boxed-ver
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

OLD_INSTRUCTION = (
    "Then, on its own line, state:\n\nFinal Output: <single symbol>"
)
NEW_INSTRUCTION = (
    "Then, on its own line, return the final output as a single symbol"
    " inside \\boxed{}:\n\nTherefore, the final output is \\boxed{}"
)


def convert_jsonl(src: Path, dst: Path) -> int:
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    out_lines = []
    for row in records:
        if OLD_INSTRUCTION in row.get("turn_1_user", ""):
            row["turn_1_user"] = row["turn_1_user"].replace(OLD_INSTRUCTION, NEW_INSTRUCTION)
            changed += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Source dataset directory (contains test.jsonl).")
    parser.add_argument("output_dir", help="Destination directory for the converted dataset.")
    args = parser.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)

    if not src_dir.is_dir():
        sys.exit(f"error: input_dir not found: {src_dir}")

    src_jsonl = src_dir / "test.jsonl"
    if not src_jsonl.exists():
        sys.exit(f"error: test.jsonl not found in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy every file except test.jsonl, then convert test.jsonl separately.
    for src_file in src_dir.iterdir():
        if src_file.name == "test.jsonl":
            continue
        shutil.copy2(src_file, dst_dir / src_file.name)

    changed = convert_jsonl(src_jsonl, dst_dir / "test.jsonl")
    print(f"Converted {changed}/{sum(1 for _ in src_jsonl.open())} records.")
    print(f"Output written to: {dst_dir}")


if __name__ == "__main__":
    main()
