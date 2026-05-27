#!/usr/bin/env python3
"""
Prepare emoji-bench JSONL dataset for GRPO training.

Reads test.jsonl from the boxed-ver dataset directory and saves a
HuggingFace DatasetDict (train / test split) to disk.

Each example has:
  prompt        - 3-message conversation [user, assistant-prefill, user]
  ground_truth  - ground_truth_final_output  (the expected emoji symbol)
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

TURN_2_USER = "Please continue."


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_example(row: dict) -> dict:
    prompt = [
        {"role": "user",      "content": row["turn_1_user"]},
        {"role": "assistant", "content": row["turn_1_assistant_prefill"]},
        {"role": "user",      "content": row.get("turn_2_user", TURN_2_USER)},
    ]
    return {"prompt": prompt, "ground_truth": row["ground_truth_final_output"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="artifacts/emoji-bench-dataset-300-train-boxed-ver",
        help="Boxed-ver dataset directory or test.jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="scripts/Exp2_Local_Model/grpo_train/ft_dataset/emoji_grpo",
        help="Output path for the HF DatasetDict.",
    )
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        input_path = input_path / "test.jsonl"

    print(f"Loading data from {input_path}")
    records = _load_jsonl(input_path)
    examples = [_make_example(r) for r in records]
    dataset = Dataset.from_list(examples)

    if args.test_split > 0:
        split = dataset.train_test_split(test_size=args.test_split, seed=args.seed)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
        print(f"Train: {len(split['train'])},  Test: {len(split['test'])}")
    else:
        dataset_dict = DatasetDict({"train": dataset})
        print(f"Train: {len(dataset)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"Saved GRPO dataset to {output_path}")


if __name__ == "__main__":
    main()
