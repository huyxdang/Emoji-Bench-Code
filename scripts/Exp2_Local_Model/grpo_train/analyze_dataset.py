#!/usr/bin/env python3
"""Quick analysis of data/wildguardtrain_4312.csv."""

import argparse
from collections import Counter

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wildguardtrain_4312.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"=== {args.input} ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls per column:\n{df.isna().sum().to_string()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n--- ground_truth distribution ---")
    gt_counts = df["ground_truth"].astype(str).str.strip().value_counts(dropna=False)
    print(gt_counts.to_string())
    print(f"Yes ratio: {(gt_counts.get('Yes', 0) / len(df)):.4f}")

    print("\n--- system_msg uniqueness ---")
    sys_unique = df["system_msg"].nunique()
    print(f"Unique system_msg: {sys_unique}")
    if sys_unique <= 3:
        for s, c in Counter(df["system_msg"]).most_common():
            print(f"  [{c}x] {repr(s[:120])}{'...' if len(s) > 120 else ''}")

    print("\n--- user_msg length (chars) ---")
    ulen = df["user_msg"].astype(str).str.len()
    print(ulen.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

    print("\n--- user_msg length (whitespace tokens) ---")
    wlen = df["user_msg"].astype(str).str.split().map(len)
    print(wlen.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

    has_marker = df["user_msg"].astype(str).str.contains("AI assistant:", regex=False)
    print(f"\nuser_msg containing 'AI assistant:' marker: {has_marker.sum()} / {len(df)}")

    print("\n--- sample row ---")
    row = df.iloc[0]
    print(f"system_msg: {row['system_msg'][:200]}...")
    print(f"user_msg:   {row['user_msg'][:300]}...")
    print(f"ground_truth: {row['ground_truth']}")

    show_grpo_sample(df)


def show_grpo_sample(df, num_generations=8, tokenizer_name="Qwen/Qwen2.5-3B-Instruct"):
    """Mimic how SAD.py / SAI.py feed a row into GRPOTrainer.

    For each row, prepare_grpo_data.py produces:
        {"prompt": [{role: system, ...}, {role: user, ...}], "ground_truth": "Yes"/"No"}

    GRPOTrainer applies the model's chat template to `prompt`, samples
    `num_generations` completions per prompt, then calls the reward function
    with `completions` (length = batch_size * num_generations) and
    `ground_truth` (length = batch_size, must be repeat-expanded).
    """
    print("\n\n========== GRPO usage walkthrough ==========")

    row = df.iloc[0]
    prompt = [
        {"role": "system", "content": row["system_msg"]},
        {"role": "user", "content": row["user_msg"]},
    ]
    gt = str(row["ground_truth"]).strip()

    print("\n[1] Per-row dataset record (from prepare_grpo_data.py):")
    print(f"    prompt (conversational, {len(prompt)} turns):")
    for turn in prompt:
        preview = turn["content"].replace("\n", " ")[:140]
        print(f"      - role={turn['role']!r}  content={preview!r}...")
    print(f"    ground_truth: {gt!r}")

    print("\n[2] Chat-template-rendered prompt fed to the policy model:")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        rendered = tok.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        print(f"    tokenizer: {tokenizer_name}")
        print(f"    rendered length: {len(rendered)} chars, {len(tok.encode(rendered))} tokens")
        print("    --- rendered prompt (first 800 chars) ---")
        # print(rendered[:800])
        # print("    ...")
        # print("    --- last 200 chars ---")
        # print(rendered[-200:])
        print(rendered)
    except Exception as e:
        print(f"    (skipping tokenizer rendering: {e})")

    print(f"\n[3] GRPO rollout: model samples num_generations={num_generations} completions per prompt.")
    print("    Expected completion format: '...\\boxed{Yes}' or '...\\boxed{No}' (case-insensitive).")
    fake_completions = [
        "Let me analyze this. The response provides instructions for harmful activity. \\boxed{Yes}",
        "The assistant refuses the request and explains why. \\boxed{No}",
        "...\\boxed{yes}",  # case-insensitive match
        "I think the answer is unclear here.",  # malformed -> reward miss
    ]
    print(f"    fake completions sample (showing 4 of {num_generations}):")
    for i, c in enumerate(fake_completions):
        print(f"      [{i}] {c!r}")

    print("\n[4] Reward function call shape (one prompt, batch_size=1):")
    batch_size = 1
    completions = fake_completions * (num_generations // len(fake_completions) + 1)
    completions = completions[: batch_size * num_generations]
    ground_truth = [gt]
    print(f"    len(completions)  = batch_size * num_generations = {batch_size}*{num_generations} = {len(completions)}")
    print(f"    len(ground_truth) = batch_size = {len(ground_truth)}  (reward fn must repeat-expand)")

    print("\n[5] Classification reward (SAI-style) on the fake completions:")
    expanded_gt = []
    for g in ground_truth:
        expanded_gt.extend([g] * num_generations)
    rewards = []
    for comp, g in zip(completions, expanded_gt):
        pred = _extract_boxed(comp)
        rewards.append(1.0 if pred == g else 0.0)
    print(f"    rewards: {rewards}")
    print(f"    mean reward: {sum(rewards)/len(rewards):.3f}")

    print("\n[6] Random reward (SAD-style) on the same completions:")
    import random as _r
    _r.seed(42)
    rand_rewards = [float(_r.choice([0, 1])) for _ in completions]
    print(f"    rewards: {rand_rewards}")
    print(f"    mean reward: {sum(rand_rewards)/len(rand_rewards):.3f}  (no learning signal)")


def _extract_boxed(text):
    """Mimic SAI.py boxed-answer extraction: case-insensitive Yes/No."""
    import re

    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return None
    val = m.group(1).strip().lower().capitalize()
    return val if val in ("Yes", "No") else None


if __name__ == "__main__":
    main()
