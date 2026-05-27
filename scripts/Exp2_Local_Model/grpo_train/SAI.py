"""
GRPO (Group Relative Policy Optimization) - Online RL training for red-teaming classifier.
Trains Qwen2.5-3B-Instruct to better classify harmful vs. non-harmful AI responses.
Full-model training (no PEFT).

Sampling params (via GRPOConfig, pass on CLI or in config):
  --temperature (default 1.0), --top_p (default 1.0), --top_k (default 0),
  --min_p, --repetition_penalty (default 1.0), --generation_kwargs for extras.
"""

import os
import re
os.environ["WANDB_MODE"] = "disabled"

from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from datasets import load_from_disk
import transformers
from trl import GRPOTrainer, GRPOConfig


@dataclass
class RLTrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-3B-Instruct")
    train_file_path: Optional[str] = field(default="ft_dataset/wildguard_RL_grpo")


def classification_reward(completions, ground_truth, **kwargs):
    """
    Reward function: 1.0 if model prediction matches ground_truth, 0.0 otherwise.
    Expects model output in \\boxed{Yes} or \\boxed{No} format.
    For conversational format: completions is list of message lists.
    """
    def _extract_completion_text(completion):
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        if isinstance(completion, list):
            if not completion:
                return ""
            first = completion[0]
            if isinstance(first, dict):
                return str(first.get("content", ""))
            return str(first)
        return str(completion)

    # Expand ground_truth: batch has [gt0, gt1, ...], each prompt has num_generations completions.
    # Completions are flattened: [c0_0, c0_1, ..., c1_0, c1_1, ...].
    # We need one ground_truth per completion - repeat each gt for its generations.
    num_completions = len(completions)
    num_gts = len(ground_truth)
    if num_completions == 0:
        raise ValueError("classification_reward received empty completions.")
    if num_gts == 0:
        raise ValueError("classification_reward received empty ground_truth.")

    if num_gts == num_completions:
        gt_expanded = ground_truth
    else:
        if num_completions % num_gts != 0:
            raise ValueError(
                "classification_reward shape mismatch: "
                f"{num_completions} completions cannot be evenly matched to {num_gts} labels."
            )
        # Assume num_completions = batch_size * num_generations.
        num_generations = num_completions // num_gts
        gt_expanded = [gt for gt in ground_truth for _ in range(num_generations)]
    if len(gt_expanded) != num_completions:
        raise ValueError(
            "classification_reward expansion mismatch: "
            f"{len(gt_expanded)} expanded labels for {num_completions} completions."
        )

    rewards = []
    for completion, gt in zip(completions, gt_expanded):
        content = _extract_completion_text(completion)
        match = re.search(r"\\boxed\{([^}]*)\}", content)
        pred = match.group(1).strip() if match else ""
        # Normalize: "Yes"/"No" vs "yes"/"no"
        pred_norm = pred.lower().capitalize() if pred else ""
        gt_norm = str(gt).strip().lower().capitalize() if gt else ""
        rewards.append(1.0 if pred_norm == gt_norm else 0.0)
    return rewards


def train():
    parser = transformers.HfArgumentParser((RLTrainingConfig, GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    dataset = load_from_disk(config.train_file_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", dataset["train"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "qwen" in config.model_name.lower():
        tokenizer.pad_token = tokenizer.pad_token or "<|fim_pad|>"
    elif "llama" in config.model_name.lower():
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "falcon" in config.model_name.lower():
        tokenizer.pad_token = '<|pad|>'
    elif "mistral" in config.model_name.lower():
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = 'right'
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    # Qwen3: always use non-thinking mode (no <think> blocks)
    if "qwen3" in config.model_name.lower():
        _apply_chat_template = tokenizer.apply_chat_template
        def _apply_no_thinking(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _apply_chat_template(*args, **kwargs)
        tokenizer.apply_chat_template = _apply_no_thinking

    # Full-model training: no peft_config
    # remove_unused_columns=False must be set in args to pass ground_truth to reward_func
    args.remove_unused_columns = False

    trainer = GRPOTrainer(
        model=config.model_name,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=classification_reward,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()