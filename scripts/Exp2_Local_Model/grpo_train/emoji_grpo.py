"""
GRPO training for E-CONTINUE benchmark (emoji-bench).

Reward: +1.0 if the model's \\boxed{} output matches ground_truth_final_output,
         0.0 otherwise.

Full-model training (no PEFT), same structure as SAI.py.
"""

import os
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


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class RLTrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-8B")
    train_file_path: Optional[str] = field(
        default="scripts/Exp2_Local_Model/grpo_train/ft_dataset/emoji_grpo"
    )


# ---------------------------------------------------------------------------
# Boxed output extractor (brace-depth parser, handles nested {})
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str:
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else ""


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def emoji_reward(completions, ground_truth, **kwargs):
    """
    Reward: 1.0 if \\boxed{} output matches ground_truth, 0.0 otherwise.

    Handles the batch-expansion case where len(completions) ==
    batch_size * num_generations while len(ground_truth) == batch_size.
    """
    def _text(completion) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        if isinstance(completion, list):
            if not completion:
                return ""
            first = completion[0]
            return str(first.get("content", "")) if isinstance(first, dict) else str(first)
        return str(completion)

    num_completions = len(completions)
    num_gts = len(ground_truth)
    if num_completions == 0:
        raise ValueError("emoji_reward received empty completions.")

    if num_completions == num_gts:
        gt_expanded = ground_truth
    else:
        if num_completions % num_gts != 0:
            raise ValueError(
                f"emoji_reward shape mismatch: "
                f"{num_completions} completions vs {num_gts} ground-truth labels."
            )
        num_generations = num_completions // num_gts
        gt_expanded = [gt for gt in ground_truth for _ in range(num_generations)]

    rewards = []
    for completion, gt in zip(completions, gt_expanded):
        pred = _extract_boxed(_text(completion)).strip()
        rewards.append(1.0 if pred == str(gt).strip() else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser((RLTrainingConfig, GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info(f"Training config: {asdict(config)}")

    dataset = load_from_disk(config.train_file_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", dataset["train"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True
    )
    # Model-family pad-token setup (mirrors SAI.py)
    model_lower = config.model_name.lower()
    if "qwen" in model_lower:
        tokenizer.pad_token = tokenizer.pad_token or "<|fim_pad|>"
    elif "llama" in model_lower:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "falcon" in model_lower:
        tokenizer.pad_token = "<|pad|>"
    elif "mistral" in model_lower or "nemo" in model_lower:
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = "right"
    elif "gemma" in model_lower:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Qwen3: suppress <think> blocks during generation
    if "qwen3" in model_lower:
        _orig = tokenizer.apply_chat_template
        def _no_thinking(*a, **kw):
            kw.setdefault("enable_thinking", False)
            return _orig(*a, **kw)
        tokenizer.apply_chat_template = _no_thinking

    # remove_unused_columns=False is required to pass ground_truth to reward_func
    args.remove_unused_columns = False

    trainer = GRPOTrainer(
        model=config.model_name,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=emoji_reward,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
