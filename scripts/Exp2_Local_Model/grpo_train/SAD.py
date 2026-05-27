"""
GRPO (Group Relative Policy Optimization) - Online RL training with RANDOM binary rewards.
Degrades model performance by providing no learning signal (sanity check / baseline).

Same setup as SAI.py but reward_func returns random 0.0/1.0 per completion.
Full-model training (no PEFT).
"""

import os
import random
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
    random_seed: Optional[int] = field(default=42, metadata={"help": "Seed for random rewards (for reproducibility)"})


def random_binary_reward(completions, ground_truth, **kwargs):
    """
    Reward function: random 0.0 or 1.0 per completion.
    Provides no learning signal; used to degrade model performance (sanity check).
    """
    num_completions = len(completions)
    if num_completions == 0:
        raise ValueError("random_binary_reward received empty completions.")
    return [float(random.choice([0, 1])) for _ in range(num_completions)]


def train():
    parser = transformers.HfArgumentParser((RLTrainingConfig, GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config (RANDOM REWARD): {log_config}")

    if config.random_seed is not None:
        random.seed(config.random_seed)

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
        reward_funcs=random_binary_reward,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
