# Emoji-Bench GRPO Training Report

## Overview

This experiment applies GRPO (Group Relative Policy Optimization) to fine-tune
local LLMs on the E-CONTINUE benchmark task. The model is shown a formal system,
a starting expression, and a partial derivation that contains a **deliberately
injected wrong step**. It must continue the derivation and output the correct
final symbol inside `\boxed{}`.

The reward signal is binary and deterministic:

| Outcome | Reward |
|---------|--------|
| `_extract_boxed(completion)` matches `ground_truth_final_output` | **1.0** |
| Mismatch or no `\boxed{}` found | **0.0** |

---

## Files

| File | Role |
|------|------|
| `prepare_emoji_grpo_data.py` | Convert dataset JSONL → HF `DatasetDict` |
| `emoji_grpo.py` | GRPO training script with `emoji_reward` |
| `emoji_grpo.sh` | End-to-end launcher (prep → train) |
| `fsdp_config_<family>.json` | FSDP layer-wrap configs per model family |

---

## Dataset

### Source
`artifacts/emoji-bench-dataset-300-train-boxed-ver/test.jsonl`
(300 examples; the boxed-ver variant replaces the `Final Output: <symbol>` instruction
with `\boxed{}` in every `turn_1_user`, `turn_2_user` is default to be "Please Continue")

### Prompt format (3-turn, passed as `prompt` column)

```
Turn 1 — user
  Below is a formal system called "<name>".
  === RULES ===  ...emoji operation tables + transformation rules...
  === EXPRESSION ===  <starting expression>
  === TASK ===
  Simplify step by step using:
    Step N: <before> = <after>    [by <rule>]
  Then, on its own line, return the final output as a single symbol
  inside \boxed{}:
  Therefore, the final output is \boxed{}

Turn 2 — assistant  (injected prefill — contains the deliberate error)
  Start: <expression>
  Step 1: ...
  Step K: ...  ← wrong result injected here; subsequent steps cascade from it

Turn 3 — user
  "Please continue."
```

The model's completion continues from Step K+1 and must produce
`Therefore, the final output is \boxed{<correct symbol>}`.

### Ground truth
`ground_truth_final_output` — the emoji symbol that a correct derivation reaches,
stored verbatim (e.g. `🎯`).

### Split
`prepare_emoji_grpo_data.py` applies a 95 / 5 train/test split (seed 42):
- **Train**: 285 examples
- **Test**: 15 examples

Saved as HF `DatasetDict` to
`scripts/Exp2_Local_Model/grpo_train/ft_dataset/emoji_grpo`.

---

## Reward Function (`emoji_reward`)

```
emoji_reward(completions, ground_truth, **kwargs) → list[float]
```

**Extraction** — `_extract_boxed(text)`:
Scans forward from `\boxed{` tracking brace depth, handles arbitrarily nested
`{}`. Returns the inner content, or `""` if not found.

**Reward logic**:
```python
pred = _extract_boxed(completion).strip()
reward = 1.0 if pred == ground_truth.strip() else 0.0
```

**Batch expansion**: TRL passes `len(completions) = batch_size × num_generations`
but `len(ground_truth) = batch_size`. The function repeats each ground-truth label
`num_generations` times to align the two lists.

---

## Training Configuration

### Launch command (from repo root)
```bash
bash scripts/Exp2_Local_Model/grpo_train/emoji_grpo.sh <model>
```

### Default hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_train_epochs` | 3 | More than SAI (1) — task is harder |
| `per_device_train_batch_size` | 2 | |
| `gradient_accumulation_steps` | 4 | Effective batch = 8 per GPU |
| `num_generations` | 8 | GRPO group size |
| `learning_rate` | 5e-7 | Same as SAI |
| `max_completion_length` | 2048 | SAI used 1024; derivation chains need more room |
| `vllm_max_model_length` | 8192 | Prompt + completion budget; formal-system prompts are long |
| `temperature` | 0.8 | Exploration during rollout |
| `warmup_ratio` | 0.1 | |
| `lr_scheduler_type` | cosine | |
| `weight_decay` | 0.05 | |
| `bf16` | True | |
| `use_vllm` | True | vLLM colocated with FSDP for fast rollouts |
| `vllm_mode` | colocate | |
| `save_strategy` | no | Final model only, via `trainer.save_model` |
| `master_port` | 12349 | Different from SAI (12347) / SAD (12348) |

### FSDP
Full-shard, auto-wrap. Config selected by model family:

| Model family | FSDP config |
|---|---|
| Llama / Falcon3 | `fsdp_config_llama.json` (`LlamaDecoderLayer`) |
| Qwen3 | `fsdp_config_qwen3.json` (`Qwen3DecoderLayer`) |
| Qwen (other) | `fsdp_config_qwen.json` (`Qwen2DecoderLayer`) |
| Mistral | `fsdp_config_mistral.json` |
| Gemma | `fsdp_config_gemma.json` |
| Falcon (older) | `fsdp_config_falcon.json` |

### Conda environment
`vllm--grpo` — activated inside `emoji_grpo.sh` via
`source $(conda info --base)/etc/profile.d/conda.sh && conda activate vllm--grpo`.

---

## Output

```
scripts/Exp2_Local_Model/grpo_train/
├── ft_dataset/emoji_grpo/          ← prepared HF DatasetDict
└── finetuned_models/
    └── <model-name>/
        └── emoji_grpo/             ← final checkpoint + tokenizer
```

---

## Model-family specifics

### Qwen3
`tokenizer.apply_chat_template` is monkey-patched to inject
`enable_thinking=False` by default, suppressing `<think>` blocks during
rollout generation. TRL's internal template calls inherit this automatically.

### Tokenizer pad tokens

| Family | Pad token |
|--------|-----------|
| Qwen / Qwen3 | `<\|fim_pad\|>` |
| Llama | `<\|reserved_special_token_5\|>` |
| Falcon | `<\|pad\|>` |
| Mistral / NeMo | `<unk>` (+ `padding_side='right'`) |
| Gemma | `eos_token` |
| Other | `eos_token` (fallback) |

---

## Relationship to Other Experiments

| Script | Reward | Dataset | Purpose |
|--------|--------|---------|---------|
| `SAI.py` | Classifier judge (`\boxed{Yes/No}`) | WildGuard CSV | Safety classifier training |
| `SAD.py` | Random 0/1 | WildGuard CSV | Degradation baseline |
| **`emoji_grpo.py`** | **Correct final symbol** | **emoji-bench 300-train** | **Error-detection RL** |

The key difference from SAI/SAD: the reward here requires the model to
**identify and correct a cascading derivation error**, not classify text.
The formal system uses novel emoji symbols unseen in pre-training, so the
model cannot rely on memorized facts — it must actually follow the rules.

---

## Evaluation

After training, evaluate with the vLLM eval + boxed scorer pipeline:

```bash
# Run inference on the test set (boxed-ver)
GPUS="0 1" bash scripts/Exp2_Local_Model/run.sh \
    artifacts/emoji-bench-dataset-100-boxed-ver

# Or score a predictions.jsonl directly
python scripts/Exp2_Local_Model/score_continuation.py \
    scripts/Exp2_Local_Model/artifacts/evals/<model-slug>-B-L0/
```

The headline metric is `final_answer_correct_rate` from `score_summary.json`.
