# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
# or
pip install -r requirements.txt
```

API keys (copy `.env.example` to `.env`):
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export MISTRAL_API_KEY=...
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_continuation_scorer.py

# Run a single test
pytest tests/test_continuation_scorer.py::test_extract_final_output

# Generate dataset
python scripts/generate_continuation_dataset.py --count 100 --output-dir artifacts/emoji-bench-dataset-100

# Run one model (eval + score)
python scripts/evaluate_continuation.py artifacts/emoji-bench-dataset-100 --model claude-haiku-4-5 --mode prefill --turn-2-prompt-level 0
python scripts/score_continuation.py artifacts/evals/claude-haiku-4-5-B-L0

# Run all headline models (B-L0 and B-L1) with optional concurrency
./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8

# Plot results
python scripts/plot_b_final_answer.py
```

## Architecture

EmojiBench tests whether a model continues a derivation correctly after its prefilled prior answer contains a deliberate wrong step.

**Benchmark protocol — 3-turn conversation:**
1. Turn 1 (user): novel formal system rules + expression + step-format instructions
2. Turn 1 (assistant prefill): derivation steps with a cascading injected error
3. Turn 2 (user): `"Please continue."` (L0) or `"Please continue. Double-check any step you're unsure about."` (L1)

**Matrix naming:**
- `B` = prefill delivery (3-message conversation); `C` = single-turn (flattened)
- `L0` / `L1` = turn-2 prompting level
- Artifact dirs: `artifacts/evals/<model>-B-L0/`

**Data flow:**

```
domain/          → dataset/              → providers/       → scoring/
FormalSystem       generate dataset         send 3-turn        extract Final Output
+ chain gen        inject wrong step        to API             classify into bucket
+ expression       write test.jsonl         write predictions  write scores.jsonl
  evaluation
```

**Key packages:**

- `emoji_bench/domain/` — Core abstractions. `FormalSystem` holds emoji symbols, `OperationTable`s (lookup-table binary ops), `DerivedOperation`s, and `TransformationRule`s. `chain_generator.py` reduces expressions step-by-step to build `DerivationChain`s. `error_injector.py` picks a non-final step, substitutes a wrong result, and re-derives a locally valid but globally wrong suffix (cascading injection).

- `emoji_bench/dataset/` — `continuation_dataset.py` and `continuation_benchmark.py` orchestrate full dataset generation. `dataset_io.py` reads/writes JSONL + manifest.

- `emoji_bench/providers/` — Thin transport layer. Each provider module (`anthropic.py`, `openai.py`, `gemini.py`, `mistral.py`, `openrouter.py`) returns a `ContinuationResponse`. `continuation.py` dispatches based on `ModelConfig.provider`. **Gemini models route through OpenRouter, not the Gemini API directly.**

- `emoji_bench/eval/` — `runner.py` manages concurrent inference with resume-from-checkpoint. `matrix.py` owns the B/C + L0/L1 naming. `paths.py` resolves input/output artifact paths and loads `.env`.

- `emoji_bench/model_registry.py` — All `ModelConfig` instances and aliases (e.g. `claude-opus-4-7-reasoning-max`). Encodes per-model reasoning settings (OpenAI `reasoning.effort`, Anthropic `thinking`/`effort`, Gemini `thinkingLevel`).

- `emoji_bench/scoring/continuation_scorer.py` — Deterministic, post-hoc. Extracts `Final Output: <symbol>` from raw text, runs loose/strict verbalized-doubt detection, and classifies into six outcome buckets (`detect_recover`, `silent_recovery`, `detect_only`, `blind_wrong_branch`, `off_rails`, `extraction_failed`). **Inference and scoring are intentionally decoupled** so saved predictions can be rescored without new API calls.

- `emoji_bench/continuation_formatter.py` — Formats turn-1 user prompt, assistant prefill (steps up to cutoff), and the single-turn collapsed variant.

**Headline metric:** `final_answer_correct_rate` — extracted `Final Output:` matches the stored `ground_truth_final_output`.
