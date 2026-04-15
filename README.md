<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench is a benchmark for **unprompted self-detection** during derivation continuation in novel formal systems.

The core question is narrow:

**If a model is shown a partially completed derivation whose last visible step is wrong, and the only user instruction is to continue, will it notice or will it blindly cascade?**

## Benchmark Shape

Each example is a continuation-only 3-turn interaction:

1. Turn 1 user: rules for a procedurally generated formal system, an expression, and a required step format
2. Turn 1 assistant prefill: steps `1..Y`, where step `Y` is an injected cascading error
3. Turn 2 user: `Please continue.` by default

The model then produces the continuation that gets scored.

Two targets are stored per row:

- `ground_truth_final_output`: the correct final symbol from the clean derivation
- `wrong_branch_final_output`: the final symbol reached by blindly continuing from the bad state

These are always different by construction.

## Why It Exists

Most self-correction benchmarks explicitly ask the model to inspect for mistakes. That over-cues the behavior. Emoji-Bench is trying to measure the harder failure mode: the model is already mid-task, sees a bad intermediate result, and has to notice without being told to audit.

The benchmark is hard to shortcut because:

- the formal systems are procedurally generated and novel
- the injected error is non-convergent, so blindly continuing reaches a different terminal symbol

## Quick Start

### Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

### Install

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
```

Provider API keys:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`

Run tests:

```bash
pytest
```

### Generate a Dataset

```bash
python scripts/generate_continuation_dataset.py \
  --count 100 \
  --output-dir artifacts/emoji-bench-dataset-100
```

Generation rejects rows that fail any of:

- realized chain length `< 4`
- insufficient continuation runway after the prefill
- no midpoint-window cascading slot
- convergent wrong branch

Per-difficulty rejection counts are written to `manifest.json`.

### Run Inference

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill \
  --limit 10
```

Delivery modes:

- `--mode prefill`: uses the 3-message `B` shape `[user, assistant, user]`
- `--mode single_turn`: flattens the conversation into one user prompt with `=== WORK SO FAR ===` and `=== NEXT MESSAGE ===`

Useful flags:

- `--turn-2-prompt-level {0,1}`: prompt-strength axis for the Turn 2 user message
- `--max-output-tokens`: increase this for reasoning models
- `--no-resume`: ignore existing `predictions.jsonl`

Default output directories look like:

```text
artifacts/evals/<dataset>-<model>-<mode>[-lvl1]/
```

### Judge and Score

```bash
python scripts/judge_continuation.py <predictions-dir>
python scripts/score_continuation.py <predictions-dir>
```

Outputs written next to predictions:

- `judge.jsonl`
- `scores.jsonl`
- `nested_scores.jsonl` when judge output is available
- `score_summary.json`

## Metrics

Headline reporting uses three nested rates:

| Metric | Meaning |
|---|---|
| `detect_rate` | The continuation shows awareness that something is wrong |
| `detect_correct_rate` | It explicitly corrects the bad step |
| `detect_correct_finaloutput_correct_rate` | It corrects the bad step and produces a valid derivation to the correct final symbol |

Judge + validator split:

- the LLM judge checks whether the continuation acknowledged and explicitly corrected the bad step
- the deterministic validator checks each emitted derivation step, continuity between steps, and the final symbol

The regex bucket scorer remains in the repo as a cheap diagnostic baseline.

## Dataset Schema

Generated dataset folders contain:

```text
artifacts/<dataset>/
├── test.jsonl
├── manifest.json
└── README.md
```

Current row fields:

| Field | Purpose |
|---|---|
| `example_id` / `base_id` / `split` | identity |
| `difficulty` | `easy` / `medium` / `hard` / `expert` |
| `error_type` | `E-CONTINUE` in the current release |
| `turn_1_user` | rules, expression, and output format |
| `turn_1_assistant_prefill` | partial derivation ending on the injected error |
| `ground_truth_final_output` | correct terminal symbol |
| `wrong_branch_final_output` | terminal symbol from blindly continuing the wrong branch |
| `chain_length_x` | realized clean derivation length |
| `prefill_error_step` | 1-indexed injected error step |
| `target_step_count` | requested target length during generation |
| `system_json` / `system_seed` / `chain_seed` / `error_seed` | reproducibility metadata |

The Turn 2 user prompt is applied at evaluation time, not stored in dataset rows.

## Repo Map

Core modules in `emoji_bench/`:

- `generator.py`, `chain_generator.py`, `expressions.py`, `interpreter.py`: formal-system and derivation generation
- `error_injector.py`: cascading wrong-result injection
- `continuation_benchmark.py`: single continuation instance generator and serializer
- `continuation_formatter.py`: Turn 1, prefill, and single-turn prompt formatting
- `continuation_dataset.py`: dataset generation and rejection accounting
- `dataset_io.py`, `jsonl_io.py`, `provider_clients.py`: shared I/O and provider helpers
- `continuation_provider.py`: raw continuation requests across OpenAI, Anthropic, Gemini, and Mistral
- `continuation_judge.py`: judge prompt and deterministic regeneration of the bad-step values
- `continuation_validator.py`: parser and deterministic derivation validator
- `continuation_scorer.py`: regex baseline plus nested metric aggregation
- `model_registry.py`: configured model definitions

CLI scripts in `scripts/`:

- `generate_continuation_dataset.py`
- `evaluate_continuation.py`
- `judge_continuation.py`
- `score_continuation.py`
- `preview_dataset.py`

## Kaggle-Compatible Shapes

The benchmark is intentionally a 2x2 matrix:

- `B / L0`: `--mode prefill --turn-2-prompt-level 0`
- `B / L1`: `--mode prefill --turn-2-prompt-level 1`
- `C / L0`: `--mode single_turn --turn-2-prompt-level 0`
- `C / L1`: `--mode single_turn --turn-2-prompt-level 1`

## Artifacts

`artifacts/` contains generated datasets and evaluation results used by the repo. New runs still write there by default.

## Contributing

Useful contribution areas:

- judge prompt and validator refinements
- provider integrations
- continuation-only ablations
- model behavior analysis on the benchmark

## License

MIT. See `LICENSE`.
