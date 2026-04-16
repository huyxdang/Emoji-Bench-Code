<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench: LLMs Don't Look Back. 

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
- `--max-output-tokens`: increase this for longer continuations or reasoning-heavy runs
- `--no-resume`: ignore existing `predictions.jsonl`
- `--reasoning-effort`: overrides model reasoning effort
  OpenAI accepts `none|minimal|low|medium|high|xhigh`
  Anthropic effort-capable models accept `none|low|medium|high|max`

Default output directories look like:

```text
artifacts/evals/<model>-<cell>/
```

Where `<cell>` is one of:

- `B-L0`
- `B-L1`
- `C-L0`
- `C-L1`

Example:

```text
artifacts/evals/gpt-4.1-mini-B-L0/
artifacts/evals/gpt-4.1-mini-C-L1/
```

If you pass `--reasoning-effort`, that suffix is retained in the model slug:

```text
artifacts/evals/gpt-5.4-reasoning-high-B-L0/
artifacts/evals/claude-sonnet-4-6-reasoning-max-C-L1/
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

Judge artifacts are now fingerprinted against `predictions.jsonl`. If you
change predictions in place, rerun `judge_continuation.py --no-resume`.
`score_continuation.py` rejects stale, duplicate, or partial `judge.jsonl`
files instead of silently scoring a subset.

### Run the Full Matrix

```bash
./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8
```

`run.sh` is the repo-level batch runner. It:

- runs the full 32-cell model matrix
- resumes partially completed eval directories
- runs judge and score after the eval phase finishes
- continues past failed cells and prints a final failure summary

Defaults:

- judge model: `gpt-5.4-mini-no-reasoning`
- judge concurrency: `8`

Example override:

```bash
JUDGE_MAX_CONCURRENT=4 ./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8
```

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

The codebase is organized by responsibility:

- `emoji_bench/domain/`: formal-system generation, derivation chains, expression types, interpretation, and deterministic validation
- `emoji_bench/dataset/`: continuation-instance generation, cascading error injection, dataset serialization, and manifest helpers
- `emoji_bench/eval/`: matrix naming (`B/C`, `L0/L1`), artifact path resolution, and the shared evaluation runner
- `emoji_bench/providers/`: provider clients plus OpenAI, Anthropic, Gemini, and Mistral continuation transport
- `emoji_bench/judge/`: judge artifact validation, LLM-as-a-judge prompting, and score aggregation
- `emoji_bench/continuation_formatter.py`: Turn 1, prefill, and single-turn prompt formatting
- `emoji_bench/model_registry.py`: configured model aliases and provider-specific defaults
- `emoji_bench/jsonl_io.py`: shared JSONL helpers

CLI scripts in `scripts/`:

- `generate_continuation_dataset.py`
- `evaluate_continuation.py`
- `judge_continuation.py`
- `score_continuation.py`
- `preview_dataset.py`
- `plot_b_results.py`

Repo-level helpers:

- `run.sh`: full-matrix eval -> judge -> score batch runner

## Kaggle-Compatible Shapes

The benchmark is intentionally a 2x2 matrix:

- `B / L0`: `--mode prefill --turn-2-prompt-level 0`
- `B / L1`: `--mode prefill --turn-2-prompt-level 1`
- `C / L0`: `--mode single_turn --turn-2-prompt-level 0`
- `C / L1`: `--mode single_turn --turn-2-prompt-level 1`

## Artifacts

`artifacts/` contains the canonical in-repo dataset plus generated evaluation results and plots. New runs still write there by default.

## Contributing

Useful contribution areas:

- judge prompt and validator refinements
- provider integrations
- continuation-only ablations
- model behavior analysis on the benchmark

## License

MIT. See `LICENSE`.
