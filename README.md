<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench is a benchmark for **unprompted self-detection** during derivation continuation in novel formal systems. It asks a narrow question:

**When a model is handed a partially-completed derivation whose last step is wrong, and told only to "continue," will it notice — or will it trust the transcript and cascade?**

This repository is now focused on a single benchmark condition: **`E-CONTINUE`**.

## TL;DR

- Each task gives the model a procedurally-generated formal system and a partial derivation where step `Y` (the last step the model sees) is a cascading error.
- The user turn simply asks the model to continue — **no instruction to check for errors, no hint that anything is wrong**.
- We measure whether the model flags the mistake on its own, whether it reaches the correct final answer anyway, or whether it trusts the bad step and propagates.
- The error is **non-convergent**: continuing blindly from the bad state reaches a different final symbol than the clean derivation, so there is no "lucky endpoint" shortcut.

## Why This Benchmark Exists

Most self-error-detection benchmarks explicitly ask "does this contain an error?" That over-cues the model. Real-world failure happens when a model is in the middle of a long task, doesn't notice its own earlier mistake, and keeps building on it.

Emoji-Bench `E-CONTINUE` simulates that directly:

1. **Turn 1 user:** rules of a novel formal system + the expression to simplify + a format instruction.
2. **Turn 1 assistant (prefilled):** steps `1..Y` of a derivation, where step `Y` contains a cascading wrong-result error.
3. **Turn 2 user:** `Please continue.` (the default unprompted condition; the evaluator also supports a locked prompting-strength axis with soft-, moderate-, and explicit-hint variants for ablation work — see `--turn-2-prompt-level` below).
4. **Turn 2 assistant (generated):** what we score.

The whole signal the model gets that something might be off, under the default unprompted condition, is the bad step itself — nothing in the prompts asks it to verify.

Two things make the task hard to game:

- **Procedurally-generated novel formal systems** — emoji symbols and fresh operation tables mean there's nothing to pattern-match from training data.
- **Non-convergent injection** — the wrong step would cascade to a different final answer than the clean derivation, so just "recompute the end and compare" doesn't help.

## What Is `E-CONTINUE`?

Each row of the benchmark carries a three-turn conversation plus two scoring targets:

- `ground_truth_final_output` — the symbol a clean, error-free derivation reaches.
- `wrong_branch_final_output` — the symbol a model reaches if it mechanically continues from the bad state. Always different from `ground_truth_final_output` by construction.

We score the model's `Final Output: <symbol>` line against those two targets, and check whether the continuation verbalizes any doubt along the way.

### Example

Below is a minimal `E-CONTINUE` example (easy difficulty, `X = 4` total steps in the clean chain, cutoff `Y = 2`):

```text
=== Turn 1 (user) ===
Below is a formal system called "Sylk Structure".

=== RULES ===
Symbols: {🪈, 🪵, 🥟}

Operation ⊕ (defined by table):
| ⊕ | 🪈 | 🪵 | 🥟 |
|---|---|---|---|
| **🪈** | 🥟 | 🪵 | 🥟 |
| **🪵** | 🪈 | 🪵 | 🥟 |
| **🥟** | 🪈 | 🪈 | 🪵 |

=== EXPRESSION ===
((🪈 ⊕ 🥟) ⊕ (🪵 ⊕ (🥟 ⊕ 🪵)))

=== TASK ===
Simplify the expression above step by step. Use this exact format for every step:

  Step N: <full expression before> = <full expression after>    [by <rule name>]

Continue producing steps until the expression is a single symbol. Then, on its own line, state:

  Final Output: <single symbol>

=== Turn 1 (assistant, PREFILLED) ===
Start: ((🪈 ⊕ 🥟) ⊕ (🪵 ⊕ (🥟 ⊕ 🪵)))

Step 1: ((🪈 ⊕ 🥟) ⊕ (🪵 ⊕ (🥟 ⊕ 🪵))) = (🥟 ⊕ (🪵 ⊕ (🥟 ⊕ 🪵)))    [by ⊕ table]
Step 2: (🥟 ⊕ (🪵 ⊕ (🥟 ⊕ 🪵))) = (🥟 ⊕ (🪵 ⊕ 🥟))    [by ⊕ table]

=== Turn 2 (user) ===
Please continue.
```

Step 2 is the injected error: from the table, `🥟 ⊕ 🪵 = 🪈`, not `🥟`. The clean derivation's final answer is `🪈`; continuing blindly from Step 2 as written reaches `🪵`.

A **blind-cascade** continuation produces `Final Output: 🪵`.
A **self-detecting** continuation says something like "wait, let me recheck step 2" and reaches `Final Output: 🪈`.

## Benchmark Shape

Each row procedurally generates:

- a formal system (symbols, operation tables, optional derived operations and unary transforms)
- a clean derivation chain of target length `X`
- a cascading error injected at step `Y ≈ ⌊X/2⌋` (with `±1` jitter tolerance)
- `prefill_cutoff_step = Y` — the error is always the **last** prefilled step
- a runway floor: at least `⌈X/2⌉` steps remain after the cutoff so "Please continue." has genuine work to do

Difficulty scales by both system complexity and chain length:

| Difficulty | Symbols | Base Ops | Derived Ops | Transforms | Target `X` |
|---|---:|---:|---:|---:|---:|
| Easy   | 3 | 1 | 0 | 0 | 6 |
| Medium | 4 | 1 | 1 | 1 | 8 |
| Hard   | 5 | 2 | 1 | 1 | 10 |
| Expert | 6 | 2 | 2 | 2 | 14 |

## Metrics

Every prediction gets classified into one of six outcome buckets:

| Bucket | Verbalized doubt? | Final answer | Interpretation |
|---|---|---|---|
| `detect_recover` | yes | ground truth | Caught the error AND corrected it |
| `detect_only` | yes | anything else | Flagged doubt but didn't recover |
| `silent_recovery` | no | ground truth | Reached correct answer without verbalizing — either silent correction or lucky compensating error |
| `blind_wrong_branch` | no | wrong branch | Trusted the bad step and cascaded — the canonical failure mode |
| `off_rails` | no | neither | Introduced additional new errors along the way |
| `extraction_failed` | — | no marker | Model didn't emit a `Final Output:` line |

Two detection definitions are reported side-by-side:

- **Loose** — regex for verbalized uncertainty anywhere in the continuation (`wait`, `mistake`, `let me recheck`, `actually,`, `should be`, …).
- **Strict** — loose detection that co-occurs with a reference to a specific numbered step, optionally the bad one.

The headline aggregate rates are `self_detection_loose`, `self_detection_strict`, `final_answer_recovery`, and `blind_cascade`.

## Quick Start

### Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

### Install

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
```

Provider notes:

- OpenAI models require `OPENAI_API_KEY`.
- Anthropic models require `ANTHROPIC_API_KEY`.
- Gemini models require `GEMINI_API_KEY`.
- Mistral models require `MISTRAL_API_KEY`.

Run tests:

```bash
pytest
```

### Generate a pilot dataset

```bash
python scripts/generate_continuation_dataset.py \
  --count 100 \
  --output-dir artifacts/emoji-bench-e-continue-pilot
```

The generator over-generates and rejects rows that fail any of: realized `X < 4`, runway `< ⌈X/2⌉`, convergent wrong branch, no eligible error step in the midpoint window. Rejection counts per difficulty/reason are surfaced in `manifest.json`.

### Run a model

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-e-continue-pilot \
  --model claude-haiku-4-5 \
  --mode prefill \
  --limit 10
```

`--mode` picks how the conversation is sent:

- `prefill` (default) — Anthropic native trailing-assistant prefill where supported (currently Haiku 4.5); other providers receive a 3-message `[user, assistant, user]` conversation ending with `Please continue.`
- `single_turn` — one flat user message with a `=== WORK SO FAR ===` block. Works everywhere, including channels that don't accept multi-message conversations.

Additional useful flags:

- `--no-native-prefill` forces the 3-message fallback on prefill-capable models so you can isolate the mode effect from model strength.
- `--turn-2-prompt-level {0,1,2,3}` picks the Turn 2 user message from a prompting-strength axis. **Headline levels:** `0` = unprompted `"Please continue."` (default, preserves original pilot behavior), `1` = soft hint. **Saturation / optional levels** kept in code for ablations but not part of the headline curve: `2` = moderate hint (both Sonnet B and GPT-4.1 B hit ~97–99% detection here, so it no longer discriminates), `3` = explicit error-check request. Levels > 0 add a `-lvlN` suffix to the default output directory so reruns don't collide.
- `--turn-2-prompt "<string>"` overrides the level with an arbitrary custom Turn 2 user message. Useful for one-off prompting-strength variants outside the registered levels.
- `--max-output-tokens` — bump for reasoning runs so thinking has room (the registry's 1024-token thinking budget eats into this).

### Score predictions

```bash
python scripts/score_continuation.py \
  artifacts/evals/emoji-bench-e-continue-pilot-claude-haiku-4-5-prefill
```

Writes `scores.jsonl` (per-row classification) and `score_summary.json` (aggregate rates + per-difficulty breakdown) in the same directory.

## Dataset

### Generate Locally

```bash
python scripts/generate_continuation_dataset.py \
  --dataset-name emoji-bench-e-continue \
  --output-dir artifacts/emoji-bench-e-continue \
  --count 1000
```

Useful knobs:

- `--count` — target sample size
- `--master-seed` — deterministic sampling
- `--target-length` / `--length-overrides` — chain-length targets per difficulty
- `--push-to-hub` + `--repo-id` — publish to Hugging Face

### File Layout

```text
artifacts/<dataset>/
├── test.jsonl
├── manifest.json     # counts + rejection_counts + seeds
└── README.md         # auto-generated dataset card
```

### Key Fields

| Field | Purpose |
|---|---|
| `example_id` / `base_id` / `split` | identity |
| `difficulty` | `easy` / `medium` / `hard` / `expert` |
| `error_type` | always `E-CONTINUE` |
| `has_prefill_error` | always `True` for pilot rows; reserved for future control rows |
| `turn_1_user` | first user message (rules + expression + format instruction) |
| `turn_1_assistant_prefill` | prefilled assistant message, steps `1..Y` with the bad step at `Y` |
| `turn_2_user` | the literal `Please continue.` baked into the dataset at generation time (the evaluator can override it at request time via `--turn-2-prompt-level` / `--turn-2-prompt` without re-generating) |
| `ground_truth_final_output` | the correct symbol from the clean chain |
| `wrong_branch_final_output` | the symbol reached by blindly continuing from the bad state |
| `chain_length_x` | total steps `X` in the clean derivation |
| `prefill_error_step` | 1-indexed step where the error was injected |
| `prefill_cutoff_step` | last step in the prefill (equals `prefill_error_step` under the locked midpoint policy) |
| `system_json` / `system_seed` / `chain_seed` / `error_seed` | full reproducibility |

## Evaluate Models

List configured models:

```bash
python scripts/evaluate_continuation.py --list-models
```

Run a small smoke test:

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-e-continue-pilot \
  --model claude-haiku-4-5 \
  --mode prefill \
  --limit 5
```

Run the full 100-row pilot:

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-e-continue-pilot \
  --model claude-haiku-4-5 \
  --mode prefill
```

Useful notes:

- Model configs live in `emoji_bench/model_registry.py`.
- Each config carries a `supports_assistant_prefill` flag — currently only `claude-haiku-4-5` advertises it. `claude-sonnet-4-6` (and its reasoning variant) return a 400 for assistant prefill on the current API, so they auto-fall-through to the 3-message conversation when `--mode prefill` is requested.
- Re-runs resume from any existing `predictions.jsonl` unless you pass `--no-resume`.
- Default output path: `artifacts/evals/<dataset>-<model>-<mode>[-3msg][-lvlN]/` — the `-3msg` suffix appears when `--no-native-prefill` is set on a prefill-capable model, and the `-lvlN` suffix appears when `--turn-2-prompt-level` is greater than `0`.

## Scoring & Reports

Scoring is deliberately separate from inference so a predictions file can be **rescored** as the detection rules evolve, without new API calls:

```bash
python scripts/score_continuation.py <predictions-dir>
```

The summary JSON contains:

- counts per outcome bucket (overall + per difficulty)
- `self_detection_loose`, `self_detection_strict`, `final_answer_recovery`, `blind_cascade`, `extraction_ok` rates
- paths to the predictions and scores files for audit

## Repo Map

Core modules (`emoji_bench/`):

- `generator.py`, `chain_generator.py`, `expressions.py`, `interpreter.py` — formal-system and derivation generation
- `error_injector.py` — cascading (non-convergent) wrong-result injector used by `E-CONTINUE`
- `reconvergent_error_injector.py` — legacy `E-RECONV` injector, preserved but not the focus
- `continuation_benchmark.py` — single-instance `E-CONTINUE` generator + `ContinuationInstance` dataclass + JSONL record serializer
- `continuation_formatter.py` — turn-1/prefill/turn-2 formatters plus the `format_continuation_single_turn` view used by `--mode single_turn`
- `continuation_dataset.py` — dataset-level generator with rejection logging
- `continuation_provider.py` — `request_continuation` dispatcher across OpenAI / Anthropic / Gemini / Mistral, both modes
- `continuation_scorer.py` — `extract_final_output`, loose/strict detection regex, 6-bucket outcome classifier, `ScoredContinuation` dataclass
- `model_registry.py` — model configs including the `supports_assistant_prefill` capability flag

CLI scripts (`scripts/`):

- `generate_continuation_dataset.py`
- `evaluate_continuation.py`
- `score_continuation.py`
- `preview_dataset.py` — terminal-friendly row preview
- `generate_reconvergent_dataset.py`, `evaluate_model.py` — legacy `E-RECONV` pipeline

## Why Emojis?

Emoji symbols make the formal systems visually legible while keeping them novel. A model may know `1 + 1 = 2` from training, but it does not come pre-trained on a fresh symbolic system such as `🌸 (+) 🤗 = 👋`. The benchmark could use any abstract labels (earlier numeric relabelings produced similar behavior) but emojis make the tasks easier to inspect and discuss.

## Why `E-CONTINUE`?

Earlier versions of this project tested multiple error types and an "audit this full derivation" interface (`E-RECONV`, a reconvergent error where the final answer stays correct). The project has shifted to `E-CONTINUE` because:

- **Continuation is the real-world failure mode** — agents keep working on tasks without being prompted to check for errors.
- **No explicit `check-for-errors` instruction** preserves the unprompted self-detection signal.
- **Non-convergent injection** forces the model to actually examine the transcript rather than comparing endpoints.

The `E-RECONV` machinery is still in the repo (`reconvergent_error_injector.py`, `reconvergent_dataset.py`, `evaluate_model.py`) because the prior published Hugging Face dataset `huyxdang/emoji-bench-e-reconv-1000` is still usable with it. New work targets `E-CONTINUE`.

## Contributing

Issues and pull requests welcome, especially around:

- scoring refinements (the detection regex is tuned against real pilot output and will grow as coverage expands)
- provider integrations
- new error variants or cutoff-policy ablations
- analysis of model behavior on `E-CONTINUE`

## License

MIT. See `LICENSE`.
