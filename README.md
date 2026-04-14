<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Focus (Single Condition)
L0 (please continue) && Condition B (3 message, pre-filled)

More, for the pleasure of it: 
- L1 (some hint to check error)
- Condition C: Single-prompt (may not even need this, since Kaggle Benchmark may support this)

# Emoji-Bench

Emoji-Bench is a benchmark for **unprompted self-detection** during derivation continuation in novel formal systems. It asks a narrow question:

**When a model is handed a partially-completed derivation whose last step is wrong, and told only to "continue," will it notice — or will it trust the transcript and cascade?**

## TL;DR

- Each task gives the model a procedurally-generated formal system and a partial derivation where step `Y` (the last step the model sees) is a cascading error.
- The user turn simply asks the model to continue — **no instruction to check for errors, no hint that anything is wrong**.
- We measure whether the model flags the mistake on its own, whether it reaches the correct final answer anyway, or whether it trusts the bad step and propagates.
- The error is **non-convergent**: continuing blindly from the bad state reaches a different final symbol than the clean derivation, so there is no "lucky endpoint" shortcut.

## Why This Benchmark Exists

Most self-error-detection benchmarks explicitly ask "does this contain an error?" That over-cues the model. Real-world failure happens when a model is in the middle of a long task, doesn't notice its own earlier mistake, and keeps building on it.

Emoji-Bench simulates that directly:

1. **Turn 1 user:** rules of a novel formal system + the expression to simplify + a format instruction.
2. **Turn 1 assistant (prefilled):** steps `1..Y` of a derivation, where step `Y` contains a cascading wrong-result error.
3. **Turn 2 user:** `Please continue.` (the default unprompted condition; the evaluator also supports a locked prompting-strength axis with soft-, moderate-, and explicit-hint variants for ablation work — see `--turn-2-prompt-level` below).
4. **Turn 2 assistant (generated):** what we score.

The whole signal the model gets that something might be off, under the default unprompted condition, is the bad step itself — nothing in the prompts asks it to verify.

Two things make the task hard to game:

- **Procedurally-generated novel formal systems** — emoji symbols and fresh operation tables mean there's nothing to pattern-match from training data.
- **Non-convergent injection** — the wrong step would cascade to a different final answer than the clean derivation, so just "recompute the end and compare" doesn't help.

## What Each Row Looks Like

Each row carries a three-turn conversation plus two scoring targets:

- `ground_truth_final_output` — the symbol a clean, error-free derivation reaches.
- `wrong_branch_final_output` — the symbol a model reaches if it mechanically continues from the bad state. Always different from `ground_truth_final_output` by construction.

We score the model's `Final Output: <symbol>` line against those two targets, and check whether the continuation verbalizes any doubt along the way.

### Example

Below is a minimal example (easy difficulty, `X = 4` total steps in the clean chain, cutoff `Y = 2`):

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

The headline scoring is **three nested rates**, each a strict refinement of the previous:

| Metric | Definition |
|---|---|
| **detect_rate** | Did the model express awareness that some step was wrong? |
| **detect_correct_rate** | Did it also explicitly restate step `Y` with the correct value? |
| **detect_correct_finaloutput_correct_rate** | Did it also produce a derivation that is mathematically valid all the way to the correct final answer? |

By construction `detect_correct_finaloutput_correct ⊆ detect_correct ⊆ detect`. Each metric isolates a specific failure mode: a model can notice without fixing, fix the named step but make later math errors, or fix the named step and ride a coincidence to a correct-looking but wrong derivation.

### How each metric is computed

- **`detect_rate`** and **`detect_correct_rate`** come from a single LLM-as-judge call per prediction (default model: `gpt-4.1-mini`, cross-family from Claude). The judge is a reading-comprehension task — it's given the correct value for the bad step and asked yes/no whether the continuation acknowledged the error and whether it explicitly restated step `Y` with the correct value. The judge never has to do the formal-system math itself.
- **`detect_correct_finaloutput_correct_rate`** combines the judge's `corrected_step_y` with a deterministic Python validator. The validator parses the model's `Step N: <before> = <after>` lines, evaluates each step against the operation table (catching wrong reductions), checks consecutive-step continuity, and verifies the terminal symbol equals `ground_truth_final_output`. This closes the **compensating-error loophole**: a model that writes two wrong steps that cancel out to ground truth is caught by the validator's per-step check before terminal match is even considered.

The narrow definition of "fix" is deliberate: implicit corrections (continuing from a different state without naming step `Y`) do **not** count for `detect_correct_rate`. That keeps the judge task purely a reading task and lets metric (3)'s validator be the authority on whether the model's downstream work was actually consistent.

### Regex baseline (diagnostic)

The original six-bucket regex classifier (`detect_recover` / `detect_only` / `silent_recovery` / `blind_wrong_branch` / `off_rails` / `extraction_failed`) is preserved as a diagnostic baseline. It runs offline with no API calls, ships in `score_summary.json` under `regex_baseline`, and is useful for cheap iteration and cross-checking the judge. Two regex detection variants are computed:

- **Loose** — regex for verbalized uncertainty anywhere in the continuation (`wait`, `mistake`, `let me recheck`, `actually,`, `should be`, …).
- **Strict** — loose detection that co-occurs with a reference to a specific numbered step, optionally the bad one.

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

### Generate the benchmark dataset

```bash
python scripts/generate_continuation_dataset.py \
  --count 100 \
  --output-dir artifacts/emoji-bench-dataset-100
```

The generator over-generates and rejects rows that fail any of: realized `X < 4`, runway `< ⌈X/2⌉`, convergent wrong branch, no eligible error step in the midpoint window. Rejection counts per difficulty/reason are surfaced in `manifest.json`.

### Run a model

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill \
  --limit 10
```

`--mode` picks how the conversation is sent. Three delivery options exist; only **B** and **C** are in the headline grid (see [Headline run grid](#headline-run-grid-kaggle-benchmark-compatible) below):

- **Option A — native prefill** (`--mode prefill` on `claude-haiku-4-5`): Anthropic trailing-assistant prefill. Excluded from the headline because the Kaggle Benchmark harness does not expose this API field.
- **Option B — 3-message fallback** (`--mode prefill` on non-Haiku models, or `--mode prefill --no-native-prefill` on Haiku): `[user, assistant(prefilled_steps), user("Please continue.")]`. The default headline delivery.
- **Option C — single-prompt** (`--mode single_turn`): one flat user message with a `=== WORK SO FAR ===` block. Works everywhere, including channels that don't accept multi-message conversations.

Additional useful flags:

- `--no-native-prefill` forces the 3-message fallback on prefill-capable models. Required on Haiku for any Kaggle-comparable run.
- `--turn-2-prompt-level {0,1,2,3}` picks the Turn 2 user message from a prompting-strength axis. **Headline levels:** `0` = unprompted `"Please continue."` (default), `1` = soft hint. **Saturation / optional levels** kept in code for ablations but not part of the headline curve: `2` = moderate hint (both Sonnet B and GPT-4.1 B hit ~97–99% detection here, so it no longer discriminates), `3` = explicit error-check request. Levels > 0 add a `-lvlN` suffix to the default output directory so reruns don't collide.
- `--turn-2-prompt "<string>"` overrides the level with an arbitrary custom Turn 2 user message. Useful for one-off prompting-strength variants outside the registered levels.
- `--max-output-tokens` — bump for reasoning runs so thinking has room (the registry's 1024-token thinking budget eats into this).

### Headline run grid (Kaggle Benchmark compatible)

Headline results are reported on a 2×2 grid that matches what the Kaggle Benchmark harness can execute:

| | **Option B — 3-message fallback** | **Option C — single-prompt** |
|---|---|---|
| **L0 (unprompted)** | `--mode prefill` [`--no-native-prefill`] | `--mode single_turn` |
| **L1 (soft hint)**  | `--mode prefill --turn-2-prompt-level 1` [`--no-native-prefill`] | `--mode single_turn --turn-2-prompt-level 1` |

Rules:

- **Only L0 and L1** are in the headline. L2 saturates near 97–99% detection on mid-tier models and L3 is an explicit error-check request; both remain in code for ablations.
- **Only Options B and C** are in the headline. Option A (native assistant prefill) is not reproducible on Kaggle Benchmark.
- For `claude-haiku-4-5`, always add `--no-native-prefill` — otherwise `--mode prefill` silently upgrades to Option A and the run is off-distribution for Kaggle.
- Every other model in `model_registry.py` already falls through to Option B under `--mode prefill` (no extra flag needed).

### Judge + score predictions

The headline nested metrics require an LLM-as-judge pass first. Two-step:

```bash
# 1. Run the judge over a predictions directory (one OpenAI call per row).
python scripts/judge_continuation.py \
  artifacts/evals/emoji-bench-dataset-100-claude-haiku-4-5-prefill

# 2. Score: emits both nested headline (judge + validator) and regex baseline.
python scripts/score_continuation.py \
  artifacts/evals/emoji-bench-dataset-100-claude-haiku-4-5-prefill
```

If you skip step 1, `score_continuation.py` falls back to the regex-only baseline (with a note that nested metrics are unavailable).

Outputs (all written next to the predictions):
- `judge.jsonl` — one row per prediction with the judge's two booleans + reasoning. Resumable: re-running the judge skips already-judged rows.
- `scores.jsonl` — per-row regex classification (six-bucket).
- `nested_scores.jsonl` — per-row nested booleans + the judge & validator payloads. Only written when `judge.jsonl` is present.
- `score_summary.json` — aggregate rates. The `headline` block is the three nested rates when the judge ran, otherwise the regex rates. The `regex_baseline` block is always present.

Useful judge flags:

- `--judge-model` — defaults to `gpt-4.1-mini`. Must be an OpenAI-provider model (the judge uses Responses-API structured output).
- `--no-resume` — re-judge every row, ignoring an existing `judge.jsonl`.
- `--limit N` — judge only the first N pending rows. Handy for cost-bounded smoke runs before scaling up.

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
| `error_type` | identifier for the error variant (a single value in this release) |
| `has_prefill_error` | always `True` in the current dataset; reserved for future control rows |
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
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill \
  --limit 5
```

Run the full 100-row benchmark:

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill
```

Useful notes:

- Model configs live in `emoji_bench/model_registry.py`.
- Each config carries a `supports_assistant_prefill` flag — currently only `claude-haiku-4-5` advertises it. `claude-sonnet-4-6` (and its reasoning variant) return a 400 for assistant prefill on the current API, so they auto-fall-through to the 3-message conversation when `--mode prefill` is requested.
- Re-runs resume from any existing `predictions.jsonl` unless you pass `--no-resume`.
- Default output path: `artifacts/evals/<dataset>-<model>-<mode>[-3msg][-lvlN]/` — the `-3msg` suffix appears when `--no-native-prefill` is set on a prefill-capable model, and the `-lvlN` suffix appears when `--turn-2-prompt-level` is greater than `0`.

## Scoring & Reports

Inference, judge, and scoring are three deliberately separate steps so any of them can be re-run without re-spending the others. The judge pass costs ~1¢ per 100 rows on `gpt-4.1-mini` and is rerunnable; the validator and the regex baseline are pure Python with no API cost. To rescore as the metrics evolve, you only need to re-run `score_continuation.py` — predictions and judge verdicts are reused from disk.

```bash
python scripts/judge_continuation.py <predictions-dir>     # judge.jsonl
python scripts/score_continuation.py <predictions-dir>     # scores.jsonl + nested_scores.jsonl + score_summary.json
```

The `score_summary.json` `headline` block always reports the three nested rates when the judge has run (with per-difficulty breakdown), and the `regex_baseline` block always reports the six-bucket counts and rates for diagnostic comparison.

## Repo Map

Core modules (`emoji_bench/`):

- `generator.py`, `chain_generator.py`, `expressions.py`, `interpreter.py` — formal-system and derivation generation
- `error_injector.py` — cascading (non-convergent) wrong-result injector
- `reconvergent_error_injector.py` — legacy reconvergent-error injector from an earlier experiment, preserved but unused
- `continuation_benchmark.py` — single-instance benchmark generator + `ContinuationInstance` dataclass + JSONL record serializer
- `continuation_formatter.py` — turn-1/prefill/turn-2 formatters plus the `format_continuation_single_turn` view used by `--mode single_turn`
- `continuation_dataset.py` — dataset-level generator with rejection logging
- `continuation_provider.py` — `request_continuation` dispatcher across OpenAI / Anthropic / Gemini / Mistral, both modes
- `continuation_judge.py` — judge prompt builder, deterministic `compute_step_values` (regenerates the correct/injected step values from the dataset's seeds), single-shot OpenAI Responses-API call returning structured JSON
- `continuation_validator.py` — recursive-descent parser for the `Step N: <before> = <after>` format plus a deterministic per-step + continuity + terminal validator that closes the compensating-error loophole in metric (3)
- `continuation_scorer.py` — both the legacy regex bucket scorer and the new `score_prediction_nested` / `summarize_nested` combinator that turns judge + validator outputs into the three nested headline metrics
- `model_registry.py` — model configs including the `supports_assistant_prefill` capability flag

CLI scripts (`scripts/`):

- `generate_continuation_dataset.py`
- `evaluate_continuation.py`
- `score_continuation.py`
- `preview_dataset.py` — terminal-friendly row preview
- `judge_continuation.py` — runs the LLM judge over a predictions directory and emits `judge.jsonl`
- `generate_reconvergent_dataset.py`, `evaluate_model.py` — legacy reconvergent pipeline (earlier experiment, preserved but unused)

## Why Emojis?

Emoji symbols make the formal systems visually legible while keeping them novel. A model may know `1 + 1 = 2` from training, but it does not come pre-trained on a fresh symbolic system such as `🌸 (+) 🤗 = 👋`. The benchmark could use any abstract labels (earlier numeric relabelings produced similar behavior) but emojis make the tasks easier to inspect and discuss.

## Design Choices

Three choices shape the task and together keep it from being gameable:

- **Continuation is the real-world failure mode** — agents keep working on tasks without being prompted to check for errors. We simulate that directly rather than asking "find the mistake."
- **No explicit `check-for-errors` instruction** preserves the unprompted self-detection signal.
- **Non-convergent injection** forces the model to actually examine the transcript rather than comparing endpoints.

Earlier experiments in this repo tested alternative error types and an "audit this full derivation" interface; the corresponding generator/evaluator modules (`reconvergent_error_injector.py`, `reconvergent_dataset.py`, `evaluate_model.py`) are preserved but not part of the current benchmark.

## Contributing

Issues and pull requests welcome, especially around:

- judge prompt + validator refinements (the judge prompt is tuned against real benchmark output; the regex baseline is preserved as a diagnostic)
- provider integrations
- new error variants or cutoff-policy ablations
- analysis of model behavior on the benchmark

## License

MIT. See `LICENSE`.


## TODO
- [ ] Add control (no-error)

## Note: Kaggle Benchmarks supports Condition B

The `kaggle-benchmarks` Python library (the harness behind Kaggle Benchmarks evaluations) supports manual construction of a pre-filled assistant turn, so **Condition B (3-message fallback)** is runnable there without falling back to Condition C.

Relevant API surface:

- `Chat.append(item: Message)` is public — messages can be appended to a `Chat` manually before calling the LLM.
- `Message(content, sender)` is a plain dataclass with no role restriction on construction; an assistant-authored message can be built without the LLM generating it, as long as the `sender` is an `LLMChat`-shaped actor so the serializer maps it to `role: "assistant"` upstream.
- `LLMChat.respond()` reads the entire visible chat history (`raw_messages = [msg for msg in chat.messages if msg.is_visible_to_llm] + temp_messages`) and sends it to the provider. It does not assume the last turn was user-generated — the conversation just needs to end on a user turn for providers (like Anthropic) that reject trailing-assistant prompts.

The L0 + Condition B run therefore looks like:

```python
with chats.new("run"):
    chats.send(Message(content=turn_1_user, sender=user_actor))
    chats.send(Message(content=prefilled_steps_1_to_Y, sender=llm))  # pre-filled assistant turn
    chats.send(Message(content="Please continue.", sender=user_actor))
    result = llm.respond()
```

Worth a 10-row smoke test against a cheap model before a full run — the Kaggle harness may impose task-shape constraints (e.g., single `prompt()` call per task) that limit manual message-list construction, and the assistant-role serialization for the pre-filled turn should be verified end-to-end on the specific provider backing each model.

Condition A (Anthropic native *trailing-assistant* prefill) remains unsupported, since that requires the request itself to end with an assistant message; that API field is not exposed by the harness.

References:

- [Kaggle/kaggle-benchmarks (GitHub)](https://github.com/Kaggle/kaggle-benchmarks)
- [kaggle-benchmarks on DeepWiki](https://deepwiki.com/Kaggle/kaggle-benchmarks)