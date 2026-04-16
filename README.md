<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench: LLMs Don't Look Back

Emoji-Bench is a benchmark for **unprompted self-detection** during derivation continuation in novel formal systems.

**The question:** if a model is shown a partially completed derivation whose last visible step is wrong, and the only user instruction is to continue, will it notice — or blindly cascade?

Each row is a 3-turn interaction:
1. **Turn 1 user:** rules for a procedurally generated formal system, an expression, and a required step format.
2. **Turn 1 assistant (prefilled):** steps `1..Y`, where step `Y` is a deliberately injected error.
3. **Turn 2 user:** `Please continue.`

The model's continuation is scored against two stored targets — the correct terminal symbol and the terminal reached by blindly cascading from the bad step. These are always different by construction.

---

## Setup

**Requirements:** Python `>=3.11` and [`uv`](https://docs.astral.sh/uv/).

Install:

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
```

Provider API keys:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export MISTRAL_API_KEY=...
```

Run the test suite to confirm everything works:

```bash
pytest
```

---

## Reproduce the dataset

```bash
python scripts/generate_continuation_dataset.py \
  --count 100 \
  --output-dir artifacts/emoji-bench-dataset-100
```

This produces:

```
artifacts/emoji-bench-dataset-100/
├── test.jsonl      # 100 rows, stratified 25 × {easy, medium, hard, expert}
├── manifest.json   # seeds, rejection counts, generator commit hash
└── README.md
```

Rows are rejected at generation time if they fail any of:

- realized chain length `< 4`
- insufficient continuation runway after the prefill
- no midpoint-window cascading slot
- convergent wrong branch (wrong terminal equals ground truth)

Every row is byte-reproducible from its `system_seed`, `chain_seed`, and `error_seed`.

---

## Run the full experiment

The evaluation matrix is **8 models × 2 delivery shapes × 2 prompt strengths = 32 cells**.

```bash
./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8
```

`run.sh` will:

1. Run `evaluate_continuation.py` over all 32 cells (resuming partially completed ones).
2. Run the LLM-as-judge over each successful prediction set.
3. Run the deterministic scorer.
4. Continue past failed cells and print a final failure summary.

Defaults:

- Judge model: `gpt-5.4-mini-no-reasoning`
- Judge concurrency: `8` (`JUDGE_MAX_CONCURRENT=N` to override)

The four matrix cells per model:

| Cell | Delivery | Turn-2 prompt |
|---|---|---|
| **B-L0** | prefill (native 3-message) | `Please continue.` |
| **B-L1** | prefill | `Please continue. Double-check any step you're unsure about.` |
| **C-L0** | single-turn flattened transcript | `Please continue.` |
| **C-L1** | single-turn | `Please continue. Double-check …` |

Artifacts for each cell land at:

```
artifacts/evals/<model>-<B|C>-L<0|1>/
├── predictions.jsonl
├── judge.jsonl
├── scores.jsonl
├── nested_scores.jsonl
└── score_summary.json
```

### Running a single cell

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill \
  --turn-2-prompt-level 0

python scripts/judge_continuation.py <predictions-dir>
python scripts/score_continuation.py <predictions-dir>
```

---

## Metrics

Three nested rates, each a strict subset of the previous:

| Metric | Meaning | Grader |
|---|---|---|
| `detect_rate` | Continuation acknowledges something is wrong | LLM judge |
| `detect_correct_rate` | Continuation explicitly identifies and fixes the bad step | LLM judge |
| `detect_correct_finaloutput_correct_rate` | Correction **and** a mechanically valid derivation to the right terminal | LLM judge + deterministic validator |

`DCF ≤ D+C ≤ Detect` always holds. The gap between `D+C` and `DCF` measures "got the diagnosis right, fumbled the cleanup."

---

## Results

All values are the strictest metric (`detect_correct_finaloutput_correct_rate`) on the `artifacts/emoji-bench-dataset-100` dataset (N=100).

### B-L0 — prefill, no audit hint (headline condition)

| Model | DCF |
|---|---:|
| gpt-5.4-reasoning-xhigh | **0.41** |
| claude-opus-4-6-reasoning-high | 0.28 |
| gemini-3.1-pro-preview-thinking-high | 0.05 |
| claude-sonnet-4-6-reasoning-high | 0.02 |
| gemini-3-flash-preview-thinking-high | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.00 |
| mistral-large-2512 | 0.00 |
| magistral-medium-2509 | 0.00 |

### B-L1 — prefill, audit hint

| Model | DCF |
|---|---:|
| claude-opus-4-6-reasoning-high | **0.60** |
| gpt-5.4-reasoning-xhigh | 0.47 |
| gemini-3.1-pro-preview-thinking-high | 0.32 |
| claude-sonnet-4-6-reasoning-high | 0.23 |
| gemini-3-flash-preview-thinking-high | 0.00 |
| mistral-large-2512 | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.00 |
| magistral-medium-2509 | 0.00 |

### C-L0 — single-turn, no audit hint

| Model | DCF |
|---|---:|
| gpt-5.4-reasoning-xhigh | **0.07** |
| claude-opus-4-6-reasoning-high | 0.02 |
| all others | 0.00 |

### C-L1 — single-turn, audit hint

| Model | DCF |
|---|---:|
| claude-opus-4-6-reasoning-high | **0.45** |
| gpt-5.4-reasoning-xhigh | 0.14 |
| claude-sonnet-4-6-reasoning-high | 0.06 |
| all others | 0.00 |

### Takeaways

- **The hint (L0 → L1) matters enormously.** Opus jumps 0.28 → 0.60 on B and 0.02 → 0.45 on C. Much of the L0 failure is "missing permission to audit," not inability.
- **Prefill (B) beats single-turn (C)** for most models. Seeing the bad step in the *assistant* role seems easier to flag than reading the same transcript as user context.
- **A floor cluster sits at ~0 across the board** (gpt-5.4-mini, magistral-medium, mistral-large). They cascade regardless of hint or delivery mode.
- **Gemini 3.1 Pro only has B cells** — C runs have not been executed yet.

Full per-difficulty breakdowns are in each cell's `score_summary.json`.

---

## Repo map

- `emoji_bench/domain/` — formal-system generation, derivation chains, interpretation, deterministic validation
- `emoji_bench/dataset/` — continuation-instance generation, cascading error injection, dataset serialization
- `emoji_bench/eval/` — matrix naming (`B/C`, `L0/L1`), artifact paths, shared runner
- `emoji_bench/providers/` — OpenAI / Anthropic / Gemini / Mistral transport
- `emoji_bench/judge/` — judge artifact validation, LLM-as-judge prompting, score aggregation
- `emoji_bench/continuation_formatter.py` — Turn-1, prefill, and single-turn prompt formatting
- `emoji_bench/model_registry.py` — model aliases and provider-specific defaults
- `scripts/` — `generate_continuation_dataset.py`, `evaluate_continuation.py`, `judge_continuation.py`, `score_continuation.py`, `preview_dataset.py`, `plot_b_results.py`
- `run.sh` — full-matrix eval → judge → score batch runner

## License

MIT. See `LICENSE`.
