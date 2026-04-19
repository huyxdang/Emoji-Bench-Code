<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

---

*Note*: Our results and those from Kaggle Benchmark differ. Our experiments control for models' reasoning capabilities and max_tokens, but 
we couldn't control such configurations for models on Kaggle Benchmark. 

---

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
# or, for the full environment including plotting + cert bundle:
pip install -r requirements.txt
```

The OpenAI and Anthropic extras pull in their official SDKs. **Gemini and Mistral go through plain HTTP** (`urllib`) in `emoji_bench/providers/clients.py`, so no extra package is needed for them — just the API key below.

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

The suite covers generation, the mechanical validator, the judge, and all four provider transports. One test in particular is worth pointing at if you want to verify the grader end-to-end on a known-good row:

```bash
pytest tests/test_continuation_validator.py::test_validate_clean_chain_reaches_gt -v
```

This parses a clean derivation with the `Step N: …` regex, re-evaluates each step under the interpreter, and asserts the terminal matches `ground_truth_final_output`.

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
├── manifest.json   # master seed, difficulty config snapshot, rejection counts, generator commit hash
└── README.md
```

Rows are rejected at generation time if they fail any of:

- realized chain length `< 4`
- insufficient continuation runway after the prefill
- no midpoint-window cascading slot
- convergent wrong branch (wrong terminal equals ground truth)

### Row schema

Each `test.jsonl` line is a JSON object with:

| Field | Meaning |
|---|---|
| `example_id`, `base_id`, `split`, `difficulty`, `error_type` | identifiers + stratification |
| `turn_1_user` | rules + expression + step-format instructions |
| `turn_1_assistant_prefill` | partial derivation ending on the injected bad step |
| `ground_truth_final_output` | correct terminal symbol (clean chain) |
| `wrong_branch_final_output` | terminal reached by cascading from the bad step |
| `chain_length_x`, `prefill_error_step`, `target_step_count` | chain shape |
| `system_json` | serialized formal system (tables, transforms) |
| `system_seed`, `chain_seed`, `error_seed` | generation seeds |

The default Turn 2 user message is not stored in the row; evaluation applies `Please continue.` or a prompt-strength override at run time.

### Reproduce a single row from its seeds

Every row is byte-reproducible from its three seeds. Given a row, you can rebuild the formal system, regenerate the chain, inject the error at the same slot, and get back an identical `turn_1_assistant_prefill`:

```python
import json
from emoji_bench.dataset.continuation_benchmark import generate_continuation_instance
from emoji_bench.domain.formatter import system_from_json

with open("artifacts/emoji-bench-dataset-100/test.jsonl") as f:
    row = json.loads(next(f))  # first row

# Rebuild the formal system from the stored JSON (equivalent to regenerating
# it from system_seed with the row's difficulty config).
system = system_from_json(row["system_json"])

instance = generate_continuation_instance(
    system,
    length=row["target_step_count"],
    chain_seed=row["chain_seed"],
    error_seed=row["error_seed"],
)

# `instance.*_final_output` are Symbol objects; the stored row has emoji strings.
assert instance.turn_1_assistant_prefill        == row["turn_1_assistant_prefill"]
assert instance.ground_truth_final_output.emoji == row["ground_truth_final_output"]
assert instance.wrong_branch_final_output.emoji == row["wrong_branch_final_output"]
```

The assertions will hold byte-for-byte on any machine as long as the generator commit matches `manifest.json → generator_commit`.

---

## Run the full experiment

The current runner matrix is **9 models × 2 delivery shapes × 2 prompt strengths = 36 cells**.

The checked-in artifact bundle in `artifacts/evals/` is intentionally partial:

- `claude-opus-4-7-reasoning-max`: `B-L0` only
- `gemini-3.1-pro-preview-thinking-high`: `B-L0` and `B-L1` only
- all other listed models: full `2 × 2` coverage

```bash
./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8
```

`run.sh` will:

1. Run `evaluate_continuation.py` over all 36 cells (resuming partially completed ones).
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

### Grader format

The deterministic validator (`emoji_bench/domain/continuation_validator.py`) parses the model's continuation one line at a time using this case-insensitive regex:

```python
_STEP_LINE_REGEX = re.compile(
    r"^\s*Step\s+(\d+)\s*:\s*(.+?)\s*=\s*(.+?)(?:\s*\[by\s+[^\]]+\])?\s*$",
    re.IGNORECASE,
)
```

Canonical step line shown to the model:

```
Step N: <full expression before> = <full expression after>    [by <rule name>]
```

Notes on the parser's acceptance surface:

- The `[by <rule>]` clause is **optional** — a bare `Step N: lhs = rhs` parses too.
- Non-matching lines (free-form prose, commentary, empty lines) are **skipped silently** — only `Step N:` lines contribute.
- The validator does **not** consult the prefill. It only checks the model's *own* emitted steps.

Validation then enforces three rules:

1. **Per-step reduction validity.** Each `before = after` is re-evaluated under the system's operation tables; any step whose `before` and `after` don't agree under the interpreter is rejected (`first_invalid_step`).
2. **Consecutivity.** Each step's `before` must equal the previous step's `after` — catches jumps and dropped intermediate steps (`first_discontinuity_step`).
3. **Terminal match.** The final `after` must be a single symbol equal to `ground_truth_final_output`.

This closes the compensating-error loophole: a chain that writes a wrong intermediate `after` but "cancels" it to land on the right terminal is still rejected.

---

## Results

All three nested rates on the `artifacts/emoji-bench-dataset-100` dataset (N=100). Rows are sorted by DCF (the strictest metric). `D = detect_rate`, `DC = detect_correct_rate`, `DCF = detect_correct_finaloutput_correct_rate`.

Coverage note:

- `claude-opus-4-7-reasoning-max` currently appears only in `B-L0`
- `gemini-3.1-pro-preview-thinking-high` currently appears only in `B-L0` and `B-L1`
- models are omitted from cell tables that do not have committed artifacts yet

### B-L0 — prefill, no audit hint (headline condition)

| Model | D | DC | DCF |
|---|---:|---:|---:|
| gpt-5.4-reasoning-xhigh | 0.54 | 0.52 | **0.41** |
| claude-opus-4-7-reasoning-max | 0.46 | 0.45 | 0.38 |
| claude-opus-4-6-reasoning-high | 0.33 | 0.31 | 0.28 |
| gemini-3.1-pro-preview-thinking-high | 0.07 | 0.06 | 0.05 |
| claude-sonnet-4-6-reasoning-high | 0.03 | 0.03 | 0.02 |
| gemini-3-flash-preview-thinking-high | 0.06 | 0.00 | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.00 | 0.00 | 0.00 |
| mistral-large-2512 | 0.00 | 0.00 | 0.00 |
| magistral-medium-2509 | 0.00 | 0.00 | 0.00 |

### B-L1 — prefill, audit hint

| Model | D | DC | DCF |
|---|---:|---:|---:|
| claude-opus-4-6-reasoning-high | 0.76 | 0.75 | **0.60** |
| gpt-5.4-reasoning-xhigh | 0.53 | 0.53 | 0.47 |
| gemini-3.1-pro-preview-thinking-high | 0.38 | 0.37 | 0.32 |
| claude-sonnet-4-6-reasoning-high | 0.27 | 0.26 | 0.23 |
| gemini-3-flash-preview-thinking-high | 0.14 | 0.06 | 0.00 |
| mistral-large-2512 | 0.10 | 0.00 | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.01 | 0.00 | 0.00 |
| magistral-medium-2509 | 0.00 | 0.00 | 0.00 |

### C-L0 — single-turn, no audit hint

| Model | D | DC | DCF |
|---|---:|---:|---:|
| gpt-5.4-reasoning-xhigh | 0.08 | 0.07 | **0.07** |
| claude-opus-4-6-reasoning-high | 0.03 | 0.03 | 0.02 |
| gemini-3-flash-preview-thinking-high | 0.13 | 0.01 | 0.00 |
| claude-sonnet-4-6-reasoning-high | 0.00 | 0.00 | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.00 | 0.00 | 0.00 |
| mistral-large-2512 | 0.00 | 0.00 | 0.00 |
| magistral-medium-2509 | 0.00 | 0.00 | 0.00 |

### C-L1 — single-turn, audit hint

| Model | D | DC | DCF |
|---|---:|---:|---:|
| claude-opus-4-6-reasoning-high | 0.59 | 0.58 | **0.45** |
| gpt-5.4-reasoning-xhigh | 0.18 | 0.16 | 0.14 |
| claude-sonnet-4-6-reasoning-high | 0.10 | 0.10 | 0.06 |
| gemini-3-flash-preview-thinking-high | 0.07 | 0.00 | 0.00 |
| mistral-large-2512 | 0.02 | 0.00 | 0.00 |
| gpt-5.4-mini-reasoning-xhigh | 0.00 | 0.00 | 0.00 |
| magistral-medium-2509 | 0.00 | 0.00 | 0.00 |

Per-difficulty breakdowns are in each available cell's `score_summary.json`.

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
