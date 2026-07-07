# Next Session Runbook: Control Evals + Survival Headline

Self-contained instructions for finishing the control-condition work. Written to
be executed by a fresh session (any model) with repo access and provider API
keys in `.env`. Everything below that costs API money is clearly marked; all
code it depends on is already implemented and tested (218 tests passing).

## Background: what this benchmark measures and why controls matter

Emoji-Bench evaluates **context-pollution recovery**: each instance is a
3-message conversation where the assistant's prior turn (a step-by-step
derivation in a procedurally generated emoji algebra) contains one planted
wrong step, and the model is told only "Please continue." The headline so far
(`final_answer_correct_rate`) says whether the model ends at the correct final
symbol.

Two findings from the research review (July 2026) motivate this runbook:

1. **Behavior decomposition.** Scoring now classifies *how* each model handles
   the bad prefill by comparing its first parsed step (expression-AST equality)
   against four reference states. On the checked-in L0 runs: GPT-5.4/5.5 re-emit
   the erroneous step corrected (~96% "in-place correction"); Claude models
   continue from a silently repaired state; weak models continue from the
   corrupted state and cascade. True restarts are rare. This is already
   computed — see `behavior_mix` in every `score_summary.json` and
   `artifacts/plots/b_behavior_mix_l{0,1}.png`.
2. **The raw headline conflates recovery with base task ability.** A model
   scoring 10% might be blindly trusting errors *or* unable to do the task at
   all. The fix is the **survival rate**: corrupted-condition accuracy divided
   by clean-control accuracy on the same instances. The clean-control dataset
   exists (`artifacts/emoji-bench-dataset-100-clean-control/` — same instances,
   clean prefill cut at the same step, ids suffixed `-control`) and **three
   models already have complete control runs** under
   `artifacts/evals-clean/<model>-clean-B-L0/` (gpt-5.4-mini, gpt-5.4-nano,
   magistral-medium-2509 — scored, with `survival_summary.json` computed in
   the corresponding corrupted cells). The proof of value is already in:
   magistral solves **84%** clean but survives only **21%** of planted errors
   [CI 0.14–0.33] — its bad headline is context-trust failure, not task
   inability. The job below is running the remaining ~10 models.

## Step 1 — Run the missing clean-control evals (API cost: ~1,100 calls at L0)

Controls already done (skip these): `gpt-5.4-mini-reasoning-xhigh`,
`gpt-5.4-nano-reasoning-xhigh`, `magistral-medium-2509` — complete 100/100
runs in `artifacts/evals-clean/<model>-clean-B-L0/`. Follow that existing
naming convention for the rest:

```bash
MODELS=(
  "claude-opus-4-8-reasoning-max"   # note: in MODEL_CONFIGS; add to run.sh MODELS if missing
  "claude-opus-4-7-reasoning-max"
  "claude-opus-4-6-reasoning-max"
  "claude-sonnet-4-6-reasoning-max"
  "gpt-5.5-reasoning-max"
  "gpt-5.2-reasoning-xhigh"
  "gpt-5.4-reasoning-xhigh"
  "gemini-3.1-pro-preview-thinking-high"
  "gemini-3-flash-preview-thinking-high"
  "grok-4.3-reasoning-high"
  "mistral-large-2512"
)
for model in "${MODELS[@]}"; do
  python scripts/evaluate_continuation.py \
    artifacts/emoji-bench-dataset-100-clean-control \
    --model "$model" --mode prefill --turn-2-prompt-level 0 \
    --output-dir "artifacts/evals-clean/${model}-clean-B-L0" \
    --max-concurrent 8
done
```

Notes:
- `--output-dir` is required here; the default naming would collide with the
  corrupted-condition cells.
- Runs are resumable (predictions.jsonl is append-only with a seen-set); rerun
  the same command after failures.
- Cost scale: reasoning-max/xhigh models emit long outputs; budget accordingly.

## Step 2 — Score the control runs (free, offline)

```bash
for model in "${MODELS[@]}"; do
  python scripts/score_continuation.py "artifacts/evals-clean/${model}-clean-B-L0" \
    --dataset-path artifacts/emoji-bench-dataset-100-clean-control
done
```

(`--dataset-path` is usually optional — scoring auto-resolves the dataset from
`summary.json`, with an `artifacts/`-relative fallback for runs made on other
machines — but passing it explicitly never hurts.)

## Step 3 — Compute the survival headline (free, offline)

```bash
for model in "${MODELS[@]}"; do
  python scripts/compute_survival.py \
    "artifacts/evals/${model}-B-L0" \
    "artifacts/evals-clean/${model}-clean-B-L0"
done
```

Already computed for the three existing controls (see
`artifacts/evals/<model>-B-L0/survival_summary.json`):

| model | clean | corrupted | survival [95% CI] |
|---|---|---|---|
| gpt-5.4-mini-reasoning-xhigh | 0.97 | 0.80 | 0.82 [0.74, 0.92] |
| gpt-5.4-nano-reasoning-xhigh | 0.98 | 0.80 | 0.82 [0.74, 0.90] |
| magistral-medium-2509 | 0.84 | 0.18 | 0.21 [0.14, 0.33] |

Each corrupted cell gets a `survival_summary.json` containing:
- `survival_rate` (+ Katz 95% CI): corrupted accuracy / clean accuracy — the
  new headline
- `conditional_recovery_rate` (+ Wilson CI): P(correct under corruption |
  correct under clean), the per-instance paired variant. Expect wide CIs for
  weak models (small clean-solved denominators) — report the interval, do not
  hide the entry.

## Step 4 — Known gaps to fill while you're spending API budget

- Missing/incomplete L1 cells (corrupted condition):
  - `claude-sonnet-4-6-reasoning-max` L1 (missing entirely)
  - `gemini-3-flash-preview-thinking-high` L1 (missing entirely)
  - `gpt-5.4-mini-reasoning-xhigh` L1 (partial: 52/100 predictions)
  - `gpt-5.4-nano-reasoning-xhigh` L1 (partial: 92/100 predictions)
  Rerunning `evaluate_continuation.py` on these resumes from what exists.
- `claude-opus-4-8-reasoning-max` is configured and has L0/L1 results but is
  not in `run.sh`'s MODELS array — add it.

## Deferred design work (agreed, not yet implemented — with rationale)

1. **No-prefill third arm**: send `turn_1_user` alone (raw solve ability, no
   partial trace). Completes the 3-arm design: no-prefill → clean prefill →
   corrupted prefill; each pairwise delta isolates one effect. Small code
   change (new mode in `emoji_bench/providers/continuation.py`).
2. **Detection probe turn**: after the continuation, ask "Was there an error
   in the steps above? Which step?" Measures detection *behaviorally* —
   the current `detected_loose` regex measures verbalization and undercounts
   reasoning models that fix silently (documented failure: GPT-5.2 L0
   recovers 58% but verbalizes 10%).
3. **Adaptive "until-N-solved" design**: run the clean condition over a large
   generated pool until each model banks N=100 clean solves, then corrupt
   those. Fixes the small-denominator blockiness of `conditional_recovery_rate`
   for weak models. Requires a pool dataset + a runner mode.
4. **1k reference dataset + generator-first README framing**: the benchmark is
   an infinite seeded generator; the checked-in 100 rows are one slice. A
   larger hidden split (private master seed) is the anti-overfitting story.
5. **On-policy variant** (highest research value): inject the minimal error
   into the model's *own* correct derivation instead of a template-generated
   one; `emoji_bench/domain/continuation_validator.py` can already parse and
   validate model-produced steps, which is the enabling piece.

## What changed in this codebase (July 2026 session)

- `emoji_bench/scoring/behavior_classifier.py` (new): AST-grounded behavior
  buckets (`inplace_correction`, `continue_from_corrected`,
  `continue_from_corrupted`, `full_restart`, `no_steps`, `other`).
- `emoji_bench/scoring/stats.py` (new): Wilson + Katz intervals.
- `scripts/score_continuation.py`: joins predictions to the source dataset
  (auto-resolved from `summary.json`, with an `artifacts/`-relative fallback
  for artifacts moved between machines), emits `behavior_mix`,
  `derivation_valid_rate`, `*_ci95`, and `chance_rate` (1/n_symbols — 17–33%,
  read raw rates against it).
- `scripts/compute_survival.py` (new): survival headline (this runbook).
- `scripts/plot_b_final_answer.py`: CI error bars + behavior-mix stacked plot.
- `emoji_bench/model_registry.py` / providers / runner: `temperature` is a
  `ModelConfig` field (0.0 for Mistral, matching its previously hardcoded
  transport; None = provider default elsewhere), recorded per prediction row
  alongside `max_output_tokens`.
- `continuation_record()` re-emits `condition`, `has_prefill_error`,
  `turn_2_user`, `prefill_cutoff_step` (schema now matches the checked-in
  dataset; do NOT regenerate `artifacts/emoji-bench-dataset-100` — existing
  evals reference it).
- All checked-in eval cells rescored; headline rates unchanged, new fields
  added. Stale tests for deleted runner scripts removed; `run.sh` tests
  updated for the 13-model list.
