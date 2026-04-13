# Codex Plan

## Goal

Pivot Emoji-Bench from a single-turn "audit this full derivation" benchmark into a multi-turn continuation benchmark:

1. User gives the formal system and the original expression.
2. Assistant is prefilled with steps `1..Y`, where step `Y` contains an error.
3. User says: `Please continue.`
4. Model continues from the bad prefix.
5. We measure whether it self-detects the mistake and whether it still recovers the correct final answer.

## Hard Constraint

- [ ] Use the cascading-error setting, not reconvergent.
- [ ] The final output of the wrong branch must be different from the clean ground-truth final output.
- [ ] This is the opposite of [emoji_bench/reconvergent_error_injector.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/reconvergent_error_injector.py:63), which preserves the final result.
- [ ] The continuation dataset should reject any row where `final_output_with_error == ground_truth`.

## Locked Design Decisions

- [ ] **Y-offset policy:** `error_step ≈ ⌊X/2⌋`, with a ±1 jitter tolerance so the injector has room to find a valid cascading slot. `prefill_cutoff_step = error_step` — the error is always the last prefilled step, and the model's first generated step is the one that would either propagate or catch it.
- [ ] **Remaining runway:** always at least `⌈X/2⌉` steps left after the cutoff, so "Please continue." has genuine work to do.
- [ ] **Chain length targets (bumped from reconvergent defaults so the midpoint is meaningful):**
  - easy: `X = 4`
  - medium: `X = 6`
  - hard: `X = 8`
  - expert: `X = 10`
- [ ] **Over-generate** by roughly 2× the final dataset size and log rejection reasons per attempt (too-early / too-late error_step, convergent wrong branch, cascade-dies-immediately, no valid eligible step in window).
- [ ] **Self-detection definition:** score two variants and report both.
  - Loose: any expressed doubt anywhere in the continuation (regex on `wait|error|mistake|wrong|correction|let me (re)?check|actually`).
  - Strict: explicitly flags the bad prefix step (judge or structured check).
- [ ] **Outcome buckets (four, plus a noise bucket):** detect+recover, silent recovery, detect-only (no recovery), blind continuation on wrong branch, off-the-rails (reaches neither ground truth nor wrong-branch output).
- [ ] **Prefill capability:** first pilot is Anthropic-only (native assistant prefill). Add a `supports_assistant_prefill` flag in the model registry before extending to OpenAI / Gemini / Mistral; report those separately so the asymmetry is visible, not hidden.
- [ ] **Prefill formatting:** cutoff must not end on a terminal marker (`Final Output:`, trailing blank line) — the prefill ends mid-derivation so the model's continuation is forced.
- [ ] **Turn-2 user message:** fixed as `Please continue.` for the pilot. Logged as a variable, not a final decision — candidate for a later ablation.

## Phase 0: Baseline Hygiene

- [x] Fix the existing failing test caused by the stray leading character in [scripts/evaluate_anthropic.py](/Users/huydang/Desktop/Emoji-Bench/scripts/evaluate_anthropic.py:1).
- [x] Re-run `pytest` to get back to a clean baseline before the pivot work continues.

## Phase 1: Define the New Dataset Schema

- [ ] Add a continuation-style dataset record format instead of relying on a single `prompt` string.
- [ ] Store the turn structure explicitly:
  - `turn_1_user`
  - `turn_1_assistant_prefill`
  - `turn_2_user`
- [ ] Store evaluation ground truth explicitly:
  - `ground_truth_final_output`
  - `wrong_branch_final_output`
  - `chain_length_x` (total steps `X` in the clean chain)
  - `prefill_error_step` (the `error_step`)
  - `prefill_cutoff_step` (equal to `error_step` under the locked policy, kept as a separate column so the policy can loosen later without a schema migration)
  - `has_prefill_error`
- [ ] Preserve debug/repro fields already used by the repo:
  - `system_json`
  - `system_seed`
  - `chain_seed`
  - `error_seed`
  - `difficulty`
  - `error_type`

## Phase 2: Build the Prefill/Continuation Generator

- [ ] Add a formatter for the first user turn: rules + original expression only.
- [ ] Add a formatter for the assistant prefill: steps `1..Y` only.
- [ ] Keep the prefill in the current numbered-step format first, instead of adding freeform "natural CoT" prose.
- [ ] Choose the cutoff step `Y` per the locked policy: `error_step ≈ ⌊X/2⌋ ± 1`, `prefill_cutoff_step = error_step`.
- [ ] Ensure the injected error occurs within the visible prefix (it is the last prefilled step by construction).
- [ ] Ensure the prefix ends before the full chain finishes, with at least `⌈X/2⌉` steps remaining.
- [ ] Ensure the prefill string does not end on a terminal marker (`Final Output:`, trailing blank line) that would short-circuit continuation.

## Phase 3: Generate the Right Kind of Error

- [ ] Reuse cascading wrong-result injection as the primary mechanism.
- [ ] Do not use reconvergent injection for the main benchmark.
- [ ] Add validation that the erroneous recomputed suffix ends at a different final symbol than the clean chain.
- [ ] Save both the clean full chain and the erroneous full chain for offline inspection and scoring.
- [ ] Reject unusable examples where the error appears too late or leaves no meaningful continuation after step `Y`.
- [ ] Simulate the erroneous continuation to its terminal symbol inside the injector and reject on convergence before emitting the row (don't only filter post-hoc).
- [ ] Over-generate ~2× the target sample size and log per-attempt rejection reasons (too-early / too-late / convergent / cascade-dies / no-eligible-step) so we can tune yield by difficulty.

## Phase 4: Add Multi-Turn Evaluation Support

- [ ] Extend provider request building so evaluators can send message lists, not just one prompt string.
- [ ] Add a `supports_assistant_prefill` capability flag to the model registry. Pilot is Anthropic-only; other providers ship behind the flag and their results are reported separately so the prefill asymmetry is visible in the numbers, not hidden.
- [ ] Support the conversation shape:
  - system prompt
  - user turn 1
  - assistant prefill
  - user turn 2 (`Please continue.`)
- [ ] Capture raw continuation text from the model.
- [ ] Keep the old structured-output evaluator intact for the legacy benchmark.
- [ ] Add a separate continuation evaluator path rather than overloading the current `has_error/error_step` path.

## Phase 5: Define Scoring

- [ ] Add a final-answer extractor, likely anchored on a required marker such as `Final Output: X`.
- [ ] Score whether the model's final answer matches the clean ground truth.
- [ ] Add an explicit self-detection signal:
  - regex-first for words like `error`, `mistake`, `wrong`, `correction`
  - judge fallback only for ambiguous cases
- [ ] Separate these outcomes:
  - explicit self-detection
  - correct final recovery
  - explicit detect + correct recovery
  - silent recovery
  - blind continuation on the wrong branch
- [ ] Keep the raw text needed to audit edge cases manually.

## Phase 6: Reporting

- [ ] Add aggregate metrics for the continuation benchmark.
- [ ] Break down by model and difficulty.
- [ ] Report both:
  - self-detection rate (loose and strict — see Locked Design Decisions)
  - final-answer recovery rate
- [ ] Add conditional recovery metrics, especially recovery given explicit detection.
- [ ] Emit a confusion matrix across the five outcome buckets, not just scalar rates.
- [ ] Save a small sample of interesting trajectories for qualitative review.

## Phase 7: Pilot Run

- [ ] Start with a small test-only pilot of about 100 examples.
- [ ] Run one or two models first before scaling out.
- [ ] Inspect prompt failures, extraction failures, and ambiguous self-detection cases.
- [ ] Tighten formatting and scoring before any larger benchmark run.

## Recommended Execution Order

- [x] First: fix the current failing test.
- [ ] Second: add the new dataset schema and continuation formatter.
- [ ] Third: add multi-turn provider evaluation support.
- [ ] Fourth: add scoring and reporting.
- [ ] Fifth: generate the 100-example pilot and run initial models.

## Notes

- [ ] The current codebase already has the core machinery needed for this pivot: system generation, derivation generation, and cascading error injection.
- [ ] The main change is the benchmark interface: from full-chain verification to multi-turn continuation from a bad prefix.
- [ ] For the first pass, prioritize controlled formatting over realism. Once the pipeline works, we can decide whether to make the assistant prefill more naturalistic.
