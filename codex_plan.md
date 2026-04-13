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
  - easy: `X = 6`
  - medium: `X = 8`
  - hard: `X = 10`
  - expert: `X = 14`
- [ ] **Realized-X floor of 4.** `chain_generator.generate_chain` is best-effort (`±2` early-exit at [chain_generator.py:348](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/chain_generator.py:348)), so a target of 6 can return anything in `[4, 8]` and — if the sampler never hits the `±2` window in 50 attempts — occasionally lower. At dataset-generation time, reject any row with `chain_length_x < 4` with reason `chain_too_short` so the midpoint policy never collapses to `Y = 1`. The rejection counter per difficulty tells us whether easy is viable or needs to be dropped.
- [ ] **Over-generate** by roughly 2× the final dataset size and log rejection reasons per attempt (too-early / too-late error_step, convergent wrong branch, cascade-dies-immediately, no valid eligible step in window, chain_too_short).
- [ ] **Self-detection definition:** score two variants and report both.
  - Loose: any expressed doubt anywhere in the continuation (regex on `wait|error|mistake|wrong|correction|let me (re)?check|actually`).
  - Strict: explicitly flags the bad prefix step (judge or structured check).
- [ ] **Outcome buckets (four, plus a noise bucket):** detect+recover, silent recovery, detect-only (no recovery), blind continuation on wrong branch, off-the-rails (reaches neither ground truth nor wrong-branch output).
- [ ] **Prefill capability:** first pilot is Anthropic-only (native assistant prefill). Add a `supports_assistant_prefill` flag in the model registry before extending to OpenAI / Gemini / Mistral; report those separately so the asymmetry is visible, not hidden.
- [ ] **Prefill formatting:** cutoff must not end on a terminal marker (`Final Output:`, trailing blank line) — the prefill ends mid-derivation so the model's continuation is forced.
- [ ] **Turn-2 user message:** fixed as `Please continue.` for the pilot. Logged as a variable, not a final decision — candidate for a later ablation.
- [ ] **Single-turn rendering is a view, not a field.** For evaluation channels that don't support assistant prefill (e.g. Kaggle Benchmark, which only accepts a single user prompt), the multi-turn record is collapsed into one prompt at request time via a formatter helper — `turn_1_user` body + a `=== WORK SO FAR ===` block carrying the prefill. The dataset row stores only the multi-turn pieces; the single-turn string is derived from them. Avoids schema duplication and keeps prefill / single-turn results comparable per row but reported as separate runs (the meta-frame differs: "I am extending my own thought" vs. "I am being shown my prior thought by someone else").

## Phase 0: Baseline Hygiene

- [x] Fix the existing failing test caused by the stray leading character in [scripts/evaluate_anthropic.py](/Users/huydang/Desktop/Emoji-Bench/scripts/evaluate_anthropic.py:1).
- [x] Re-run `pytest` to get back to a clean baseline before the pivot work continues.

## Phase 1: Define the New Dataset Schema — DONE (commit `12076b8`)

- [x] Add a continuation-style dataset record format instead of relying on a single `prompt` string. _(See `continuation_record` in [emoji_bench/continuation_benchmark.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/continuation_benchmark.py).)_
- [x] Store the turn structure explicitly:
  - `turn_1_user`
  - `turn_1_assistant_prefill`
  - `turn_2_user`
- [x] Store evaluation ground truth explicitly:
  - `ground_truth_final_output`
  - `wrong_branch_final_output`
  - `chain_length_x` (total steps `X` in the clean chain)
  - `prefill_error_step` (the `error_step`)
  - `prefill_cutoff_step` (equal to `error_step` under the locked policy, kept as a separate column so the policy can loosen later without a schema migration)
  - `has_prefill_error`
- [x] Preserve debug/repro fields already used by the repo:
  - `system_json`
  - `system_seed`
  - `chain_seed`
  - `error_seed`
  - `difficulty`
  - `error_type`

## Phase 2: Build the Prefill/Continuation Generator — MOSTLY DONE (commit `12076b8`)

- [x] Add a formatter for the first user turn: rules + original expression only. _(`format_continuation_turn_1_user` in [emoji_bench/continuation_formatter.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/continuation_formatter.py).)_ Includes an explicit `Step N: ... [by <rule>]` format instruction so the prefill format is primed, not just observed.
- [x] Add a formatter for the assistant prefill: steps `1..Y` only. _(`format_continuation_prefill` in the same module.)_
- [x] Keep the prefill in the current numbered-step format first, instead of adding freeform "natural CoT" prose.
- [x] Choose the cutoff step `Y` per the locked policy: `error_step ≈ ⌊X/2⌋ ± 1`, `prefill_cutoff_step = error_step`. _(`_preferred_error_steps` + `generate_continuation_instance` in [emoji_bench/continuation_benchmark.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/continuation_benchmark.py).)_
- [x] Ensure the injected error occurs within the visible prefix (it is the last prefilled step by construction).
- [ ] Ensure the prefix ends before the full chain finishes, with at least `⌈X/2⌉` steps remaining. _Partial: the cascading injector guarantees ≥ 1 remaining step, but the `⌈X/2⌉` runway floor is not yet enforced — deferred to Phase 3 as a rejection filter alongside `chain_too_short`._
- [x] Ensure the prefill string does not end on a terminal marker (`Final Output:`, trailing blank line) that would short-circuit continuation. _(Covered by `test_prefill_string_does_not_end_on_terminal_marker`.)_

## Phase 3: Generate the Right Kind of Error — DONE (commit `cfa5ea6`)

- [x] Reuse cascading wrong-result injection as the primary mechanism. _(Via `inject_cascading_wrong_result` in [emoji_bench/error_injector.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/error_injector.py:255), wired through `generate_continuation_instance`.)_
- [x] Do not use reconvergent injection for the main benchmark.
- [x] Add validation that the erroneous recomputed suffix ends at a different final symbol than the clean chain. _(Guaranteed at injection time by the cascading injector itself; rejected as `cascade_convergent` if no non-convergent slot exists for the chosen seed.)_
- [x] Save both the clean full chain and the erroneous full chain for offline inspection and scoring. _(The full mutated chain lives on `ContinuationInstance.mutated_chain`; the dataset record carries `wrong_branch_final_output` and the seeds needed to reconstruct the full chains.)_
- [x] Reject unusable examples where the error appears too late or leaves no meaningful continuation after step `Y`. _(Runway floor of `⌈X/2⌉` enforced as `R_INSUFFICIENT_RUNWAY` in [emoji_bench/continuation_dataset.py](/Users/huydang/Desktop/Emoji-Bench/emoji_bench/continuation_dataset.py).)_
- [x] Simulate the erroneous continuation to its terminal symbol inside the injector and reject on convergence before emitting the row (don't only filter post-hoc). _(The cascading injector reduces the candidate suffix internally and only returns mutations whose final symbol differs from the clean final symbol — see `inject_cascading_wrong_result:288–296`.)_
- [x] Over-generate ~2× the target sample size and log per-attempt rejection reasons. _(`generate_continuation_dataset_records` tracks 6 reasons per difficulty: `chain_too_short`, `insufficient_runway`, `no_eligible_in_chain`, `no_eligible_in_window`, `cascade_convergent`, `other_injector_error`. Counts surfaced through the new optional `DatasetManifest.rejection_counts` field.)_

**Pilot smoke (count=100, master_seed=20260413):** 100/100 produced (25 per difficulty), ~31% rejection. `chain_too_short` = 0 across all difficulties (target bumps work); `insufficient_runway` is dominant on medium (19); `cascade_convergent` is rare (1–5 per difficulty); `no_eligible_in_*` = 0.

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

- [x] First: fix the current failing test. _(Commit `12076b8`.)_
- [x] Second: add the new dataset schema and continuation formatter. _(Commit `12076b8` — Phase 1 complete; Phase 2 complete apart from the `⌈X/2⌉` runway floor, addressed in Phase 3.)_
- [x] Third: dataset-level generation with rejection logging. _(Commit `cfa5ea6` — Phase 3 complete. 100-row pilot generates clean.)_
- [ ] Fourth: add multi-turn provider evaluation support.
- [ ] Fifth: add scoring and reporting.
- [ ] Sixth: run the 100-example pilot against models.

## Notes

- [ ] The current codebase already has the core machinery needed for this pivot: system generation, derivation generation, and cascading error injection.
- [ ] The main change is the benchmark interface: from full-chain verification to multi-turn continuation from a bad prefix.
- [ ] For the first pass, prioritize controlled formatting over realism. Once the pipeline works, we can decide whether to make the assistant prefill more naturalistic.
