# Metric Design Memo, Part II: Strict Mechanical Correctness Still Misses Self-Repair

This is Part II of a two-part metric memo.

- Part I: DCF should not be the headline because it conflates explicit
  acknowledgment with actual derivation correctness
- Part II: strict `mechanical_correct_rate` is a much better baseline, but it
  still misses a second behavior we may care about: later self-repair after a
  newly introduced continuation mistake

## TL;DR

`mechanical_correct_rate` is a much better primary deterministic metric than
DCF, but it has its own blind spot:

- It rewards **clean, mechanically correct derivations**.
- It does **not** reward a model that makes a new mistake during continuation,
  then later catches and fixes that mistake by itself.

That means it cannot distinguish between:

1. a model that goes off the rails and never recovers, and
2. a model that briefly goes wrong, then successfully self-repairs and lands on
   a correct final derivation.

Those are different behaviors. The second is arguably one of the most
interesting capabilities this benchmark could measure.

So the recommendation is not to replace DCF with `mechanical_correct_rate` and
stop there. The better endpoint is:

- `mechanical_correct_rate` for strict clean correctness
- `repair_aware_correct_rate` for successful self-repair
- judge-backed detection metrics as diagnostics

## The Problem

Right now:

`mechanical_correct = validator.derivation_valid AND validator.terminal_matches_gt`

This is good for measuring strict correctness. It is deterministic,
style-agnostic, and resistant to lucky final answers.

But it assumes the continuation is a **single clean linear chain**. If the model:

1. emits an incorrect step,
2. notices the problem,
3. retracts or rewrites that step,
4. and then finishes correctly,

the current validator still marks the row as false.

Why?

- The bad step fails per-step validity.
- The rewritten step often fails continuity relative to the previously emitted bad step.
- The parser/validator treats the trace as one flat sequence of emitted lines,
  not as a derivation with revisions.

So `mechanical_correct_rate` currently answers:

*Did the model emit a clean correct chain?*

It does **not** answer:

*Could the model recover from its own later mistake and still produce a correct
final derivation?*

## Why This Matters

This is not just a minor edge case. It goes to the core behavioral question.

If the benchmark claims to study whether models can "catch and fix" their own
mistakes, then later self-repair during continuation is important behavior.
Under the current strict validator, that behavior is collapsed into failure.

That would make the benchmark good at measuring:

- strict derivation discipline
- clean recovery from the seeded prefill error

but weaker at measuring:

- self-repair after a newly introduced continuation error

## What We Should Not Do

We should **not** try to force one metric to cover both clean correctness and
recovery-after-error.

We should also avoid weakening the benchmark to:

`final_output == ground_truth`

or anything close to "the final answer is right, so count it."

That would reopen the compensating-error / lucky-answer loophole and throw away
one of the strongest properties of the current deterministic validator.

## Candidate Approaches

### 1. Keep The Current Strict Metric, And Add A Second Recovery-Aware Metric

This is the cleanest approach.

Keep:

- `mechanical_correct_rate`: strict, no emitted mistake forgiven

Add:

- `repair_aware_correct_rate`: allows the model to revise earlier emitted steps,
  as long as the final canonical derivation is mechanically valid and reaches GT

This preserves an important distinction:

- "never slipped"
- "slipped, but recovered"

That distinction is valuable and should not be collapsed.

### 2. Retraction-Aware Parsing With "Last Version Wins"

This is the most promising implementation strategy for the new recovery-aware metric.

Possible rule:

1. Parse all `Step N:` lines in order.
2. If a step number appears multiple times, keep the **last** version as canonical.
3. Reconstruct the canonical chain from those retained steps, sorted by step number.
4. Validate that reconstructed chain using the same deterministic checks:
   per-step validity, continuity, and terminal match to GT.

Under this scheme, a row can fail strict `mechanical_correct`, but still pass
`repair_aware_correct` if the final retained chain is sound.

This keeps the metric deterministic while still crediting genuine recovery.

### 3. Add An Explicit Self-Repair Flag/Rate

In addition to the strict and repair-aware correctness rates, we could report:

- `self_repair_rate`

For example, a row could count as self-repaired if:

- the raw emitted trace contains an earlier invalid or discontinuous step, and
- the repair-aware canonical chain passes validation

This would separate:

- clean correct runs
- recovered runs
- failed runs

### 4. Treat Explicit Verbal Correction As A Separate Diagnostic Only

The judge-backed metrics are still useful, but they should stay separate from
the deterministic correctness metrics.

That means:

- `detect_rate`
- `detect_correct_rate`

remain useful for understanding whether the model *narrated* the correction,
but they should not be mixed into the definition of strict or repair-aware
correctness.

## Recommendation

The best combined reporting structure across Part I and Part II is probably:

- `mechanical_correct_rate`
- `repair_aware_correct_rate`
- `self_repair_rate`
- `detect_rate`
- `detect_correct_rate`
- `detect_correct_finaloutput_correct_rate` as a narrow judge-backed diagnostic,
  not the headline

Interpretation:

- `mechanical_correct_rate` = strict clean correctness
- `repair_aware_correct_rate` = correctness after allowing revision/self-repair
- `repair_aware_correct_rate - mechanical_correct_rate` = how often models
  recover from their own newly introduced continuation mistakes
- `detect_*` rates = whether they explicitly verbalize that recovery

## Why This Is Better

This avoids a false choice between two bad options:

1. **Strict-only** scoring, which misses meaningful recovery behavior
2. **Outcome-only** scoring, which over-credits lucky or messy traces

Instead, it gives two deterministic views:

- a strict view of derivation quality
- a recovery-aware view of self-repair ability

That seems closer to the actual behavioral question.

## Open Questions / Implementation Notes

- We should inspect real predictions first to see how often this pattern occurs.
  If retract-and-restate behavior is rare, the extra parser complexity may not
  be worth it.
- "Last version wins" is probably the simplest recovery rule, but we may still
  need guardrails for messy cases where the model repeats step numbers without
  clear retraction intent.
- The recovery-aware metric should still require a mechanically valid canonical
  chain. It should not degrade into "final answer matches GT".
- We should keep the current strict metric even if we add recovery-aware scoring.
  Otherwise we lose the distinction between clean derivations and repaired ones.
- If this is implemented, the README should be explicit that the benchmark now
  reports two deterministic correctness views: strict correctness and
  recovery-aware correctness.

## Bottom Line

`mechanical_correct_rate` is still useful, but it should be understood as a
**strict correctness** metric, not a full measure of self-correction.

If we want to capture the behavior "the model makes a new mistake, then later
fixes it by itself," we likely need a second deterministic metric:

- `repair_aware_correct_rate`

and possibly a companion diagnostic:

- `self_repair_rate`

Together with Part I, the full proposal becomes:

- demote DCF from headline status
- keep strict `mechanical_correct_rate` as the deterministic baseline
- add `repair_aware_correct_rate` if later self-repair turns out to be common
  enough to justify the added parser complexity
