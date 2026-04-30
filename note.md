# Metric Design Memo, Part I: DCF Is Not The Right Headline

Update: the current scoring pipeline now uses a simpler two-metric headline:
`error_recovery_rate` and `final_answer_correct_rate`. This memo is kept as
historical rationale for why DCF was demoted.

This is Part I of a two-part metric memo.

- Part I: why `detect_correct_finaloutput_correct_rate` (DCF) is the wrong
  headline metric
- Part II: why strict `mechanical_correct_rate` is better, but still needs a
  recovery-aware companion metric for later self-repair

## TL;DR

The current headline metric — `detect_correct_finaloutput_correct_rate` (DCF) —
silently conflates *two independent things*: whether the model **explicitly
acknowledges** the error, and whether it **mechanically produces a correct
derivation**. That conflation penalizes models that fix errors silently (no
meta-commentary, just right answer with right work) as if they failed.

We now record a cleaner separate metric — `mechanical_correct_rate` — and that
is a much better primary deterministic metric than DCF.

But that is not the end of the story. Part II argues that strict mechanical
correctness still misses an important behavior: a model making a *new* mistake
during continuation and later fixing it. So the final recommendation is not
"DCF -> mechanical only"; it is:

- use `mechanical_correct_rate` for strict correctness
- add a second deterministic `repair_aware_correct_rate` for genuine self-repair
- keep judge-backed metrics as diagnostics, not headlines

## Three competing views of "correct"

For the same 100 B/L0 predictions from Magistral Medium 1.2:

| Lens | Rate | What it checks |
|---|---|---|
| Judge DCF (current headline) | 0/100 | Model *explicitly* said "step Y was wrong, it should be X". |
| Regex baseline (any-recovery) | 18/100 | Final emoji == GT, regardless of how. (1 loud + 17 silent) |
| Validator: mechanical correct | 8/100 | Final emoji == GT **AND** every step reduces correctly under the operator table **AND** consecutive steps are continuous. |

All three are defensible; they answer different questions. Picking "the" one
without saying so is what makes the benchmark misleading.

## Why DCF is problematic

DCF = `judge.detected_error AND judge.corrected_step_y AND validator.derivation_valid AND validator.terminal_matches_gt`.

Those first two are judged by an LLM reading the continuation text looking for
explicit "I see the error" / "step Y should be X" statements. The last two are
deterministic math on the derivation chain.

Problems:

1. **Punishes silent fixers.** Magistral (and most Mistral models) tends to
   catch an upstream error, quietly use the corrected value, and continue —
   producing a valid derivation that ends on GT without ever saying "wait, that
   was wrong". Judge sees no acknowledgment → DCF = 0. But the work is correct.
2. **LLM judge noise.** The first two axes are subjective. A judge run with a
   different model (or a rerun with the same model) can flip booleans on
   borderline cases.
3. **Conflates two things that should be measured separately.** "Did it get
   the right answer with valid work?" is a different question from "did it
   narrate its self-correction out loud?". DCF can't tell you which one the
   model failed on.

## Why mechanical_correct is a cleaner primary metric

`mechanical_correct = validator.derivation_valid AND validator.terminal_matches_gt`.

It's:

- **Deterministic** — no judge, no LLM noise.
- **Style-agnostic** — silent fixers and loud fixers get scored identically if
  their final chain is sound.
- **Luck-proof** — catches compensating errors (right final emoji via a bad
  step along the way) that outcome-only scoring would credit.

It answers the plain-English question: *did the model produce a correct
derivation ending at the right answer?* That is a much better core question
than "did the model explicitly narrate the correction out loud?"

## Limitation To Keep In Mind

It **penalizes mid-stream self-correction**. If the model makes a *new*
mistake mid-continuation, catches it, retracts, re-emits the step correctly,
and arrives at the right answer, the validator flags the chain as invalid:

- The wrong step fails per-step validity (`after ≠ evaluate(before)`).
- The re-emitted step fails continuity (its `before` doesn't match the
  previous step's `after`).

This is exactly the limitation addressed in Part II. Strict mechanical
correctness is still useful, but it is not a full measure of self-correction.
If later self-repair is common enough to matter, we should add a second
deterministic metric rather than weakening the strict one.

## Recommendation

The combined recommendation across Part I and Part II is:

- **Do not use DCF as the headline.**
- **Promote deterministic metrics over judge-mixed metrics.**
- **Use `mechanical_correct_rate` as the strict correctness metric.**
- **Add `repair_aware_correct_rate` if later self-repair is a meaningful pattern.**
- **Keep `detect_rate`, `detect_correct_rate`, and DCF as diagnostics about
  style and explicitness, not as the primary claim about correctness.**

In short:

- DCF is too narrow and too judge-dependent to be the headline
- strict mechanical correctness is the right baseline
- strict correctness alone is still incomplete if we care about later self-repair
