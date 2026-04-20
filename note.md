# Metric problem: what does "catch and fix" actually mean?

## TL;DR

The current headline metric — `detect_correct_finaloutput_correct_rate` (DCF) —
silently conflates *two independent things*: whether the model **explicitly
acknowledges** the error, and whether it **mechanically produces a correct
derivation**. That conflation penalizes models that fix errors silently (no
meta-commentary, just right answer with right work) as if they failed.

We now record a cleaner separate metric — `mechanical_correct_rate` — and
**that should be the headline**. D and DC stay as diagnostics.

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

## Why mechanical_correct is a cleaner headline

`mechanical_correct = validator.derivation_valid AND validator.terminal_matches_gt`.

It's:

- **Deterministic** — no judge, no LLM noise.
- **Style-agnostic** — silent fixers and loud fixers get scored identically if
  their final chain is sound.
- **Luck-proof** — catches compensating errors (right final emoji via a bad
  step along the way) that outcome-only scoring would credit.

It answers the plain-English question: *did the model produce a correct
derivation ending at the right answer?* That is, arguably, the question the
benchmark should actually be asking.

## Known limitation of mechanical_correct

It **penalizes mid-stream self-correction**. If the model makes a *new*
mistake mid-continuation, catches it, retracts, re-emits the step correctly,
and arrives at the right answer, the validator flags the chain as invalid:

- The wrong step fails per-step validity (`after ≠ evaluate(before)`).
- The re-emitted step fails continuity (its `before` doesn't match the
  previous step's `after`).

This is exactly the most interesting behavior the README tagline gestures at —
"catch and fix its own mistakes" — and the current validator can't see it.

**Possible fixes (not yet implemented):**

1. **Retraction-aware parsing.** Recognize "wait, step N should be…" plus a
   re-emitted `Step N:` line; take the last version of each step number as
   canonical; validate that reduced chain.
2. **Allow discontinuity if terminal matches GT.** Weakens to outcome-only
   for self-correcting chains; loses compensating-error rejection.
3. **Fourth metric: "terminal=GT AND every step number's *last* reduction is
   valid".** Middle ground — permits a rewrite, still catches lucky guesses.

Empirical check before investing in a parser rewrite: grep the predictions for
retraction phrases ("wait", "actually", "correction", "let me redo") and see
how often the pattern actually occurs.

## Recommendation

- **Headline:** `mechanical_correct_rate`.
- **Diagnostics (kept, not promoted):** `detect_rate`, `detect_correct_rate`,
  `detect_correct_finaloutput_correct_rate`. Useful for telling *how* a model
  fails — explicit but wrong vs. silent but right vs. silent and wrong.
- **README tagline:** clarify that "catch" here means "land on a mechanically
  correct derivation", *not* "verbally acknowledge the mistake". Or add a
  second headline for the loud variant if that's what's actually being asked.
- **Follow-up:** quantify how often mid-stream retract-and-restate occurs.
  If it's a meaningful fraction (>5%), implement retraction-aware parsing.
