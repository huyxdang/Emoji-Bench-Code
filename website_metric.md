# Website Quality Metric

This rubric scores the Emoji-Bench research page on a 10-point scale. The main goal is simple: the page should be easy to look at, easy to understand, and credible as a research artifact.

## Score Weights

| Category | Weight | What It Measures |
|---|---:|---|
| Immediate comprehension | 45% | Can someone understand the page in the first 5-10 seconds? |
| User engagement | 25% | Does the page invite scrolling, exploration, and retention? |
| Visual clarity | 20% | Does the design feel polished, balanced, and uncluttered? |
| Research credibility | 10% | Does it clearly connect claims to artifacts and avoid overclaiming? |

## 1. Immediate Comprehension - 45%

Score from 0 to 10.

- 9-10: A new reader can state the benchmark question, the condition tested, and the main result metric within 5-10 seconds.
- 7-8: The point is clear, but the reader has to slow down or parse a dense sentence.
- 5-6: The page looks professional but the benchmark premise takes effort to understand.
- 0-4: A reader cannot quickly tell what was tested or what the result means.

Checklist:

- The first viewport states the research question in plain English.
- The first viewport says "no hint" or equivalent.
- The first viewport says the score is final-answer correctness.
- Body text is short, specific, and jargon-light.
- Captions explain what the chart is and where it came from.
- No paragraph is doing more than one job.

## 2. User Engagement - 25%

Score from 0 to 10.

- 9-10: The page has a strong first impression, a clear path through the story, and enough visual variety to keep scrolling without feeling decorative.
- 7-8: Good first impression and flow, but one or two sections feel static or repetitive.
- 5-6: Functional but flat. The reader can get the information, but nothing pulls them forward.
- 0-4: Feels unfinished, generic, or visually monotonous.

Checklist:

- Hero image reinforces the benchmark concept.
- Primary action points to the results.
- Sections create a natural story: question -> method -> result -> artifact.
- Result image is prominent enough to feel like the page payoff.
- Mobile view keeps the same narrative order.

## 3. Visual Clarity - 20%

Score from 0 to 10.

- 9-10: Layout is balanced, calm, and professional. Spacing, alignment, colors, and image scale are deliberate. Nothing overlaps or feels cramped.
- 7-8: Generally polished, with minor spacing or proportion issues.
- 5-6: Usable but uneven. Some elements feel too large, too small, or visually unrelated.
- 0-4: Cluttered, inconsistent, or visually distracting.

Checklist:

- The design uses restraint: mostly white space, a few accent colors, no noisy decoration.
- Headline scale is appropriate for a research page.
- The hero image and result chart are large enough to inspect.
- Cards and bordered regions are used sparingly.
- Text does not collide, wrap awkwardly, or overflow on mobile.

## 4. Research Credibility - 10%

Score from 0 to 10.

- 9-10: The page is precise about the benchmark setting, metric, artifacts, and limitations. It distinguishes current results from broader supported code paths.
- 7-8: Mostly precise, with minor ambiguity around provenance or scope.
- 5-6: Claims are plausible but underspecified. A reader has to inspect the repo to know what was actually measured.
- 0-4: Overclaims, hides limitations, or mixes old and current metrics.

Checklist:

- The page says the current result is B-L0 final-answer correctness.
- It notes Gemini artifacts came through OpenRouter.
- It does not present judge recovery or mechanical correctness as the headline.
- It points to `artifacts/evals/*-B-L0/score_summary.json`.
- It points to `artifacts/plots/b_final_answer_l0.png`.

## Overall Score Formula

```text
overall =
  immediate_comprehension * 0.45 +
  user_engagement * 0.25 +
  visual_clarity * 0.20 +
  research_credibility * 0.10
```

Round to one decimal place.

## Current Review

| Category | Score | Notes |
|---|---:|---|
| Immediate comprehension | 9.7 | The first viewport states the question, hidden wrong step, no-hint prompt, scoring rule, and best result. |
| User engagement | 9.5 | The hero creates a clear reason to continue, and the result section is positioned as the page payoff. |
| Visual clarity | 9.4 | The layout is calm, balanced, and uncluttered, with a strong hero image and clear fact strip. |
| Research credibility | 9.6 | Artifact paths, OpenRouter note, metric scope, and current B-L0 limitation are explicit. |

Overall: **9.6 / 10**

## Improvement Targets

- Keep immediate comprehension above 9.5 by making every first-viewport edit answer: "What is this?", "What was the model asked?", "How is it scored?", and "What happened?"
- Keep engagement above 9.5 by preserving the current question -> fact strip -> results path.
- Push visual clarity above 9 by checking real browser screenshots at desktop and mobile widths when Playwright or another browser renderer is available.
