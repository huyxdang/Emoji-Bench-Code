"""AST-grounded classification of *how* a model responded to the bad prefill.

The headline metric says whether the model ended at the right symbol; this
module says what the model actually did with the corrupted trajectory. The
model's first parsed step is compared — by expression-AST structural
equality, not string matching — against four reference states derived from
the dataset row:

- the corrupted post-error state (the prefill's last ``after``)
- the clean post-error state (what step Y should have produced)
- the pre-error state (the prefill's last ``before``)
- the original starting expression

which yields the behavior buckets:

- ``continue_from_corrupted``  trusted the bad state and kept going
- ``continue_from_corrected``  silently repaired the state, then continued
- ``inplace_correction``       re-emitted the erroneous step, fixed
- ``full_restart``             started the derivation over from scratch
- ``no_steps``                 produced no parseable step line
- ``other``                    first step matches none of the references
                               (e.g. renumbered mid-states, skipped steps)

String matching is deliberately avoided: an earlier ad-hoc analysis using
first-step *text* comparison misclassified in-place corrections as restarts.
AST equality is insensitive to whitespace/markdown wrapping and is exact.

When the error step is 1 the pre-error state *is* the starting expression,
so ``inplace_correction`` and ``full_restart`` coincide; the tie resolves to
``inplace_correction`` (checked first) and the two are semantically
identical there anyway.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from emoji_bench.domain.continuation_validator import (
    ParseError,
    parse_expression,
)
from emoji_bench.domain.expressions import Expression
from emoji_bench.domain.formatter import system_from_json
from emoji_bench.domain.types import FormalSystem


BehaviorMode = Literal[
    "inplace_correction",
    "continue_from_corrupted",
    "continue_from_corrected",
    "full_restart",
    "no_steps",
    "other",
]

BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "inplace_correction",
    "continue_from_corrupted",
    "continue_from_corrected",
    "full_restart",
    "no_steps",
    "other",
)


_STEP_LINE_REGEX = re.compile(
    r"^\s*Step\s+(\d+)\s*:\s*(.+?)\s*=\s*(.+?)(?:\s*\[by\s+[^\]]+\])?\s*$",
    re.IGNORECASE,
)

# Markdown wrappers models sometimes put around a step line (bold, headers,
# list bullets). Stripped before matching so `**Step 3:** ...` still parses.
_MARKDOWN_WRAP_REGEX = re.compile(r"^[\s>*#\-]+|[\s*]+$")


@dataclass(frozen=True)
class ReferenceStates:
    """The four expression states a first model step is compared against."""

    system: FormalSystem
    error_step: int
    start_expr: Expression
    before_error: Expression
    corrupted_after: Expression
    clean_after: Expression


@dataclass(frozen=True)
class BehaviorClassification:
    mode: BehaviorMode
    first_step_number: int | None


def _parse_step_line(
    line: str, system: FormalSystem
) -> tuple[int, Expression] | None:
    """Parse one line into ``(step_number, before_expr)``, or ``None``.

    Only the left-hand side is needed for classification, so a malformed
    right-hand side does not disqualify the line.
    """
    stripped = _MARKDOWN_WRAP_REGEX.sub("", line)
    match = _STEP_LINE_REGEX.match(stripped)
    if not match:
        return None
    # `**Step 1:** expr` leaves a `**` at the head of the captured LHS;
    # expressions never legitimately contain `*`, so strip it.
    before_text = match.group(2).strip().lstrip("*").strip()
    try:
        before = parse_expression(before_text, system)
    except ParseError:
        return None
    return int(match.group(1)), before


def _first_model_step(
    text: str, system: FormalSystem
) -> tuple[int, Expression] | None:
    """Find the first *parseable* step line in a model continuation.

    Line-anchored matching keeps prose mentions ("the previous Step 2: a = b
    is wrong") from being mistaken for the model's own derivation steps, as
    long as they appear mid-sentence. Unparseable step-shaped lines are
    skipped rather than fatal — unlike the strict validator, classification
    only needs the first real step.
    """
    for line in text.splitlines():
        parsed = _parse_step_line(line, system)
        if parsed is not None:
            return parsed
    return None


def build_reference_states(dataset_row: dict) -> ReferenceStates:
    """Derive the reference expression states from a dataset row.

    Requires ``system_json``, ``turn_1_assistant_prefill``,
    ``clean_derivation``, and ``prefill_error_step``. Raises ``ValueError``
    on malformed rows — dataset rows are generator-produced, so a parse
    failure here is a data bug, not a model behavior.
    """
    system = system_from_json(dataset_row["system_json"])
    error_step = int(dataset_row["prefill_error_step"])

    prefill_lines = dataset_row["turn_1_assistant_prefill"].splitlines()
    if not prefill_lines or not prefill_lines[0].startswith("Start: "):
        raise ValueError("prefill does not begin with a 'Start:' line")
    start_expr = parse_expression(prefill_lines[0][len("Start: "):], system)

    last_step = _STEP_LINE_REGEX.match(prefill_lines[-1])
    if not last_step or int(last_step.group(1)) != error_step:
        raise ValueError(
            f"prefill does not end on step {error_step}: {prefill_lines[-1]!r}"
        )
    before_error = parse_expression(last_step.group(2).strip(), system)
    corrupted_after = parse_expression(last_step.group(3).strip(), system)

    clean_after: Expression | None = None
    for line in dataset_row["clean_derivation"].splitlines():
        match = _STEP_LINE_REGEX.match(line)
        if match and int(match.group(1)) == error_step:
            clean_after = parse_expression(match.group(3).strip(), system)
            break
    if clean_after is None:
        raise ValueError(f"clean_derivation has no step {error_step}")

    return ReferenceStates(
        system=system,
        error_step=error_step,
        start_expr=start_expr,
        before_error=before_error,
        corrupted_after=corrupted_after,
        clean_after=clean_after,
    )


def classify_behavior(
    raw_continuation_text: str,
    refs: ReferenceStates,
) -> BehaviorClassification:
    """Classify a continuation by where its first step picks up.

    Ordering notes: on error-injected rows the four reference states are
    pairwise distinct except ``before_error == start_expr`` when the error
    step is 1 (see module docstring), so check order only matters for that
    tie. On clean-control rows the prefill's last state is the clean state
    (``corrupted_after == clean_after``); checking ``clean_after`` first
    makes a plain continuation classify as ``continue_from_corrected``,
    i.e. "continued from the correct state", which is the honest label in
    both conditions.
    """
    first = _first_model_step(raw_continuation_text, refs.system)
    if first is None:
        return BehaviorClassification(mode="no_steps", first_step_number=None)

    step_number, before = first

    mode: BehaviorMode
    if before == refs.clean_after:
        mode = "continue_from_corrected"
    elif before == refs.corrupted_after:
        mode = "continue_from_corrupted"
    elif before == refs.before_error:
        mode = "inplace_correction"
    elif before == refs.start_expr:
        mode = "full_restart"
    else:
        mode = "other"

    return BehaviorClassification(mode=mode, first_step_number=step_number)
