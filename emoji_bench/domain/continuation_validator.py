"""Phase 5b: deterministic Python validator for E-CONTINUE metric (3).

Parses a model's continuation into ``Step N: <before> = <after>    [by <rule>]``
tuples and validates the derivation against the formal system. Reuses the
existing interpreter so "valid" means "evaluates to the same symbol under the
system's operation table." This catches compensating-error continuations: a
step that writes a wrong ``after`` is caught here even if the final output
happens to match ground truth.

The validator does NOT consult the prefill. It checks only that the model's
own steps are (a) individually correct reductions, (b) consecutively continuous,
and (c) terminate at a single symbol equal to ``ground_truth_final_output``.
This keeps the validator independent of whether the model cascades from the bad
state or restarts with a correction — as long as its math is right, metric (3)
is True.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from emoji_bench.domain.expressions import (
    BinaryOp,
    Expression,
    SymbolLiteral,
    UnaryTransform,
)
from emoji_bench.domain.interpreter import evaluate
from emoji_bench.domain.types import FormalSystem


_STEP_LINE_REGEX = re.compile(
    r"^\s*Step\s+(\d+)\s*:\s*(.+?)\s*=\s*(.+?)(?:\s*\[by\s+[^\]]+\])?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedStep:
    step_number: int
    before: Expression
    after: Expression
    before_str: str
    after_str: str


@dataclass(frozen=True)
class ValidationResult:
    parseable: bool
    derivation_valid: bool
    terminal_matches_gt: bool
    first_invalid_step: int | None
    first_discontinuity_step: int | None
    parsed_step_count: int
    reason: str | None


# --- Parsing ---------------------------------------------------------------


class ParseError(ValueError):
    pass


def _build_token_lookups(
    system: FormalSystem,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Return (symbol_map, op_map, transform_names) indexed by rendered string.

    ``symbol_map`` maps emoji strings to themselves (identity — we return
    emoji strings out of the parser because downstream uses ``SymbolLiteral``
    whose ``symbol.emoji`` is what we need). ``op_map`` maps the operation's
    rendered symbol (e.g. ``⊕``) to its internal ``op.name``. Likewise for
    transforms (which are plain ASCII names).
    """
    # Lookup from rendered symbol to the Symbol object.
    symbol_map: dict[str, str] = {sym.emoji: sym.emoji for sym in system.symbols}
    op_map: dict[str, str] = {}
    for op in system.base_operations:
        op_map[op.symbol_id] = op.name
    for op in system.derived_operations:
        op_map[op.symbol_id] = op.name
    transform_names = {t.name for t in system.transformations}
    return symbol_map, op_map, transform_names


def _parse_expression(text: str, system: FormalSystem) -> Expression:
    """Recursive-descent parser matching ``expr_to_str_with_system``'s output.

    Grammar:
        expr := '(' expr op_sym expr ')'
              | transform_name '(' expr ')'
              | symbol
    """
    text = text.strip()
    if not text:
        raise ParseError("empty expression")

    symbol_map, op_map, transform_names = _build_token_lookups(system)
    symbol_strs = sorted(symbol_map.keys(), key=len, reverse=True)
    op_strs = sorted(op_map.keys(), key=len, reverse=True)
    transform_strs = sorted(transform_names, key=len, reverse=True)

    pos = 0
    n = len(text)

    def skip_ws():
        nonlocal pos
        while pos < n and text[pos].isspace():
            pos += 1

    def peek(s: str) -> bool:
        return text.startswith(s, pos)

    def try_match(candidates: list[str]) -> str | None:
        for c in candidates:
            if peek(c):
                return c
        return None

    def expect(ch: str) -> None:
        nonlocal pos
        skip_ws()
        if pos >= n or text[pos] != ch:
            raise ParseError(
                f"expected {ch!r} at position {pos} in {text!r}; "
                f"got {text[pos:pos+10]!r}"
            )
        pos += 1

    def parse_expr() -> Expression:
        nonlocal pos
        skip_ws()
        if pos >= n:
            raise ParseError(f"unexpected end of input in {text!r}")

        if text[pos] == "(":
            pos += 1
            left = parse_expr()
            skip_ws()
            op_str = try_match(op_strs)
            if op_str is None:
                raise ParseError(
                    f"expected operator at position {pos} in {text!r}; "
                    f"got {text[pos:pos+10]!r}"
                )
            pos += len(op_str)
            right = parse_expr()
            expect(")")
            return BinaryOp(op_name=op_map[op_str], left=left, right=right)

        tr_str = try_match(transform_strs)
        if tr_str is not None and pos + len(tr_str) < n and text[pos + len(tr_str)] == "(":
            pos += len(tr_str)
            expect("(")
            operand = parse_expr()
            expect(")")
            return UnaryTransform(transform_name=tr_str, operand=operand)

        sym_str = try_match(symbol_strs)
        if sym_str is not None:
            pos += len(sym_str)
            # Need a Symbol object; find the matching one in system.symbols.
            for sym in system.symbols:
                if sym.emoji == sym_str:
                    return SymbolLiteral(symbol=sym)
            raise ParseError(f"symbol {sym_str!r} not in system")  # unreachable

        raise ParseError(
            f"unknown token at position {pos} in {text!r}: {text[pos:pos+10]!r}"
        )

    result = parse_expr()
    skip_ws()
    if pos != n:
        raise ParseError(
            f"trailing content at position {pos} in {text!r}: {text[pos:]!r}"
        )
    return result


def parse_continuation_steps(
    text: str,
    system: FormalSystem,
) -> list[ParsedStep] | None:
    """Parse a raw continuation into ordered ParsedStep entries.

    Returns ``None`` if no Step-N lines match or any matched line fails to
    parse into valid expressions. Non-step lines (prose, Final Output:) are
    silently ignored as long as at least some step lines parse cleanly.
    """
    steps: list[ParsedStep] = []
    seen_any_step_line = False
    for line in text.splitlines():
        match = _STEP_LINE_REGEX.match(line)
        if not match:
            continue
        seen_any_step_line = True
        step_number = int(match.group(1))
        before_str = match.group(2).strip()
        after_str = match.group(3).strip()
        try:
            before_expr = _parse_expression(before_str, system)
            after_expr = _parse_expression(after_str, system)
        except ParseError:
            # One malformed step line invalidates the whole parse — we can't
            # confidently say the derivation is correct if we can't read it.
            return None
        steps.append(
            ParsedStep(
                step_number=step_number,
                before=before_expr,
                after=after_expr,
                before_str=before_str,
                after_str=after_str,
            )
        )

    if not seen_any_step_line:
        return None
    return steps


# --- Validation ------------------------------------------------------------


def _expressions_equal(a: Expression, b: Expression) -> bool:
    """Structural equality on Expression ASTs (dataclasses compare by value)."""
    return a == b


def validate_derivation(
    raw_continuation_text: str,
    system: FormalSystem,
    ground_truth_final_output: str,
) -> ValidationResult:
    """Validate a model's continuation against the formal system.

    Rules:
    - Every parsed step's ``after`` must equal ``evaluate(before)`` under
      the system's operation table (catches wrong reductions, including
      compensating-error pairs that happen to land on ground truth).
    - Consecutive steps must be continuous: step[i+1].before == step[i].after.
    - The final step's ``after`` must be a ``SymbolLiteral`` whose emoji
      matches ``ground_truth_final_output``.
    """
    steps = parse_continuation_steps(raw_continuation_text, system)
    if steps is None:
        return ValidationResult(
            parseable=False,
            derivation_valid=False,
            terminal_matches_gt=False,
            first_invalid_step=None,
            first_discontinuity_step=None,
            parsed_step_count=0,
            reason="unparseable: no Step-N lines or a step failed to parse",
        )

    if not steps:
        return ValidationResult(
            parseable=True,
            derivation_valid=False,
            terminal_matches_gt=False,
            first_invalid_step=None,
            first_discontinuity_step=None,
            parsed_step_count=0,
            reason="no steps produced",
        )

    # Per-step validity.
    for step in steps:
        try:
            before_val = evaluate(step.before, system)
            after_val = evaluate(step.after, system)
        except (ValueError, KeyError):
            return ValidationResult(
                parseable=True,
                derivation_valid=False,
                terminal_matches_gt=False,
                first_invalid_step=step.step_number,
                first_discontinuity_step=None,
                parsed_step_count=len(steps),
                reason=f"step {step.step_number}: evaluate raised",
            )
        if before_val != after_val:
            return ValidationResult(
                parseable=True,
                derivation_valid=False,
                terminal_matches_gt=False,
                first_invalid_step=step.step_number,
                first_discontinuity_step=None,
                parsed_step_count=len(steps),
                reason=(
                    f"step {step.step_number}: before evaluates to "
                    f"{before_val.emoji!r} but after evaluates to {after_val.emoji!r}"
                ),
            )

    # Consecutive-step continuity.
    for prev, curr in zip(steps, steps[1:]):
        if not _expressions_equal(prev.after, curr.before):
            return ValidationResult(
                parseable=True,
                derivation_valid=False,
                terminal_matches_gt=False,
                first_invalid_step=None,
                first_discontinuity_step=curr.step_number,
                parsed_step_count=len(steps),
                reason=(
                    f"discontinuity at step {curr.step_number}: prev after "
                    f"({prev.after_str!r}) != curr before ({curr.before_str!r})"
                ),
            )

    # Terminal check.
    terminal = steps[-1].after
    if not isinstance(terminal, SymbolLiteral):
        return ValidationResult(
            parseable=True,
            derivation_valid=True,
            terminal_matches_gt=False,
            first_invalid_step=None,
            first_discontinuity_step=None,
            parsed_step_count=len(steps),
            reason="terminal is not a single symbol",
        )
    terminal_matches = terminal.symbol.emoji == ground_truth_final_output
    return ValidationResult(
        parseable=True,
        derivation_valid=True,
        terminal_matches_gt=terminal_matches,
        first_invalid_step=None,
        first_discontinuity_step=None,
        parsed_step_count=len(steps),
        reason=None,
    )
