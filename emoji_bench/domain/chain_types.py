from __future__ import annotations

from dataclasses import dataclass

from emoji_bench.expressions import Expression, expr_to_str
from emoji_bench.types import Symbol


@dataclass(frozen=True)
class ChainStep:
    step_number: int
    before: Expression              # full expression before this reduction
    reduced_subexpr: Expression     # the subexpression that was reduced
    result_symbol: Symbol | None    # what it reduced to (None for derived-op expansion steps)
    after: Expression               # full expression after substituting the result
    rule_used: str                  # display name: "⊕ table", "definition of ⊗", "inv"
    rule_type: str                  # "base_op" | "derived_op" | "transform"
    expanded_to: Expression | None = None  # for derived-op steps: the expanded expression

    def __repr__(self) -> str:
        before = expr_to_str(self.before)
        after = expr_to_str(self.after)
        return f"Step {self.step_number}: {before} = {after} [by {self.rule_used}]"


@dataclass(frozen=True)
class DerivationChain:
    starting_expression: Expression
    steps: tuple[ChainStep, ...]
    final_result: Symbol
    seed: int | None

    def __repr__(self) -> str:
        start = expr_to_str(self.starting_expression)
        lines = [
            f"DerivationChain(start={start}, {len(self.steps)} steps, "
            f"result={self.final_result}, seed={self.seed})"
        ]
        for step in self.steps:
            lines.append(f"  {step!r}")
        return "\n".join(lines)
