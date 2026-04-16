from __future__ import annotations

from dataclasses import dataclass

from emoji_bench.domain.chain_types import DerivationChain
from emoji_bench.domain.expressions import Expression
from emoji_bench.domain.types import Symbol


@dataclass(frozen=True)
class ErrorInfo:
    step_number: int
    correct_result: Symbol | None
    injected_result: Symbol | None
    correct_after: Expression
    injected_after: Expression
    original_chain: DerivationChain
    correct_rule_used: str | None = None
    injected_rule_used: str | None = None
