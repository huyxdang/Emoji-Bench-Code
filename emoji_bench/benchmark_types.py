from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from emoji_bench.chain_types import DerivationChain
from emoji_bench.expressions import Expression
from emoji_bench.types import Symbol


class ErrorType(str, Enum):
    E_CASC = "E-CASC"
    E_CONTINUE = "E-CONTINUE"


@dataclass(frozen=True)
class ErrorInfo:
    error_type: ErrorType
    step_number: int
    correct_result: Symbol | None
    injected_result: Symbol | None
    correct_after: Expression
    injected_after: Expression
    original_chain: DerivationChain
    correct_rule_used: str | None = None
    injected_rule_used: str | None = None
