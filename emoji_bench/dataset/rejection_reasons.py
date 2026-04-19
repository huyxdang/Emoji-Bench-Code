from __future__ import annotations


R_CHAIN_TOO_SHORT = "chain_too_short"
R_INSUFFICIENT_RUNWAY = "insufficient_runway"
R_NO_ELIGIBLE_IN_CHAIN = "no_eligible_in_chain"
R_NO_ELIGIBLE_IN_WINDOW = "no_eligible_in_window"
R_CASCADE_CONVERGENT = "cascade_convergent"
R_OTHER_INJECTOR_ERROR = "other_injector_error"

REJECTION_REASONS: tuple[str, ...] = (
    R_CHAIN_TOO_SHORT,
    R_INSUFFICIENT_RUNWAY,
    R_NO_ELIGIBLE_IN_CHAIN,
    R_NO_ELIGIBLE_IN_WINDOW,
    R_CASCADE_CONVERGENT,
    R_OTHER_INJECTOR_ERROR,
)


class ContinuationGenerationError(ValueError):
    """Structured generation failure with a stable rejection reason code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
