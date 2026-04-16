from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from emoji_bench.providers.clients import ProviderUsage


ContinuationMode = Literal["prefill", "single_turn"]


@dataclass(frozen=True)
class ContinuationResponse:
    raw_continuation_text: str
    response_id: str | None
    usage: ProviderUsage | None
    mode: ContinuationMode
