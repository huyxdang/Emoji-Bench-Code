from __future__ import annotations

import random

from emoji_bench.domain.types import Symbol

# Curated pool of ~30 emoji with no mathematical/logical associations.
# Invariants (enforced by tests/test_symbols.py): each entry is a single
# Unicode codepoint with no variation selectors or ZWJ sequences, so
# string-equality scoring and expression parsing never depend on
# normalization. Token counts vary by model family and are deliberately
# not part of the contract.
EMOJI_POOL: tuple[str, ...] = (
    # Animals
    "🦩", "🐙", "🦔", "🪼", "🦎", "🐌",
    # Objects
    "🧲", "🪣", "🪆", "🧿", "🪤", "🪩",
    # Nature
    "🍄", "🫧", "🪸", "🪻", "🌵", "🪨",
    # Food
    "🧁", "🫐", "🥟", "🪺", "🧄", "🫑",
    # Misc
    "🪬", "🧊", "🪈", "🪭", "🧶", "🪵",
)


def sample_symbols(n: int, rng: random.Random) -> tuple[Symbol, ...]:
    """Sample n distinct emoji from the curated pool, return as Symbols."""
    if n < 1 or n > len(EMOJI_POOL):
        raise ValueError(f"n must be between 1 and {len(EMOJI_POOL)}, got {n}")
    pool = list(EMOJI_POOL)
    rng.shuffle(pool)
    return tuple(Symbol(e) for e in pool[:n])
