from __future__ import annotations

import random

from emoji_bench.types import Symbol

# Curated pool of ~30 emoji with no mathematical/logical associations.
# Each must tokenize as a single token across major model families.
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
