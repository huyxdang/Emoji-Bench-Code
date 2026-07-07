import random

from emoji_bench.domain.symbols import EMOJI_POOL, sample_symbols


def test_sample_symbols_count():
    rng = random.Random(42)
    for n in (3, 4, 5, 6):
        syms = sample_symbols(n, rng)
        assert len(syms) == n


def test_sample_symbols_unique():
    rng = random.Random(42)
    syms = sample_symbols(6, rng)
    emojis = [s.emoji for s in syms]
    assert len(set(emojis)) == 6


def test_sample_symbols_from_pool():
    rng = random.Random(42)
    syms = sample_symbols(6, rng)
    for s in syms:
        assert s.emoji in EMOJI_POOL


def test_sample_symbols_deterministic():
    syms1 = sample_symbols(4, random.Random(123))
    syms2 = sample_symbols(4, random.Random(123))
    assert syms1 == syms2


def test_sample_symbols_different_seeds():
    syms1 = sample_symbols(4, random.Random(1))
    syms2 = sample_symbols(4, random.Random(2))
    assert syms1 != syms2


def test_pool_entries_are_single_codepoints():
    # String-equality scoring and expression parsing rely on each symbol
    # being one codepoint with no combining machinery.
    for emoji in EMOJI_POOL:
        assert len(emoji) == 1, f"{emoji!r} is {len(emoji)} codepoints"


def test_pool_entries_have_no_variation_selectors_or_zwj():
    forbidden = {"︎", "️", "‍"}  # VS15, VS16, ZWJ
    for emoji in EMOJI_POOL:
        assert not (set(emoji) & forbidden), f"{emoji!r} contains VS/ZWJ"


def test_pool_entries_are_unique():
    assert len(set(EMOJI_POOL)) == len(EMOJI_POOL)
