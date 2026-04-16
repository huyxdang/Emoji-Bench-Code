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
