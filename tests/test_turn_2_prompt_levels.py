import pytest

from emoji_bench.continuation_formatter import (
    TURN_2_PROMPT_LEVELS,
    TURN_2_USER,
    get_turn_2_prompt,
)


def test_level_0_matches_original_turn_2_user():
    assert TURN_2_PROMPT_LEVELS[0] == TURN_2_USER
    assert get_turn_2_prompt(0) == "Please continue."


def test_levels_are_monotonically_longer():
    lengths = [len(TURN_2_PROMPT_LEVELS[i]) for i in sorted(TURN_2_PROMPT_LEVELS)]
    assert lengths == sorted(lengths), (
        f"prompting-strength levels should grow in length, got {lengths}"
    )


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="unknown turn_2 prompt level"):
        get_turn_2_prompt(42)


def test_all_expected_levels_present():
    assert set(TURN_2_PROMPT_LEVELS) == {0, 1, 2, 3}


def test_level_1_is_a_soft_hint():
    level_1 = get_turn_2_prompt(1)
    assert "Please continue" in level_1
    assert "unsure" in level_1


def test_level_2_mentions_verifying_against_the_rules():
    level_2 = get_turn_2_prompt(2)
    assert "Verify" in level_2 or "verify" in level_2
    assert "rules" in level_2


def test_level_3_is_the_explicit_error_check_variant():
    level_3 = get_turn_2_prompt(3)
    assert "errors" in level_3 or "error" in level_3
    assert "correct" in level_3.lower()
