import pytest

from emoji_bench.model_registry import (
    CLAUDE_OPUS_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONFIGS,
    apply_reasoning_effort_override,
    get_model_config,
    model_choices,
)
from emoji_bench.providers.clients import resolve_api_key


def test_requested_model_configs_are_present():
    assert {
        "claude-opus-4-7-reasoning-max",
        "claude-opus-4-6-reasoning-high",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-6-reasoning",
        "claude-sonnet-4-6-reasoning-high",
        "gemini-3-flash-preview-thinking-high",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview-thinking-high",
        "gemini-3.1-pro-preview",
        "gpt-5.4-reasoning-xhigh",
        "gpt-5.4",
        "gpt-5.4-mini-no-reasoning",
        "gpt-5.4-mini-reasoning-xhigh",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "magistral-medium-2509",
        "mistral-large-2512",
        "mistral-medium-2508",
    }.issubset(set(model_choices()))


def test_gpt54_models_default_to_medium_reasoning():
    for key in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"):
        config = get_model_config(key)
        assert config.provider == "openai"
        assert config.openai_reasoning is not None
        assert config.openai_reasoning.effort == "medium"
    assert get_model_config("gpt-5.4").default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert get_model_config("gpt-5.4-mini").default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert get_model_config("gpt-5.4-nano").default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_pinned_openai_reasoning_xhigh_aliases_are_present():
    for key in ("gpt-5.4-reasoning-xhigh", "gpt-5.4-mini-reasoning-xhigh"):
        config = get_model_config(key)
        assert config.provider == "openai"
        assert config.openai_reasoning is not None
        assert config.openai_reasoning.effort == "xhigh"


def test_gpt54_mini_no_reasoning_alias_is_present():
    config = get_model_config("gpt-5.4-mini-no-reasoning")
    assert config.provider == "openai"
    assert config.api_model == "gpt-5.4-mini"
    assert config.openai_reasoning is not None
    assert config.openai_reasoning.effort == "none"


def test_claude_sonnet_46_models_default_to_high_anthropic_effort():
    baseline = get_model_config("claude-sonnet-4-6")
    assert baseline.provider == "anthropic"
    assert baseline.anthropic_effort == "high"
    assert baseline.anthropic_thinking is not None
    assert baseline.anthropic_thinking.enabled is False

    reasoning = get_model_config("claude-sonnet-4-6-reasoning")
    assert reasoning.provider == "anthropic"
    assert reasoning.anthropic_effort == "high"
    assert reasoning.anthropic_thinking is not None
    assert reasoning.anthropic_thinking.enabled is True
    assert reasoning.anthropic_thinking.budget_tokens == 1024


def test_pinned_claude_reasoning_high_aliases_are_present():
    for key, api_model in (
        ("claude-opus-4-6-reasoning-high", "claude-opus-4-6"),
        ("claude-sonnet-4-6-reasoning-high", "claude-sonnet-4-6"),
    ):
        config = get_model_config(key)
        assert config.provider == "anthropic"
        assert config.api_model == api_model
        assert config.anthropic_effort == "high"
        assert config.anthropic_thinking is not None
        assert config.anthropic_thinking.enabled is True
        assert config.anthropic_thinking.budget_tokens == 1024


def test_pinned_claude_opus_47_reasoning_max_alias_is_present():
    config = get_model_config("claude-opus-4-7-reasoning-max")
    assert config.provider == "anthropic"
    assert config.api_model == "claude-opus-4-7"
    assert config.anthropic_effort == "max"
    assert config.default_max_output_tokens == CLAUDE_OPUS_MAX_OUTPUT_TOKENS
    assert config.anthropic_thinking is not None
    assert config.anthropic_thinking.enabled is True
    assert config.anthropic_thinking.mode == "adaptive"
    assert config.anthropic_thinking.budget_tokens is None


def test_pinned_gemini_thinking_high_aliases_are_present():
    for key, api_model in (
        ("gemini-3.1-pro-preview-thinking-high", "gemini-3.1-pro-preview"),
        ("gemini-3-flash-preview-thinking-high", "gemini-3-flash-preview"),
    ):
        config = get_model_config(key)
        assert config.provider == "gemini"
        assert config.api_model == api_model
        assert config.gemini_thinking is not None
        assert config.gemini_thinking.level == "high"


def test_model_choices_put_stronger_claude_and_gemini_variants_first():
    choices = model_choices()
    assert choices.index("claude-opus-4-7-reasoning-max") < choices.index(
        "claude-opus-4-6-reasoning-high"
    )
    assert choices.index("claude-opus-4-6-reasoning-high") < choices.index(
        "claude-sonnet-4-6-reasoning-high"
    )
    assert choices.index("gemini-3.1-pro-preview-thinking-high") < choices.index(
        "gemini-3-flash-preview-thinking-high"
    )
    assert choices.index("mistral-large-2512") < choices.index("magistral-medium-2509")


def test_all_configured_models_use_expected_default_max_output_tokens():
    assert DEFAULT_MAX_OUTPUT_TOKENS == 4096
    for config in MODEL_CONFIGS.values():
        if config.key == "claude-opus-4-7-reasoning-max":
            assert config.default_max_output_tokens == CLAUDE_OPUS_MAX_OUTPUT_TOKENS
        else:
            assert config.default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_resolve_api_key_uses_provider_specific_env_var():
    config = get_model_config("claude-sonnet-4-6")
    api_key = resolve_api_key(
        model_config=config,
        explicit_api_key=None,
        env={"ANTHROPIC_API_KEY": "test-key"},
    )
    assert api_key == "test-key"


def test_resolve_api_key_supports_gemini_env_var():
    config = get_model_config("gemini-3-flash-preview")
    api_key = resolve_api_key(
        model_config=config,
        explicit_api_key=None,
        env={"GEMINI_API_KEY": "test-gemini-key"},
    )
    assert api_key == "test-gemini-key"


def test_apply_reasoning_effort_override_updates_openai_and_anthropic_configs():
    gpt = apply_reasoning_effort_override(get_model_config("gpt-5.4"), "high")
    assert gpt.openai_reasoning is not None
    assert gpt.openai_reasoning.effort == "high"

    sonnet = apply_reasoning_effort_override(get_model_config("claude-sonnet-4-6"), "max")
    assert sonnet.anthropic_effort == "max"
    assert sonnet.anthropic_thinking is not None
    assert sonnet.anthropic_thinking.enabled is False

    sonnet_reasoning = apply_reasoning_effort_override(
        get_model_config("claude-sonnet-4-6-reasoning"),
        "low",
    )
    assert sonnet_reasoning.anthropic_effort == "low"
    assert sonnet_reasoning.anthropic_thinking is not None
    assert sonnet_reasoning.anthropic_thinking.enabled is True
    assert sonnet_reasoning.anthropic_thinking.budget_tokens == 1024

    opus_47 = apply_reasoning_effort_override(get_model_config("claude-opus-4-7-reasoning-max"), "xhigh")
    assert opus_47.anthropic_effort == "xhigh"
    assert opus_47.anthropic_thinking is not None
    assert opus_47.anthropic_thinking.enabled is True
    assert opus_47.anthropic_thinking.mode == "adaptive"


def test_apply_reasoning_effort_override_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="Anthropic effort does not support 'minimal'"):
        apply_reasoning_effort_override(get_model_config("claude-sonnet-4-6"), "minimal")

    with pytest.raises(ValueError, match="xhigh is available only on Claude Opus 4.7"):
        apply_reasoning_effort_override(get_model_config("claude-sonnet-4-6"), "xhigh")

    with pytest.raises(ValueError, match="does not support it"):
        apply_reasoning_effort_override(get_model_config("claude-haiku-4-5"), "low")

    with pytest.raises(ValueError, match="OpenAI reasoning does not support effort='max'"):
        apply_reasoning_effort_override(get_model_config("gpt-5.4"), "max")
