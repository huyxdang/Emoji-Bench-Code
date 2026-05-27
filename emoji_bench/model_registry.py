from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal


ProviderName = Literal["openai", "anthropic", "mistral", "gemini", "openrouter", "xai"]
OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
AnthropicEffort = Literal["low", "medium", "high", "max"]
GeminiThinkingLevel = Literal["minimal", "low", "medium", "high"]
AnthropicThinkingMode = Literal["manual", "adaptive"]
ReasoningEffortOverride = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

DEFAULT_MAX_OUTPUT_TOKENS = 4096
GPT_5_2_MAX_OUTPUT_TOKENS = 128000
GPT_5_5_MAX_OUTPUT_TOKENS = 128000
GPT_5_4_MAX_OUTPUT_TOKENS = 128000
CLAUDE_OPUS_MAX_OUTPUT_TOKENS = 128000
CLAUDE_SONNET_MAX_OUTPUT_TOKENS = 64000
GEMINI_3_MAX_OUTPUT_TOKENS = 64000
MISTRAL_MAX_OUTPUT_TOKENS = 40000
REASONING_EFFORT_CHOICES: tuple[ReasoningEffortOverride, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

OPENAI_DOCS_BASE = "https://developers.openai.com/api/docs/models"
ANTHROPIC_MODELS_DOCS_URL = "https://platform.claude.com/docs/en/about-claude/models/overview"
GEMINI_MODELS_DOCS_URL = "https://ai.google.dev/gemini-api/docs/gemini-3"
GEMINI_THINKING_DOCS_URL = "https://ai.google.dev/gemini-api/docs/thinking"
XAI_REASONING_DOCS_URL = "https://docs.x.ai/developers/model-capabilities/text/reasoning"


@dataclass(frozen=True)
class OpenAIReasoningConfig:
    effort: OpenAIReasoningEffort
    summary: str | None = None


@dataclass(frozen=True)
class AnthropicThinkingConfig:
    enabled: bool
    budget_tokens: int | None = None
    mode: AnthropicThinkingMode = "manual"


@dataclass(frozen=True)
class GeminiThinkingConfig:
    level: GeminiThinkingLevel


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    provider: ProviderName
    api_model: str
    docs_url: str
    api_key_env_var: str
    default_max_output_tokens: int
    openai_reasoning: OpenAIReasoningConfig | None = None
    anthropic_thinking: AnthropicThinkingConfig | None = None
    anthropic_effort: AnthropicEffort | None = None
    gemini_thinking: GeminiThinkingConfig | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _openai_model(
    *,
    key: str,
    label: str,
    api_model: str,
    docs_suffix: str | None = None,
    openai_reasoning: OpenAIReasoningConfig | None = None,
    notes: str | None = None,
) -> ModelConfig:
    docs_url = OPENAI_DOCS_BASE if docs_suffix is None else f"{OPENAI_DOCS_BASE}/{docs_suffix}"
    return ModelConfig(
        key=key,
        label=label,
        provider="openai",
        api_model=api_model,
        docs_url=docs_url,
        api_key_env_var="OPENAI_API_KEY",
        default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        openai_reasoning=openai_reasoning,
        notes=notes,
    )


def _anthropic_model(
    *,
    key: str,
    label: str,
    api_model: str,
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    anthropic_thinking: AnthropicThinkingConfig | None = None,
    anthropic_effort: AnthropicEffort | None = None,
    notes: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        key=key,
        label=label,
        provider="anthropic",
        api_model=api_model,
        docs_url=ANTHROPIC_MODELS_DOCS_URL,
        api_key_env_var="ANTHROPIC_API_KEY",
        default_max_output_tokens=default_max_output_tokens,
        anthropic_thinking=anthropic_thinking,
        anthropic_effort=anthropic_effort,
        notes=notes,
    )


def _gemini_model(
    *,
    key: str,
    label: str,
    api_model: str,
    gemini_thinking: GeminiThinkingConfig | None = None,
    notes: str | None = None,
) -> ModelConfig:
    docs_url = GEMINI_THINKING_DOCS_URL if gemini_thinking is not None else GEMINI_MODELS_DOCS_URL
    return ModelConfig(
        key=key,
        label=label,
        provider="openrouter",
        api_model=f"google/{api_model}",
        docs_url=docs_url,
        api_key_env_var="OPENROUTER_API_KEY",
        default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        gemini_thinking=gemini_thinking,
        notes=notes,
    )


def _mistral_model(
    *,
    key: str,
    label: str,
    api_model: str,
    docs_url: str,
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    notes: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        key=key,
        label=label,
        provider="mistral",
        api_model=api_model,
        docs_url=docs_url,
        api_key_env_var="MISTRAL_API_KEY",
        default_max_output_tokens=default_max_output_tokens,
        notes=notes,
    )


def _xai_model(
    *,
    key: str,
    label: str,
    api_model: str,
    openai_reasoning: OpenAIReasoningConfig | None = None,
    notes: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        key=key,
        label=label,
        provider="xai",
        api_model=api_model,
        docs_url=XAI_REASONING_DOCS_URL,
        api_key_env_var="XAI_API_KEY",
        default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        openai_reasoning=openai_reasoning,
        notes=notes,
    )


GPT_4_1_MINI = _openai_model(
    key="gpt-4.1-mini",
    label="GPT-4.1 mini",
    api_model="gpt-4.1-mini",
    notes="Legacy non-reasoning baseline kept for backward compatibility.",
)
GPT_5_2 = _openai_model(
    key="gpt-5.2",
    label="GPT-5.2",
    api_model="gpt-5.2",
    docs_suffix="gpt-5.2",
    openai_reasoning=OpenAIReasoningConfig(effort="medium"),
    notes=(
        "Previous frontier model for professional work. Defaults to medium "
        "reasoning for non-headline evaluation runs."
    ),
)
GPT_5_5 = _openai_model(
    key="gpt-5.5",
    label="GPT-5.5",
    api_model="gpt-5.5",
    docs_suffix="gpt-5.5",
    openai_reasoning=OpenAIReasoningConfig(effort="medium"),
    notes="Configured to use medium reasoning effort for evaluation runs.",
)
GPT_5_4 = _openai_model(
    key="gpt-5.4",
    label="GPT-5.4",
    api_model="gpt-5.4",
    docs_suffix="gpt-5.4",
    openai_reasoning=OpenAIReasoningConfig(effort="medium"),
    notes="Configured to use medium reasoning effort for evaluation runs.",
)
GPT_5_4_MINI = _openai_model(
    key="gpt-5.4-mini",
    label="GPT-5.4 mini",
    api_model="gpt-5.4-mini",
    docs_suffix="gpt-5.4-mini",
    openai_reasoning=OpenAIReasoningConfig(effort="medium"),
    notes="Configured to use medium reasoning effort for evaluation runs.",
)
GPT_5_4_NANO = _openai_model(
    key="gpt-5.4-nano",
    label="GPT-5.4 nano",
    api_model="gpt-5.4-nano",
    docs_suffix="gpt-5.4-nano",
    openai_reasoning=OpenAIReasoningConfig(effort="medium"),
    notes=(
        "Configured to use medium reasoning effort for evaluation runs. "
        "Defaults to OpenAI's published 128k max output tokens."
    ),
)

CLAUDE_SONNET_4_6 = _anthropic_model(
    key="claude-sonnet-4-6",
    label="Claude Sonnet 4.6",
    api_model="claude-sonnet-4-6",
    default_max_output_tokens=CLAUDE_SONNET_MAX_OUTPUT_TOKENS,
    anthropic_thinking=AnthropicThinkingConfig(enabled=False),
    anthropic_effort="max",
    notes=(
        "Extended thinking is supported by the model, but disabled by default in this "
        "evaluator. Anthropic effort defaults to max unless overridden."
    ),
)
CLAUDE_SONNET_4_6_REASONING = _anthropic_model(
    key="claude-sonnet-4-6-reasoning",
    label="Claude Sonnet 4.6 (reasoning)",
    api_model="claude-sonnet-4-6",
    default_max_output_tokens=CLAUDE_SONNET_MAX_OUTPUT_TOKENS,
    anthropic_thinking=AnthropicThinkingConfig(enabled=True, mode="adaptive"),
    anthropic_effort="max",
    notes=(
        "Uses Claude Sonnet 4.6 with Anthropic adaptive extended thinking enabled. "
        "`budget_tokens` is deprecated on Sonnet 4.6; adaptive thinking lets the "
        "model decide depth. Anthropic effort defaults to max."
    ),
)
CLAUDE_HAIKU_4_5 = _anthropic_model(
    key="claude-haiku-4-5",
    label="Claude Haiku 4.5",
    api_model="claude-haiku-4-5",
    anthropic_thinking=AnthropicThinkingConfig(enabled=False),
    notes=(
        "Anthropic's official docs list claude-haiku-4-5 as the alias and "
        "claude-haiku-4-5-20251001 as the snapshot ID."
    ),
)

GEMINI_3_FLASH_PREVIEW = _gemini_model(
    key="gemini-3-flash-preview",
    label="Gemini 3 Flash Preview",
    api_model="gemini-3-flash-preview",
    notes=(
        "Google's Gemini 3 guide documents Gemini 3 Flash Preview with a 1M token "
        "context window, 64k max output tokens, and dynamic high thinking by default."
    ),
)
GEMINI_3_1_PRO_PREVIEW = _gemini_model(
    key="gemini-3.1-pro-preview",
    label="Gemini 3.1 Pro Preview",
    api_model="gemini-3.1-pro-preview",
    notes=(
        "Google's Gemini 3 guide documents Gemini 3.1 Pro Preview with a 1M token "
        "context window, 64k max output tokens, and dynamic high thinking by default."
    ),
)

MISTRAL_LARGE_2512 = _mistral_model(
    key="mistral-large-2512",
    label="Mistral Large 3",
    api_model="mistral-large-2512",
    docs_url="https://docs.mistral.ai/models/mistral-large-3-25-12",
    default_max_output_tokens=MISTRAL_MAX_OUTPUT_TOKENS,
    notes=(
        "Mistral Large 3 v25.12 supports chat completions and structured outputs "
        "via the Chat Completions API. Default output budget set to 40k for "
        "headroom; the API enforces prompt + max_tokens <= context (256k)."
    ),
)
MAGISTRAL_MEDIUM_2509 = _mistral_model(
    key="magistral-medium-2509",
    label="Magistral Medium 1.2",
    api_model="magistral-medium-2509",
    docs_url="https://docs.mistral.ai/models/magistral-medium-1-2-25-09",
    default_max_output_tokens=MISTRAL_MAX_OUTPUT_TOKENS,
    notes=(
        "Magistral Medium 1.2 v25.09 is Mistral's frontier-class multimodal "
        "reasoning model and supports chat completions and structured outputs "
        "via the Chat Completions API. Default output budget set to 40k."
    ),
)
MISTRAL_MEDIUM_2508 = _mistral_model(
    key="mistral-medium-2508",
    label="Mistral Medium 3.1",
    api_model="mistral-medium-2508",
    docs_url="https://docs.mistral.ai/models/mistral-medium-3-1-25-08",
    notes=(
        "Mistral Medium 3.1 v25.08 supports chat completions and structured outputs "
        "via the Chat Completions API."
    ),
)

GROK_4_3_REASONING_HIGH = _xai_model(
    key="grok-4.3-reasoning-high",
    label="Grok 4.3 (reasoning high)",
    api_model="grok-4.3",
    openai_reasoning=OpenAIReasoningConfig(effort="high"),
    notes=(
        "Pinned benchmark alias for xAI Grok 4.3 with reasoning.effort='high'. "
        "Uses xAI's OpenAI-compatible Responses API."
    ),
)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    GPT_4_1_MINI.key: GPT_4_1_MINI,
    GPT_5_2.key: GPT_5_2,
    "gpt-5.2-reasoning-xhigh": replace(
        GPT_5_2,
        key="gpt-5.2-reasoning-xhigh",
        label="GPT-5.2 (reasoning xhigh)",
        default_max_output_tokens=GPT_5_2_MAX_OUTPUT_TOKENS,
        openai_reasoning=OpenAIReasoningConfig(effort="xhigh"),
        notes=(
            "Pinned benchmark alias for GPT-5.2 at reasoning.effort='xhigh'. "
            "Defaults to OpenAI's published 128k max output tokens."
        ),
    ),
    GPT_5_5.key: GPT_5_5,
    "gpt-5.5-reasoning-max": replace(
        GPT_5_5,
        key="gpt-5.5-reasoning-max",
        label="GPT-5.5 (reasoning max)",
        default_max_output_tokens=GPT_5_5_MAX_OUTPUT_TOKENS,
        openai_reasoning=OpenAIReasoningConfig(effort="xhigh"),
        notes=(
            "Pinned benchmark alias for GPT-5.5 at OpenAI's highest documented "
            "reasoning.effort value, 'xhigh'. The alias keeps the local benchmark "
            "naming convention of reasoning-max while sending the valid API value."
        ),
    ),
    GPT_5_4.key: GPT_5_4,
    GPT_5_4_MINI.key: GPT_5_4_MINI,
    "gpt-5.4-mini-no-reasoning": replace(
        GPT_5_4_MINI,
        key="gpt-5.4-mini-no-reasoning",
        label="GPT-5.4 mini (no reasoning)",
        openai_reasoning=OpenAIReasoningConfig(effort="none"),
        notes="Pinned alias for GPT-5.4 mini with reasoning.effort='none'.",
    ),
    "gpt-5.4-reasoning-xhigh": replace(
        GPT_5_4,
        key="gpt-5.4-reasoning-xhigh",
        label="GPT-5.4 (reasoning xhigh)",
        default_max_output_tokens=GPT_5_4_MAX_OUTPUT_TOKENS,
        openai_reasoning=OpenAIReasoningConfig(effort="xhigh"),
        notes=(
            "Pinned benchmark alias for GPT-5.4 at reasoning.effort='xhigh'. "
            "Defaults to OpenAI's published 128k max output tokens."
        ),
    ),
    "gpt-5.4-mini-reasoning-xhigh": replace(
        GPT_5_4_MINI,
        key="gpt-5.4-mini-reasoning-xhigh",
        label="GPT-5.4 mini (reasoning xhigh)",
        default_max_output_tokens=GPT_5_4_MAX_OUTPUT_TOKENS,
        openai_reasoning=OpenAIReasoningConfig(effort="xhigh"),
        notes=(
            "Pinned benchmark alias for GPT-5.4 mini at reasoning.effort='xhigh'. "
            "Defaults to OpenAI's published 128k max output tokens."
        ),
    ),
    GPT_5_4_NANO.key: replace(
        GPT_5_4_NANO,
        default_max_output_tokens=GPT_5_4_MAX_OUTPUT_TOKENS,
    ),
    "gpt-5.4-nano-reasoning-xhigh": replace(
        GPT_5_4_NANO,
        key="gpt-5.4-nano-reasoning-xhigh",
        label="GPT-5.4 nano (reasoning xhigh)",
        default_max_output_tokens=GPT_5_4_MAX_OUTPUT_TOKENS,
        openai_reasoning=OpenAIReasoningConfig(effort="xhigh"),
        notes=(
            "Pinned benchmark alias for GPT-5.4 nano at reasoning.effort='xhigh'. "
            "Defaults to OpenAI's published 128k max output tokens."
        ),
    ),
    CLAUDE_SONNET_4_6.key: CLAUDE_SONNET_4_6,
    CLAUDE_SONNET_4_6_REASONING.key: CLAUDE_SONNET_4_6_REASONING,
    "claude-opus-4-6-reasoning-high": _anthropic_model(
        key="claude-opus-4-6-reasoning-high",
        label="Claude Opus 4.6 (reasoning high)",
        api_model="claude-opus-4-6",
        anthropic_thinking=AnthropicThinkingConfig(enabled=True, budget_tokens=1024),
        anthropic_effort="high",
        notes=(
            "Pinned benchmark alias for Claude Opus 4.6 with extended thinking "
            "enabled at the minimum 1024-token budget and effort='high'."
        ),
    ),
    "claude-opus-4-6-reasoning-max": _anthropic_model(
        key="claude-opus-4-6-reasoning-max",
        label="Claude Opus 4.6 (reasoning max)",
        api_model="claude-opus-4-6",
        anthropic_thinking=AnthropicThinkingConfig(enabled=True, mode="adaptive"),
        anthropic_effort="max",
        default_max_output_tokens=CLAUDE_OPUS_MAX_OUTPUT_TOKENS,
        notes=(
            "Pinned benchmark alias for Claude Opus 4.6 with adaptive extended "
            "thinking enabled and effort='max'. Defaults to Anthropic's published "
            "128k max output for synchronous Messages API requests. Used to put "
            "Opus 4.6 on equal footing with Opus 4.7 reasoning-max."
        ),
    ),
    "claude-sonnet-4-6-reasoning-max": replace(
        CLAUDE_SONNET_4_6_REASONING,
        key="claude-sonnet-4-6-reasoning-max",
        label="Claude Sonnet 4.6 (reasoning max)",
        anthropic_effort="max",
        notes=(
            "Pinned benchmark alias for Claude Sonnet 4.6 with adaptive extended "
            "thinking enabled and effort='max'."
        ),
    ),
    "claude-opus-4-7-reasoning-max": _anthropic_model(
        key="claude-opus-4-7-reasoning-max",
        label="Claude Opus 4.7 (reasoning max)",
        api_model="claude-opus-4-7",
        anthropic_thinking=AnthropicThinkingConfig(enabled=True, mode="adaptive"),
        anthropic_effort="max",
        default_max_output_tokens=CLAUDE_OPUS_MAX_OUTPUT_TOKENS,
        notes=(
            "Pinned benchmark alias for Claude Opus 4.7 with adaptive thinking "
            "enabled and effort='max'. Defaults to Anthropic's published 128k "
            "max output for synchronous Messages API requests."
        ),
    ),
    CLAUDE_HAIKU_4_5.key: CLAUDE_HAIKU_4_5,
    GEMINI_3_FLASH_PREVIEW.key: GEMINI_3_FLASH_PREVIEW,
    "gemini-3-flash-preview-thinking-high": replace(
        GEMINI_3_FLASH_PREVIEW,
        key="gemini-3-flash-preview-thinking-high",
        label="Gemini 3 Flash Preview (thinking high)",
        docs_url=GEMINI_THINKING_DOCS_URL,
        default_max_output_tokens=GEMINI_3_MAX_OUTPUT_TOKENS,
        gemini_thinking=GeminiThinkingConfig(level="high"),
        notes=(
            "Pinned benchmark alias for Gemini 3 Flash Preview with explicit "
            "thinkingConfig.thinkingLevel='high'. Defaults to Google's published "
            "64k max output tokens."
        ),
    ),
    GEMINI_3_1_PRO_PREVIEW.key: GEMINI_3_1_PRO_PREVIEW,
    "gemini-3.1-pro-preview-thinking-high": replace(
        GEMINI_3_1_PRO_PREVIEW,
        key="gemini-3.1-pro-preview-thinking-high",
        label="Gemini 3.1 Pro Preview (thinking high)",
        docs_url=GEMINI_THINKING_DOCS_URL,
        default_max_output_tokens=GEMINI_3_MAX_OUTPUT_TOKENS,
        gemini_thinking=GeminiThinkingConfig(level="high"),
        notes=(
            "Pinned benchmark alias for Gemini 3.1 Pro Preview with explicit "
            "thinkingConfig.thinkingLevel='high'. Defaults to Google's published "
            "64k max output tokens."
        ),
    ),
    MISTRAL_LARGE_2512.key: MISTRAL_LARGE_2512,
    MAGISTRAL_MEDIUM_2509.key: MAGISTRAL_MEDIUM_2509,
    MISTRAL_MEDIUM_2508.key: MISTRAL_MEDIUM_2508,
    GROK_4_3_REASONING_HIGH.key: GROK_4_3_REASONING_HIGH,
}

_MODEL_ORDER: tuple[str, ...] = (
    "claude-opus-4-7-reasoning-max",
    "claude-opus-4-6-reasoning-max",
    "claude-sonnet-4-6-reasoning-max",
    "gpt-5.5-reasoning-max",
    "gpt-5.2-reasoning-xhigh",
    "gpt-5.4-reasoning-xhigh",
    "gpt-5.4-mini-reasoning-xhigh",
    "gpt-5.4-nano-reasoning-xhigh",
    "gemini-3.1-pro-preview-thinking-high",
    "gemini-3-flash-preview-thinking-high",
    "mistral-large-2512",
    "magistral-medium-2509",
    "grok-4.3-reasoning-high",
)
_MODEL_ORDER_INDEX: dict[str, int] = {key: index for index, key in enumerate(_MODEL_ORDER)}


def get_model_config(key: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model config '{key}'. Expected one of: {known}") from exc


def list_model_configs(*, provider: ProviderName | None = None) -> list[ModelConfig]:
    configs = sorted(
        MODEL_CONFIGS.values(),
        key=lambda config: (_MODEL_ORDER_INDEX.get(config.key, len(_MODEL_ORDER)), config.key),
    )
    if provider is None:
        return configs
    return [config for config in configs if config.provider == provider]


def model_choices(*, providers: tuple[ProviderName, ...] | None = None) -> tuple[str, ...]:
    if providers is None:
        return tuple(config.key for config in list_model_configs())
    allowed = set(providers)
    return tuple(config.key for config in list_model_configs() if config.provider in allowed)


_ANTHROPIC_EFFORT_MODELS: frozenset[str] = frozenset(
    {
        "claude-mythos-preview",
        "claude-opus-4-5",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)
_ANTHROPIC_MAX_EFFORT_MODELS: frozenset[str] = frozenset(
    {
        "claude-mythos-preview",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)
_ANTHROPIC_XHIGH_EFFORT_MODELS: frozenset[str] = frozenset(
    {
        "claude-opus-4-7",
    }
)


def supports_anthropic_effort(model_config: ModelConfig) -> bool:
    return model_config.provider == "anthropic" and model_config.api_model in _ANTHROPIC_EFFORT_MODELS


def apply_reasoning_effort_override(
    model_config: ModelConfig,
    effort: ReasoningEffortOverride,
) -> ModelConfig:
    if model_config.openai_reasoning is not None:
        if model_config.provider == "xai" and effort not in {"none", "low", "medium", "high"}:
            raise ValueError(
                f"xAI reasoning does not support effort={effort!r}; use one of "
                f"{('none', 'low', 'medium', 'high')}."
            )
        if effort == "max":
            raise ValueError(
                f"OpenAI reasoning does not support effort={effort!r}; use one of "
                f"{('none', 'minimal', 'low', 'medium', 'high', 'xhigh')}."
            )
        return replace(
            model_config,
            openai_reasoning=replace(model_config.openai_reasoning, effort=effort),
        )

    if supports_anthropic_effort(model_config):
        if effort == "none":
            return replace(model_config, anthropic_effort=None)
        if effort == "minimal":
            raise ValueError(
                f"Anthropic effort does not support {effort!r}; use one of "
                f"{('low', 'medium', 'high', 'max')} and xhigh on Claude Opus 4.7."
            )
        if effort == "xhigh" and model_config.api_model not in _ANTHROPIC_XHIGH_EFFORT_MODELS:
            raise ValueError(
                f"Anthropic effort {effort!r} is unsupported for {model_config.key}; "
                "xhigh is available only on Claude Opus 4.7."
            )
        if effort == "max" and model_config.api_model not in _ANTHROPIC_MAX_EFFORT_MODELS:
            raise ValueError(
                f"Anthropic effort {effort!r} is unsupported for {model_config.key}; "
                "max is available only on Claude Mythos Preview, Claude Opus 4.7, "
                "Claude Opus 4.6, and Claude Sonnet 4.6."
            )
        return replace(model_config, anthropic_effort=effort)

    raise ValueError(
        "--reasoning-effort requires a model with OpenAI reasoning or Anthropic "
        f"effort support; {model_config.key} does not support it."
    )
