from __future__ import annotations

from typing import Any

from emoji_bench.providers.transport import ContinuationMode, ContinuationResponse
from emoji_bench.model_registry import ModelConfig
from emoji_bench.providers.clients import extract_anthropic_usage


def request_anthropic_messages(
    *,
    client: Any,
    model_config: ModelConfig,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    mode: ContinuationMode,
) -> ContinuationResponse:
    options: dict[str, Any] = {
        "model": model_config.api_model,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if (
        model_config.anthropic_thinking is not None
        and model_config.anthropic_thinking.enabled
        and model_config.anthropic_thinking.budget_tokens is not None
    ):
        if model_config.anthropic_thinking.budget_tokens >= max_output_tokens:
            raise ValueError("Anthropic thinking budget must be less than max_output_tokens")
        options["thinking"] = {
            "type": "enabled",
            "budget_tokens": model_config.anthropic_thinking.budget_tokens,
        }
    if model_config.anthropic_effort is not None:
        options["output_config"] = {"effort": model_config.anthropic_effort}

    response = client.messages.create(**options)
    return ContinuationResponse(
        raw_continuation_text=_anthropic_text(response),
        response_id=getattr(response, "id", None),
        usage=extract_anthropic_usage(response),
        mode=mode,
    )


def _anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or ()
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)
