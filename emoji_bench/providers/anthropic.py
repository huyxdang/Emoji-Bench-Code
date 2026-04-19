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
    thinking = model_config.anthropic_thinking
    if thinking is not None and thinking.enabled:
        if thinking.mode == "adaptive":
            options["thinking"] = {"type": "adaptive"}
        elif thinking.budget_tokens is not None:
            if thinking.budget_tokens >= max_output_tokens:
                raise ValueError("Anthropic thinking budget must be less than max_output_tokens")
            options["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens,
            }
    if model_config.anthropic_effort is not None:
        options["output_config"] = {"effort": model_config.anthropic_effort}

    response = _send_anthropic_request(client=client, options=options)
    return ContinuationResponse(
        raw_continuation_text=_anthropic_text(response),
        response_id=getattr(response, "id", None),
        usage=extract_anthropic_usage(response),
        mode=mode,
    )


def _send_anthropic_request(*, client: Any, options: dict[str, Any]) -> Any:
    messages_api = client.messages
    stream = getattr(messages_api, "stream", None)
    if callable(stream):
        # Anthropic's SDK requires streaming for long-running requests. Using the
        # helper keeps the final response shape identical to messages.create().
        with stream(**options) as response_stream:
            return response_stream.get_final_message()
    return messages_api.create(**options)


def _anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or ()
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)
