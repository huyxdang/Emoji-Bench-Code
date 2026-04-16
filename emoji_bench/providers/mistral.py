from __future__ import annotations

from typing import Any

from emoji_bench.continuation_transport import ContinuationMode, ContinuationResponse
from emoji_bench.model_registry import ModelConfig
from emoji_bench.provider_clients import extract_mistral_usage


def request_mistral_messages(
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
        "temperature": 0,
    }
    response = client.chat_complete(options)
    return ContinuationResponse(
        raw_continuation_text=_mistral_text(response),
        response_id=response.get("id"),
        usage=extract_mistral_usage(response),
        mode=mode,
    )


def _mistral_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or ()
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
