from __future__ import annotations

from typing import Any

from emoji_bench.providers.transport import ContinuationMode, ContinuationResponse
from emoji_bench.model_registry import ModelConfig
from emoji_bench.providers.clients import extract_openai_usage


def request_openai_messages(
    *,
    client: Any,
    model_config: ModelConfig,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    mode: ContinuationMode,
) -> ContinuationResponse:
    options: dict[str, Any] = {
        "model": model_config.api_model,
        "input": messages,
        "max_output_tokens": max_output_tokens,
    }
    if model_config.openai_reasoning is not None:
        reasoning: dict[str, str] = {"effort": model_config.openai_reasoning.effort}
        if model_config.openai_reasoning.summary:
            reasoning["summary"] = model_config.openai_reasoning.summary
        options["reasoning"] = reasoning

    response = client.responses.create(**options)
    return ContinuationResponse(
        raw_continuation_text=_openai_text(response),
        response_id=getattr(response, "id", None),
        usage=extract_openai_usage(response),
        mode=mode,
    )


def _openai_text(response: Any) -> str:
    direct = getattr(response, "output_text", "")
    if direct:
        return direct

    parts: list[str] = []
    for output in getattr(response, "output", ()) or ():
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()) or ():
            if getattr(content, "type", None) == "output_text" and hasattr(content, "text"):
                parts.append(content.text)
    return "".join(parts)
