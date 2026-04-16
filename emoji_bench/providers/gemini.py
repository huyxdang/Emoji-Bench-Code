from __future__ import annotations

from typing import Any

from emoji_bench.providers.transport import ContinuationMode, ContinuationResponse
from emoji_bench.model_registry import ModelConfig
from emoji_bench.providers.clients import extract_gemini_usage


def request_gemini_messages(
    *,
    client: Any,
    model_config: ModelConfig,
    contents: list[dict[str, Any]],
    max_output_tokens: int,
    mode: ContinuationMode,
) -> ContinuationResponse:
    options: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
        },
    }
    if model_config.gemini_thinking is not None:
        options["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": model_config.gemini_thinking.level,
        }
    response = client.generate_content(model=model_config.api_model, options=options)
    return ContinuationResponse(
        raw_continuation_text=_gemini_text(response),
        response_id=response.get("responseId"),
        usage=extract_gemini_usage(response),
        mode=mode,
    )


def _gemini_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or ()
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces)
