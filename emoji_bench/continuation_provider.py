"""Phase 4: provider plumbing for the E-CONTINUE benchmark.

Two request modes are supported:

- ``prefill``      Every provider receives a 3-message conversation list
                   (user -> assistant-prefill -> user "Please continue.").
- ``single_turn``  Every provider receives one flat user message produced
                   by ``format_continuation_single_turn``. This is the mode
                   used on channels that do not accept multi-message chats.

Requests do not use a system prompt, do not request structured output, and
return raw text only. Judge and validator logic run later against that raw
continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from emoji_bench.continuation_formatter import (
    format_continuation_single_turn,
    get_turn_2_prompt,
)
from emoji_bench.model_registry import ModelConfig
from emoji_bench.provider_clients import (
    ProviderUsage,
    extract_anthropic_usage,
    extract_gemini_usage,
    extract_mistral_usage,
    extract_openai_usage,
)


ContinuationMode = Literal["prefill", "single_turn"]


@dataclass(frozen=True)
class ContinuationResponse:
    raw_continuation_text: str
    response_id: str | None
    usage: ProviderUsage | None
    mode: ContinuationMode


# --- Public entry point -----------------------------------------------------


def request_continuation(
    *,
    client: Any,
    model_config: ModelConfig,
    turn_1_user: str,
    turn_1_assistant_prefill: str,
    max_output_tokens: int,
    mode: ContinuationMode = "prefill",
    turn_2_user: str = get_turn_2_prompt(0),
) -> ContinuationResponse:
    """Send a continuation request to a provider and return raw output text."""
    if mode == "single_turn":
        prompt = format_continuation_single_turn(
            turn_1_user=turn_1_user,
            turn_1_assistant_prefill=turn_1_assistant_prefill,
            turn_2_user=turn_2_user,
        )
        return _dispatch_single_turn(
            client=client,
            model_config=model_config,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )

    if mode == "prefill":
        return _dispatch_three_message_list(
            client=client,
            model_config=model_config,
            turn_1_user=turn_1_user,
            turn_1_assistant_prefill=turn_1_assistant_prefill,
            turn_2_user=turn_2_user,
            max_output_tokens=max_output_tokens,
        )

    raise ValueError(f"Unsupported continuation mode: {mode}")

# --- 3-message conversation list -------------------------------------------


def _dispatch_three_message_list(
    *,
    client: Any,
    model_config: ModelConfig,
    turn_1_user: str,
    turn_1_assistant_prefill: str,
    turn_2_user: str,
    max_output_tokens: int,
) -> ContinuationResponse:
    provider = model_config.provider
    if provider == "openai":
        return _request_openai_messages(
            client=client,
            model_config=model_config,
            messages=[
                {"role": "user", "content": turn_1_user},
                {"role": "assistant", "content": turn_1_assistant_prefill},
                {"role": "user", "content": turn_2_user},
            ],
            max_output_tokens=max_output_tokens,
            mode="prefill",
        )
    if provider == "mistral":
        return _request_mistral_messages(
            client=client,
            model_config=model_config,
            messages=[
                {"role": "user", "content": turn_1_user},
                {"role": "assistant", "content": turn_1_assistant_prefill},
                {"role": "user", "content": turn_2_user},
            ],
            max_output_tokens=max_output_tokens,
            mode="prefill",
        )
    if provider == "gemini":
        return _request_gemini_messages(
            client=client,
            model_config=model_config,
            contents=[
                {"role": "user", "parts": [{"text": turn_1_user}]},
                {"role": "model", "parts": [{"text": turn_1_assistant_prefill}]},
                {"role": "user", "parts": [{"text": turn_2_user}]},
            ],
            max_output_tokens=max_output_tokens,
            mode="prefill",
        )
    if provider == "anthropic":
        return _request_anthropic_messages(
            client=client,
            model_config=model_config,
            messages=[
                {"role": "user", "content": turn_1_user},
                {"role": "assistant", "content": turn_1_assistant_prefill},
                {"role": "user", "content": turn_2_user},
            ],
            max_output_tokens=max_output_tokens,
            mode="prefill",
        )
    raise ValueError(f"Unsupported provider: {provider}")


# --- Single-turn dispatch --------------------------------------------------


def _dispatch_single_turn(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
) -> ContinuationResponse:
    provider = model_config.provider
    if provider == "openai":
        return _request_openai_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "anthropic":
        return _request_anthropic_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "mistral":
        return _request_mistral_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "gemini":
        return _request_gemini_messages(
            client=client,
            model_config=model_config,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    raise ValueError(f"Unsupported provider: {provider}")


# --- Per-provider message senders ------------------------------------------


def _request_openai_messages(
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


def _request_anthropic_messages(
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


def _request_mistral_messages(
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


def _request_gemini_messages(
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
    response = client.generate_content(model=model_config.api_model, options=options)
    return ContinuationResponse(
        raw_continuation_text=_gemini_text(response),
        response_id=response.get("responseId"),
        usage=extract_gemini_usage(response),
        mode=mode,
    )


# --- Output extraction (text only — no JSON schema) ------------------------


def _anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or ()
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)


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
