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

from typing import Any

from emoji_bench.continuation_formatter import (
    format_continuation_single_turn,
    get_turn_2_prompt,
)
from emoji_bench.continuation_provider_anthropic import request_anthropic_messages
from emoji_bench.continuation_provider_gemini import request_gemini_messages
from emoji_bench.continuation_provider_mistral import request_mistral_messages
from emoji_bench.continuation_provider_openai import request_openai_messages
from emoji_bench.continuation_transport import ContinuationMode, ContinuationResponse
from emoji_bench.model_registry import ModelConfig


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
        return request_openai_messages(
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
        return request_mistral_messages(
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
        return request_gemini_messages(
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
        return request_anthropic_messages(
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


def _dispatch_single_turn(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
) -> ContinuationResponse:
    provider = model_config.provider
    if provider == "openai":
        return request_openai_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "anthropic":
        return request_anthropic_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "mistral":
        return request_mistral_messages(
            client=client,
            model_config=model_config,
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    if provider == "gemini":
        return request_gemini_messages(
            client=client,
            model_config=model_config,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            max_output_tokens=max_output_tokens,
            mode="single_turn",
        )
    raise ValueError(f"Unsupported provider: {provider}")
