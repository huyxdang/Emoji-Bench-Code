from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from emoji_bench.model_registry import ModelConfig, ProviderName


GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class _MistralClient:
    api_key: str

    def chat_complete(self, options: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(options).encode("utf-8")
        request = urllib_request.Request(
            MISTRAL_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, context=_api_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            message = f"Mistral API request failed with status {exc.code}"
            if body:
                message += f": {body}"
            raise RuntimeError(message) from exc


@dataclass(frozen=True)
class _GeminiClient:
    api_key: str

    def generate_content(self, *, model: str, options: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(options).encode("utf-8")
        request = urllib_request.Request(
            f"{GEMINI_API_BASE_URL}/{model}:generateContent",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, context=_api_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            message = f"Gemini API request failed with status {exc.code}"
            if body:
                message += f": {body}"
            raise RuntimeError(message) from exc


def _api_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def resolve_api_key(
    *,
    model_config: ModelConfig,
    explicit_api_key: str | None,
    env: dict[str, str],
) -> str:
    api_key = explicit_api_key or env.get(model_config.api_key_env_var)
    if api_key:
        return api_key
    raise RuntimeError(
        f"{model_config.api_key_env_var} is required for {model_config.key}. "
        "Set it in the environment or pass --api-key."
    )


def make_client(provider: ProviderName, *, api_key: str) -> Any:
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAI evaluation. "
                'Install with `pip install -e ".[openai]"`.'
            ) from exc
        return OpenAI(api_key=api_key)

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is required for Anthropic evaluation. "
                'Install with `pip install -e ".[anthropic]"`.'
            ) from exc
        return Anthropic(api_key=api_key)

    if provider == "mistral":
        return _MistralClient(api_key=api_key)

    if provider == "gemini":
        return _GeminiClient(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")


def extract_openai_usage(response: Any) -> ProviderUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    reasoning_tokens = None

    output_details = getattr(usage, "output_tokens_details", None)
    if output_details is not None:
        reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def extract_anthropic_usage(response: Any) -> ProviderUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=None,
        total_tokens=total_tokens,
    )


def extract_mistral_usage(response: dict[str, Any]) -> ProviderUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    return ProviderUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        reasoning_tokens=None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )


def extract_gemini_usage(response: dict[str, Any]) -> ProviderUsage | None:
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")
    thoughts_tokens = usage.get("thoughtsTokenCount")
    total_tokens = usage.get("totalTokenCount")

    return ProviderUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        reasoning_tokens=thoughts_tokens if isinstance(thoughts_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )
