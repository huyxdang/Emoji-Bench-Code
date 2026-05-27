from dataclasses import replace
from types import SimpleNamespace

import pytest

from emoji_bench.continuation_formatter import (
    SINGLE_TURN_NEXT_MESSAGE_HEADER,
    SINGLE_TURN_WORK_HEADER,
    format_continuation_single_turn,
    get_turn_2_prompt,
)
from emoji_bench.providers.continuation import (
    ContinuationResponse,
    request_continuation,
)
from emoji_bench.model_registry import GeminiThinkingConfig, ModelConfig, get_model_config


class _FakeMessagesAPI:
    def __init__(self, response: object):
        self._response = response
        self.calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def create(self, **options):
        self.create_calls.append(options)
        self.calls.append(options)
        return self._response

    def stream(self, **options):
        self.stream_calls.append(options)
        self.calls.append(options)
        return _FakeAnthropicStream(self._response)


class _FakeAnthropicStream:
    def __init__(self, response: object):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get_final_message(self):
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: object):
        self.messages = _FakeMessagesAPI(response)


class _FakeResponsesAPI:
    def __init__(self, response: object):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **options):
        self.calls.append(options)
        return self._response


class _FakeOpenAIClient:
    def __init__(self, response: object):
        self.responses = _FakeResponsesAPI(response)


class _FakeMistralClient:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[dict] = []

    def chat_complete(self, options: dict) -> dict:
        self.calls.append(options)
        return self._response


class _FakeGeminiClient:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[dict] = []

    def generate_content(self, *, model: str, options: dict) -> dict:
        self.calls.append({"model": model, "options": options})
        return self._response


class _FakeOpenRouterClient:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[dict] = []

    def chat_complete(self, options: dict) -> dict:
        self.calls.append(options)
        return self._response


def _anthropic_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_anthropic_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test",
        content=[_anthropic_text_block(text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )


def _make_openai_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_test",
        output_text=text,
        output=[],
        usage=None,
    )


def _native_gemini_config(*, thinking_level: str | None = None) -> ModelConfig:
    return ModelConfig(
        key="native-gemini-test",
        label="Native Gemini Test",
        provider="gemini",
        api_model="gemini-test",
        docs_url="https://ai.google.dev/gemini-api/docs",
        api_key_env_var="GEMINI_API_KEY",
        default_max_output_tokens=4096,
        gemini_thinking=(
            None
            if thinking_level is None
            else GeminiThinkingConfig(level=thinking_level)  # type: ignore[arg-type]
        ),
    )


def test_single_turn_format_contains_turn_1_then_work_header_then_prefill():
    level_0 = get_turn_2_prompt(0)
    rendered = format_continuation_single_turn(
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="Start: x\nStep 1: x = y    [by ⊕ table]",
        turn_2_user=level_0,
    )

    assert rendered.startswith("[T1U]\n\n")
    assert SINGLE_TURN_WORK_HEADER in rendered
    assert SINGLE_TURN_NEXT_MESSAGE_HEADER in rendered
    assert rendered.endswith(f"{SINGLE_TURN_NEXT_MESSAGE_HEADER}\n{level_0}")


def test_request_continuation_prefill_anthropic_sends_three_message_conversation():
    level_0 = get_turn_2_prompt(0)
    response = _make_anthropic_response(" continuing the work...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-haiku-4-5")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert isinstance(result, ContinuationResponse)
    assert result.mode == "prefill"
    assert result.raw_continuation_text == " continuing the work..."

    sent = client.messages.calls[0]
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": level_0},
    ]
    assert "system" not in sent
    assert "output_config" not in sent


def test_request_continuation_anthropic_uses_stream_helper_when_available():
    response = _make_anthropic_response(" continuing the work...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-haiku-4-5")

    request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert len(client.messages.stream_calls) == 1
    assert client.messages.create_calls == []


def test_request_continuation_single_turn_anthropic_sends_one_user_message():
    level_0 = get_turn_2_prompt(0)
    response = _make_anthropic_response("Step 3: ...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-haiku-4-5")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="single_turn",
    )

    assert result.mode == "single_turn"

    sent = client.messages.calls[0]
    messages = sent["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "[T1U]" in messages[0]["content"]
    assert SINGLE_TURN_WORK_HEADER in messages[0]["content"]
    assert "[PREFILL]" in messages[0]["content"]
    assert level_0 in messages[0]["content"]


def test_request_continuation_single_turn_uses_custom_turn_2_prompt():
    response = _make_anthropic_response("Step 3: ...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-haiku-4-5")
    custom_turn_2 = "Please continue, but double-check the last step first."

    request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        turn_2_user=custom_turn_2,
        max_output_tokens=512,
        mode="single_turn",
    )

    sent = client.messages.calls[0]
    assert custom_turn_2 in sent["messages"][0]["content"]


def test_request_continuation_sonnet_sets_default_anthropic_effort():
    response = _make_anthropic_response("Step 3: ...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-sonnet-4-6")

    request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="single_turn",
    )

    sent = client.messages.calls[0]
    assert sent["output_config"] == {"effort": "max"}
    assert "thinking" not in sent


def test_request_continuation_sonnet_reasoning_sends_thinking_and_effort():
    response = _make_anthropic_response("Step 3: ...")
    client = _FakeAnthropicClient(response)
    model_config = replace(get_model_config("claude-sonnet-4-6-reasoning"), anthropic_effort="low")

    request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=2048,
        mode="prefill",
    )

    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "low"}


def test_request_continuation_opus_47_reasoning_max_sends_adaptive_thinking_and_effort():
    response = _make_anthropic_response("Step 3: ...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-opus-4-7-reasoning-max")

    request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=2048,
        mode="prefill",
    )

    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "max"}


def test_request_continuation_prefill_openai_sends_three_message_conversation():
    level_0 = get_turn_2_prompt(0)
    response = _make_openai_response("(continuation)")
    client = _FakeOpenAIClient(response)
    model_config = get_model_config("gpt-5.4-mini")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.mode == "prefill"
    assert result.raw_continuation_text == "(continuation)"

    sent = client.responses.calls[0]
    assert sent["max_output_tokens"] == 512
    assert sent["input"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": level_0},
    ]
    assert sent.get("reasoning") == {"effort": "medium"}


def test_request_continuation_prefill_mistral_sends_three_message_conversation():
    level_0 = get_turn_2_prompt(0)
    client = _FakeMistralClient(
        {
            "id": "mistral_id",
            "choices": [{"message": {"content": "(mistral output)"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
    )
    model_config = get_model_config("mistral-medium-2508")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.raw_continuation_text == "(mistral output)"
    assert result.mode == "prefill"

    sent = client.calls[0]
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": level_0},
    ]
    assert "response_format" not in sent


def test_request_continuation_prefill_gemini_uses_model_role_for_prefill():
    level_0 = get_turn_2_prompt(0)
    client = _FakeGeminiClient(
        {
            "responseId": "gem_id",
            "candidates": [{"content": {"parts": [{"text": "(gemini continuation)"}]}}],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 6,
                "totalTokenCount": 11,
            },
        }
    )
    model_config = _native_gemini_config()

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.raw_continuation_text == "(gemini continuation)"
    assert result.mode == "prefill"

    sent = client.calls[0]["options"]
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "[T1U]"}]},
        {"role": "model", "parts": [{"text": "[PREFILL]"}]},
        {"role": "user", "parts": [{"text": level_0}]},
    ]
    gen_cfg = sent["generationConfig"]
    assert "responseJsonSchema" not in gen_cfg
    assert "responseMimeType" not in gen_cfg


def test_request_continuation_prefill_gemini_thinking_high_sets_thinking_level():
    level_0 = get_turn_2_prompt(0)
    client = _FakeGeminiClient(
        {
            "responseId": "gem_id",
            "candidates": [{"content": {"parts": [{"text": "(gemini continuation)"}]}}],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 6,
                "totalTokenCount": 11,
                "thoughtsTokenCount": 3,
            },
        }
    )
    model_config = _native_gemini_config(thinking_level="high")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.raw_continuation_text == "(gemini continuation)"
    assert result.mode == "prefill"

    sent = client.calls[0]["options"]
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "[T1U]"}]},
        {"role": "model", "parts": [{"text": "[PREFILL]"}]},
        {"role": "user", "parts": [{"text": level_0}]},
    ]
    assert sent["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_request_continuation_openrouter_gemini_thinking_high_sends_reasoning_effort():
    level_0 = get_turn_2_prompt(0)
    client = _FakeOpenRouterClient(
        {
            "id": "or_id",
            "choices": [{"message": {"content": "(openrouter gemini continuation)"}}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "total_tokens": 14,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
    )
    model_config = get_model_config("gemini-3.1-pro-preview-thinking-high")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.raw_continuation_text == "(openrouter gemini continuation)"
    assert result.mode == "prefill"
    assert result.usage is not None
    assert result.usage.reasoning_tokens == 3

    sent = client.calls[0]
    assert sent["model"] == "google/gemini-3.1-pro-preview"
    assert sent["max_tokens"] == 512
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": level_0},
    ]


def test_request_continuation_single_turn_openai_one_user_message():
    level_0 = get_turn_2_prompt(0)
    response = _make_openai_response("(single)")
    client = _FakeOpenAIClient(response)
    model_config = get_model_config("gpt-5.4-mini")

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=256,
        mode="single_turn",
    )

    assert result.mode == "single_turn"
    sent = client.responses.calls[0]
    msgs = sent["input"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "[T1U]" in msgs[0]["content"]
    assert SINGLE_TURN_WORK_HEADER in msgs[0]["content"]
    assert "[PREFILL]" in msgs[0]["content"]
    assert level_0 in msgs[0]["content"]


def test_request_continuation_rejects_unknown_mode():
    client = _FakeAnthropicClient(_make_anthropic_response("x"))
    with pytest.raises(ValueError, match="Unsupported continuation mode"):
        request_continuation(
            client=client,
            model_config=get_model_config("claude-haiku-4-5"),
            turn_1_user="[T1U]",
            turn_1_assistant_prefill="[PREFILL]",
            max_output_tokens=64,
            mode="bogus",  # type: ignore[arg-type]
        )
