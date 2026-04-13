from dataclasses import replace
from types import SimpleNamespace

import pytest

from emoji_bench.continuation_formatter import (
    SINGLE_TURN_WORK_HEADER,
    TURN_2_USER,
    format_continuation_single_turn,
)
from emoji_bench.continuation_provider import (
    ContinuationResponse,
    request_continuation,
)
from emoji_bench.model_registry import get_model_config


# --- Fakes -----------------------------------------------------------------


class _FakeMessagesAPI:
    def __init__(self, response: object):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **options):
        self.calls.append(options)
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


# --- format_continuation_single_turn --------------------------------------


def test_single_turn_format_contains_turn_1_then_work_header_then_prefill():
    rendered = format_continuation_single_turn(
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="Start: x\nStep 1: x = y    [by ⊕ table]",
    )
    # Order is: T1U, blank line, header, blank line, prefill.
    assert rendered.startswith("[T1U]\n\n")
    assert SINGLE_TURN_WORK_HEADER in rendered
    assert rendered.endswith("Start: x\nStep 1: x = y    [by ⊕ table]")
    # No "Please continue." literal in the single-turn rendering — the WORK
    # SO FAR header carries the same instruction without duplicating it.
    assert TURN_2_USER not in rendered


# --- Anthropic native prefill ---------------------------------------------


def test_request_continuation_prefill_anthropic_uses_two_message_native_prefill():
    response = _make_anthropic_response(" continuing the work...")
    client = _FakeAnthropicClient(response)
    model_config = get_model_config("claude-haiku-4-5")
    assert model_config.supports_assistant_prefill is True

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
    assert result.used_native_prefill is True
    assert result.raw_continuation_text == " continuing the work..."

    assert len(client.messages.calls) == 1
    sent = client.messages.calls[0]
    assert sent["model"] == model_config.api_model
    assert sent["max_tokens"] == 512
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
    ]
    # No system prompt, no JSON schema / output_config — raw text only.
    assert "system" not in sent
    assert "output_config" not in sent


def test_request_continuation_single_turn_anthropic_sends_one_user_message():
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
    assert result.used_native_prefill is False

    sent = client.messages.calls[0]
    messages = sent["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "[T1U]" in messages[0]["content"]
    assert SINGLE_TURN_WORK_HEADER in messages[0]["content"]
    assert "[PREFILL]" in messages[0]["content"]


# --- Non-Anthropic providers in prefill mode (3-message list) -------------


def test_request_continuation_prefill_openai_sends_three_message_conversation():
    response = _make_openai_response("(continuation)")
    client = _FakeOpenAIClient(response)
    model_config = get_model_config("gpt-5.4-mini")
    assert model_config.supports_assistant_prefill is False

    result = request_continuation(
        client=client,
        model_config=model_config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=512,
        mode="prefill",
    )

    assert result.mode == "prefill"
    assert result.used_native_prefill is False
    assert result.raw_continuation_text == "(continuation)"

    sent = client.responses.calls[0]
    assert sent["max_output_tokens"] == 512
    assert sent["input"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": TURN_2_USER},
    ]
    # Reasoning passed through from the registry config.
    assert sent.get("reasoning") == {"effort": "medium"}


def test_request_continuation_prefill_mistral_sends_three_message_conversation():
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
    assert result.used_native_prefill is False

    sent = client.calls[0]
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": TURN_2_USER},
    ]
    # No JSON response_format — raw text only.
    assert "response_format" not in sent


def test_request_continuation_prefill_gemini_uses_model_role_for_prefill():
    client = _FakeGeminiClient(
        {
            "responseId": "gem_id",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "(gemini continuation)"}],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 6,
                "totalTokenCount": 11,
            },
        }
    )
    model_config = get_model_config("gemini-3-flash-preview")

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
    assert result.used_native_prefill is False

    sent = client.calls[0]["options"]
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "[T1U]"}]},
        # Gemini convention: the assistant role is named "model".
        {"role": "model", "parts": [{"text": "[PREFILL]"}]},
        {"role": "user", "parts": [{"text": TURN_2_USER}]},
    ]
    # No responseJsonSchema, no responseMimeType=application/json — raw text.
    gen_cfg = sent["generationConfig"]
    assert "responseJsonSchema" not in gen_cfg
    assert "responseMimeType" not in gen_cfg


# --- Single-turn dispatch parity across providers --------------------------


def test_request_continuation_single_turn_openai_one_user_message():
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


# --- Unknown mode handling -------------------------------------------------


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


# --- Capability flag fallback ---------------------------------------------


def test_anthropic_with_supports_assistant_prefill_false_falls_back_to_three_messages():
    """A future Anthropic config with the flag flipped off must not crash."""
    response = _make_anthropic_response(" continuing")
    client = _FakeAnthropicClient(response)
    base = get_model_config("claude-haiku-4-5")
    # Frozen dataclass — replace flips the flag without mutating the registry.
    config = replace(base, supports_assistant_prefill=False)

    result = request_continuation(
        client=client,
        model_config=config,
        turn_1_user="[T1U]",
        turn_1_assistant_prefill="[PREFILL]",
        max_output_tokens=128,
        mode="prefill",
    )

    assert result.used_native_prefill is False
    sent = client.messages.calls[0]
    assert sent["messages"] == [
        {"role": "user", "content": "[T1U]"},
        {"role": "assistant", "content": "[PREFILL]"},
        {"role": "user", "content": TURN_2_USER},
    ]


# --- All claude-* models advertise the prefill capability -----------------


def test_supports_assistant_prefill_matches_empirical_anthropic_capability():
    """Per-model truth from the Anthropic API as of 2026-04-13.

    Both Sonnet 4.6 entries (the reasoning variant shares ``api_model``)
    return a 400 for assistant-message prefill: "This model does not
    support assistant message prefill. The conversation must end with a
    user message." Haiku 4.5 accepts prefill normally. The flag is what
    the dispatcher reads, so it must reflect reality, not aspiration.
    """
    from emoji_bench.model_registry import MODEL_CONFIGS
    expected = {
        "claude-haiku-4-5": True,
        "claude-sonnet-4-6": False,
        "claude-sonnet-4-6-reasoning": False,
    }
    for key, want in expected.items():
        got = MODEL_CONFIGS[key].supports_assistant_prefill
        assert got is want, f"{key}: expected supports_assistant_prefill={want}, got {got}"


def test_non_anthropic_configs_default_to_no_prefill():
    from emoji_bench.model_registry import MODEL_CONFIGS
    others = [c for c in MODEL_CONFIGS.values() if c.provider != "anthropic"]
    for config in others:
        assert config.supports_assistant_prefill is False, (
            f"{config.key} should default to supports_assistant_prefill=False"
        )
