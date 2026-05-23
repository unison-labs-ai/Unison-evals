"""OpenAIGpt5Adapter — mocked unit tests.

All OpenAI SDK calls are mocked via monkeypatch.setitem on sys.modules
so the tests run without network access or real API keys.

Verifies:
  * registered name is 'openai-gpt5'
  * setup refuses without OPENAI_API_KEY
  * answer with oracle_context wraps prompt in <context> delimiters
  * answer without oracle_context sends the question alone
  * response is parsed into AdapterResult with non-zero cost + latency
  * API error path: error set, answer empty, no crash
  * pricing math: 1M input + 1M output → $5.00 + $15.00 = $20.00
  * teardown closes the client
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.openai_gpt5 import (
    GPT5_INPUT_USD_PER_MTOK,
    GPT5_OUTPUT_USD_PER_MTOK,
    OpenAIGpt5Adapter,
    _build_prompt,
)

# ---------------------------------------------------------------------------
# Helpers — fake OpenAI module
# ---------------------------------------------------------------------------


def _make_fake_openai(
    answer_text: str = "test answer",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    raise_on_create: Exception | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    message = SimpleNamespace(content=answer_text)
    choice = SimpleNamespace(message=message)

    class _FakeCompletions:
        def __init__(self) -> None:
            self.last_call: dict[str, Any] = {}

        async def create(self, **kw: Any) -> Any:
            self.last_call = kw
            if raise_on_create:
                raise raise_on_create
            return SimpleNamespace(
                choices=[choice],
                usage=usage,
                model="gpt-5",
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, **_kw: Any) -> None:
            self.chat = _FakeChat()
            self._closed = False

        async def close(self) -> None:
            self._closed = True

    return SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_adapter(monkeypatch):
    """Returns a ready-to-setup OpenAIGpt5Adapter with openai module mocked."""
    fake_module = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "test-oai-key")
    return OpenAIGpt5Adapter(), fake_module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_name_registered() -> None:
    """'openai-gpt5' must appear in the REGISTRY."""
    assert "openai-gpt5" in REGISTRY
    assert REGISTRY["openai-gpt5"] is OpenAIGpt5Adapter


def test_adapter_name_class_attr() -> None:
    assert OpenAIGpt5Adapter.name == "openai-gpt5"


async def test_setup_refuses_without_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "")
    adapter = OpenAIGpt5Adapter()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await adapter.setup()


async def test_setup_creates_client(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()
    assert adapter._client is not None


async def test_answer_without_oracle_context(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()

    result = await adapter.answer("What is 2 + 2?")

    assert result.error is None
    assert result.answer == "test answer"
    call_kw = adapter._client.chat.completions.last_call  # type: ignore[union-attr]
    prompt = call_kw["messages"][0]["content"]
    assert prompt == "What is 2 + 2?"
    assert "<context>" not in prompt
    assert result.raw["has_oracle_context"] is False


async def test_answer_with_oracle_context(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()

    result = await adapter.answer(
        question="What is the capital of France?",
        oracle_context="Paris is the capital of France.",
    )

    assert result.error is None
    assert result.answer == "test answer"
    call_kw = adapter._client.chat.completions.last_call  # type: ignore[union-attr]
    prompt = call_kw["messages"][0]["content"]
    assert "<context>" in prompt
    assert "Paris is the capital of France." in prompt
    assert "What is the capital of France?" in prompt
    assert result.raw["has_oracle_context"] is True


async def test_answer_cost_and_latency(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()

    result = await adapter.answer("q")

    assert result.cost_usd > 0
    assert result.latency_ms > 0


async def test_api_error_sets_error_field(monkeypatch) -> None:
    fake_module = _make_fake_openai(raise_on_create=RuntimeError("OpenAI down"))
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    adapter = OpenAIGpt5Adapter()
    await adapter.setup()
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "OpenAI down" in result.error
    assert result.latency_ms > 0


async def test_pricing_math(monkeypatch) -> None:
    """1M prompt + 1M completion must cost $5.00 + $15.00 = $20.00."""
    fake_module = _make_fake_openai(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "k")

    adapter = OpenAIGpt5Adapter()
    await adapter.setup()
    result = await adapter.answer("q")

    expected = (
        1_000_000 * GPT5_INPUT_USD_PER_MTOK + 1_000_000 * GPT5_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    assert result.cost_usd == pytest.approx(expected)
    assert expected == pytest.approx(20.0)


async def test_teardown_closes_client(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()
    client_ref = adapter._client
    await adapter.teardown()
    assert adapter._client is None
    assert client_ref._closed is True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _build_prompt helper
# ---------------------------------------------------------------------------


def test_build_prompt_no_context() -> None:
    assert _build_prompt("hello?", None) == "hello?"


def test_build_prompt_with_context() -> None:
    prompt = _build_prompt("What?", "Some context.")
    assert "<context>" in prompt
    assert "Some context." in prompt
    assert "What?" in prompt
    assert prompt.index("<context>") < prompt.index("What?")
