"""AnthropicRawAdapter — mocked unit tests.

All Anthropic SDK calls are mocked via monkeypatch.setitem on sys.modules
so the tests run without network access or real API keys.

Verifies:
  * registered name is 'anthropic-raw'
  * setup refuses without ANTHROPIC_API_KEY
  * answer with oracle_context wraps prompt in <context> delimiters
  * answer without oracle_context sends the question alone
  * response is parsed into AdapterResult with non-zero cost + latency
  * API error path: error set, answer empty, no crash
  * pricing math: 1M input + 1M output → $3.00 + $15.00 = $18.00
  * teardown closes the client
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.anthropic_raw import (
    SONNET_INPUT_USD_PER_MTOK,
    SONNET_OUTPUT_USD_PER_MTOK,
    AnthropicRawAdapter,
    _build_prompt,
)

# ---------------------------------------------------------------------------
# Helpers — fake Anthropic module
# ---------------------------------------------------------------------------


def _make_fake_anthropic(
    answer_text: str = "test answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
    raise_on_create: Exception | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    content_block = SimpleNamespace(text=answer_text)

    class _FakeMessages:
        def __init__(self) -> None:
            self.last_call: dict[str, Any] = {}

        async def create(self, **kw: Any) -> Any:
            self.last_call = kw
            if raise_on_create:
                raise raise_on_create
            return SimpleNamespace(
                content=[content_block],
                usage=usage,
                model="claude-sonnet-4-5",
            )

    class _FakeAsyncAnthropic:
        def __init__(self, **_kw: Any) -> None:
            self.messages = _FakeMessages()
            self._closed = False

        async def close(self) -> None:
            self._closed = True

    return SimpleNamespace(AsyncAnthropic=_FakeAsyncAnthropic)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_adapter(monkeypatch):
    """Returns a ready-to-setup AnthropicRawAdapter with anthropic module mocked."""
    fake_module = _make_fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-ant-key")
    return AnthropicRawAdapter(), fake_module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_name_registered() -> None:
    """'anthropic-raw' must appear in the REGISTRY."""
    assert "anthropic-raw" in REGISTRY
    assert REGISTRY["anthropic-raw"] is AnthropicRawAdapter


def test_adapter_name_class_attr() -> None:
    assert AnthropicRawAdapter.name == "anthropic-raw"


async def test_setup_refuses_without_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    adapter = AnthropicRawAdapter()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await adapter.setup()


async def test_setup_creates_client(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()
    assert adapter._client is not None


async def test_answer_without_oracle_context(patched_adapter) -> None:
    adapter, _fake_module = patched_adapter
    await adapter.setup()

    result = await adapter.answer("What is 2 + 2?")

    assert result.error is None
    assert result.answer == "test answer"
    # Prompt sent to the API should be just the question.
    call_kw = adapter._client.messages.last_call  # type: ignore[union-attr]
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
    call_kw = adapter._client.messages.last_call  # type: ignore[union-attr]
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
    fake_module = _make_fake_anthropic(raise_on_create=RuntimeError("API down"))
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    adapter = AnthropicRawAdapter()
    await adapter.setup()
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "API down" in result.error
    assert result.latency_ms > 0


async def test_pricing_math() -> None:
    """1M input + 1M output must cost exactly $3.00 + $15.00 = $18.00."""
    fake_module = _make_fake_anthropic(input_tokens=1_000_000, output_tokens=1_000_000)
    import sys as _sys

    original = _sys.modules.get("anthropic")
    _sys.modules["anthropic"] = fake_module  # type: ignore[assignment]
    try:
        settings = get_settings()
        settings.__dict__["anthropic_api_key"] = "k"
        adapter = AnthropicRawAdapter()
        await adapter.setup()
        result = await adapter.answer("q")
    finally:
        if original is None:
            _sys.modules.pop("anthropic", None)
        else:
            _sys.modules["anthropic"] = original  # type: ignore[assignment]
        settings.__dict__["anthropic_api_key"] = ""

    expected = (
        1_000_000 * SONNET_INPUT_USD_PER_MTOK + 1_000_000 * SONNET_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    assert result.cost_usd == pytest.approx(expected)
    assert expected == pytest.approx(18.0)


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
    # Context comes before the question.
    assert prompt.index("<context>") < prompt.index("What?")
