"""GoogleGeminiAdapter — mocked unit tests.

All google-genai SDK calls are mocked via monkeypatch.setitem on sys.modules
so the tests run without network access or real API keys.

Verifies:
  * registered name is 'google-gemini'
  * setup refuses without GOOGLE_API_KEY
  * answer with oracle_context wraps prompt in <context> delimiters
  * answer without oracle_context sends the question alone
  * response is parsed into AdapterResult with non-zero cost + latency
  * API error path: error set, answer empty, no crash
  * pricing math: 1M input + 1M output → $1.25 + $10.00 = $11.25
  * teardown clears the client reference
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.google_gemini import (
    GEMINI_INPUT_USD_PER_MTOK,
    GEMINI_OUTPUT_USD_PER_MTOK,
    GoogleGeminiAdapter,
    _build_prompt,
)

# ---------------------------------------------------------------------------
# Helpers — fake google-genai module
# ---------------------------------------------------------------------------


def _make_fake_genai(
    answer_text: str = "test answer",
    prompt_token_count: int = 100,
    candidates_token_count: int = 50,
    raise_on_generate: Exception | None = None,
) -> SimpleNamespace:
    """Build a minimal fake of the google.genai module tree."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
    )

    class _FakeModels:
        def __init__(self) -> None:
            self.last_call: dict[str, Any] = {}

        async def generate_content(self, model: str, contents: str, **kw: Any) -> Any:
            self.last_call = {"model": model, "contents": contents, **kw}
            if raise_on_generate:
                raise raise_on_generate
            return SimpleNamespace(text=answer_text, usage_metadata=usage)

    class _FakeAio:
        def __init__(self) -> None:
            self.models = _FakeModels()

    class _FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.aio = _FakeAio()

    # google.genai.types is imported inside the method; stub it to avoid ImportError.
    fake_types = SimpleNamespace(GenerateContentConfig=lambda **_kw: SimpleNamespace())

    return SimpleNamespace(Client=_FakeClient, types=fake_types)


def _make_google_namespace(fake_genai: SimpleNamespace) -> SimpleNamespace:
    """Return a fake 'google' top-level namespace with genai inside."""
    return SimpleNamespace(genai=fake_genai)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_adapter(monkeypatch):
    """Returns a ready-to-setup GoogleGeminiAdapter with google-genai mocked."""
    fake_genai = _make_fake_genai()
    fake_google = _make_google_namespace(fake_genai)
    # The adapter does `from google import genai` — needs google + google.genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai.types)

    settings = get_settings()
    monkeypatch.setattr(settings, "google_api_key", "test-google-key")

    return GoogleGeminiAdapter(), fake_genai


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_name_registered() -> None:
    """'google-gemini' must appear in the REGISTRY."""
    assert "google-gemini" in REGISTRY
    assert REGISTRY["google-gemini"] is GoogleGeminiAdapter


def test_adapter_name_class_attr() -> None:
    assert GoogleGeminiAdapter.name == "google-gemini"


async def test_setup_refuses_without_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "google_api_key", "")
    adapter = GoogleGeminiAdapter()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        await adapter.setup()


async def test_setup_creates_client(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()
    assert adapter._client is not None


async def test_answer_without_oracle_context(patched_adapter) -> None:
    adapter, _fake_genai = patched_adapter
    await adapter.setup()

    result = await adapter.answer("What is 2 + 2?")

    assert result.error is None
    assert result.answer == "test answer"
    call = adapter._client.aio.models.last_call  # type: ignore[union-attr]
    assert call["contents"] == "What is 2 + 2?"
    assert "<context>" not in call["contents"]
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
    call = adapter._client.aio.models.last_call  # type: ignore[union-attr]
    prompt = call["contents"]
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
    fake_genai = _make_fake_genai(raise_on_generate=RuntimeError("Gemini down"))
    fake_google = _make_google_namespace(fake_genai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai.types)
    settings = get_settings()
    monkeypatch.setattr(settings, "google_api_key", "test-key")

    adapter = GoogleGeminiAdapter()
    await adapter.setup()
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "Gemini down" in result.error
    assert result.latency_ms > 0


async def test_pricing_math(monkeypatch) -> None:
    """1M prompt + 1M candidates must cost $1.25 + $10.00 = $11.25."""
    fake_genai = _make_fake_genai(prompt_token_count=1_000_000, candidates_token_count=1_000_000)
    fake_google = _make_google_namespace(fake_genai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai.types)
    settings = get_settings()
    monkeypatch.setattr(settings, "google_api_key", "k")

    adapter = GoogleGeminiAdapter()
    await adapter.setup()
    result = await adapter.answer("q")

    expected = (
        1_000_000 * GEMINI_INPUT_USD_PER_MTOK + 1_000_000 * GEMINI_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    assert result.cost_usd == pytest.approx(expected)
    assert expected == pytest.approx(11.25)


async def test_teardown_clears_client(patched_adapter) -> None:
    adapter, _ = patched_adapter
    await adapter.setup()
    assert adapter._client is not None
    await adapter.teardown()
    assert adapter._client is None


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
