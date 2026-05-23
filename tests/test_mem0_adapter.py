"""Mem0 BrainAdapter and AgentAdapter — mocked unit tests.

All mem0 and anthropic calls are mocked via monkeypatch.setitem on sys.modules
so the tests run without network access or real API keys.

Verifies:
  BrainAdapter:
    * name is registered correctly
    * setup refuses without MEM0_API_KEY
    * ingest calls client.add per doc with correct shape
    * search maps Mem0 result shape → RetrievedChunk (rank, score, doc_path)
    * reset rotates the user_id
    * search error wrapped as BrainSearchResult with empty chunks + error field
    * empty ingest is a no-op (no client.add calls)

  AgentAdapter:
    * name is registered correctly
    * setup refuses without MEM0_API_KEY
    * setup refuses without ANTHROPIC_API_KEY
    * answer with oracle_context calls mem0.add then mem0.search then Claude
    * answer without oracle_context skips the initial mem0.add
    * answer uses a fresh user_id per call
    * cost is computed from anthropic usage tokens
    * anthropic error is wrapped in AdapterResult with error set
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters.mem0 import Mem0AgentAdapter, Mem0BrainAdapter, _stable_id
from unison_evals.types import Document

# ---------------------------------------------------------------------------
# Helpers — fake mem0 and anthropic modules
# ---------------------------------------------------------------------------


class _FakeMemoryClient:
    def __init__(self, **_kw: Any) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self._search_results: list[dict[str, Any]] = []

    def add(self, messages: list[dict[str, str]], user_id: str, **kw: Any) -> None:
        self.add_calls.append({"messages": messages, "user_id": user_id, **kw})

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        filters: dict[str, Any] | None = None,
        version: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        effective_user_id = user_id or (filters or {}).get("user_id")
        self.search_calls.append(
            {"query": query, "user_id": effective_user_id, "limit": limit, "version": version}
        )
        return self._search_results[:]


def _make_fake_mem0(client: _FakeMemoryClient) -> SimpleNamespace:
    return SimpleNamespace(MemoryClient=lambda **_kw: client)


# Minimal anthropic message response shape.
def _make_fake_anthropic(
    answer_text: str = "test answer",
    input_tokens: int = 10,
    output_tokens: int = 5,
    raise_on_create: Exception | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    content_block = SimpleNamespace(text=answer_text)

    class _FakeMessages:
        async def create(self, **_kw: Any) -> Any:
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

        async def close(self) -> None:
            pass

    return SimpleNamespace(AsyncAnthropic=_FakeAsyncAnthropic)


# ---------------------------------------------------------------------------
# BrainAdapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mem0_client() -> _FakeMemoryClient:
    return _FakeMemoryClient()


@pytest.fixture
def patched_brain(monkeypatch, fake_mem0_client):
    """Returns a ready-to-setup Mem0BrainAdapter with mem0 module mocked."""
    monkeypatch.setitem(sys.modules, "mem0", _make_fake_mem0(fake_mem0_client))

    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "test-mem0-key")

    adapter = Mem0BrainAdapter()
    return adapter, fake_mem0_client


# ---------------------------------------------------------------------------
# BrainAdapter tests
# ---------------------------------------------------------------------------


def test_brain_adapter_name() -> None:
    """The adapter declares the canonical registry key 'mem0'."""
    assert Mem0BrainAdapter.name == "mem0"


async def test_brain_setup_refuses_without_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "")
    adapter = Mem0BrainAdapter()
    with pytest.raises(RuntimeError, match="MEM0_API_KEY"):
        await adapter.setup()


async def test_brain_setup_creates_client(patched_brain) -> None:
    adapter, client = patched_brain
    await adapter.setup()
    assert adapter._client is client


async def test_ingest_calls_add_per_doc(patched_brain) -> None:
    adapter, client = patched_brain
    await adapter.setup()
    docs = [
        Document(path="/a.md", body="apple facts"),
        Document(path="/b.md", body="banana facts"),
    ]
    await adapter.ingest(docs)
    assert len(client.add_calls) == 2
    # First call: message content matches first doc body.
    assert client.add_calls[0]["messages"] == [{"role": "user", "content": "apple facts"}]
    # Both calls use the same user_id.
    assert client.add_calls[0]["user_id"] == client.add_calls[1]["user_id"]
    assert client.add_calls[0]["user_id"] == adapter._current_user_id


async def test_ingest_passes_metadata(patched_brain) -> None:
    adapter, client = patched_brain
    await adapter.setup()
    doc = Document(path="/c.md", body="cherry", metadata={"source": "test"})
    await adapter.ingest([doc])
    assert client.add_calls[0]["metadata"] == {"source": "test"}


async def test_ingest_empty_is_noop(patched_brain) -> None:
    adapter, client = patched_brain
    await adapter.setup()
    await adapter.ingest([])
    assert client.add_calls == []


async def test_search_maps_results_to_chunks(patched_brain) -> None:
    adapter, client = patched_brain
    client._search_results = [
        {"id": "mem-001", "memory": "apple is a fruit", "score": 0.95},
        {"id": "mem-002", "memory": "banana is yellow", "score": 0.80},
    ]
    await adapter.setup()
    result = await adapter.search("fruit", k=5)

    assert result.error is None
    assert len(result.chunks) == 2
    # Rank is 1-indexed.
    assert result.chunks[0].rank == 1
    assert result.chunks[1].rank == 2
    # Score is preserved.
    assert result.chunks[0].score == pytest.approx(0.95)
    assert result.chunks[1].score == pytest.approx(0.80)
    # doc_path comes from the memory id.
    assert result.chunks[0].doc_path == "mem-001"
    assert result.chunks[1].doc_path == "mem-002"
    # chunk_text carries the memory text.
    assert result.chunks[0].chunk_text == "apple is a fruit"


async def test_search_uses_current_user_id(patched_brain) -> None:
    adapter, client = patched_brain
    client._search_results = []
    await adapter.setup()
    await adapter.search("query", k=3)
    assert client.search_calls[0]["user_id"] == adapter._current_user_id
    assert client.search_calls[0]["limit"] == 3


async def test_search_fallback_doc_path_when_no_id(patched_brain) -> None:
    adapter, client = patched_brain
    client._search_results = [
        {"id": "", "memory": "no id memory", "score": 0.5},
    ]
    await adapter.setup()
    result = await adapter.search("q")
    chunk = result.chunks[0]
    # Falls back to _stable_id based on memory text.
    assert chunk.doc_path == _stable_id("no id memory")
    assert chunk.doc_path.startswith("mem0-")


async def test_reset_rotates_user_id(patched_brain) -> None:
    adapter, _ = patched_brain
    await adapter.setup()
    first_id = adapter._current_user_id
    await adapter.reset()
    second_id = adapter._current_user_id
    assert first_id != second_id
    assert second_id.startswith("unison-evals-")


async def test_search_wraps_error(patched_brain, monkeypatch) -> None:
    adapter, client = patched_brain
    await adapter.setup()

    def boom(**_kw: Any) -> Any:
        raise RuntimeError("mem0 is down")

    monkeypatch.setattr(client, "search", boom)
    result = await adapter.search("anything")
    assert result.chunks == []
    assert result.error is not None
    assert "mem0 is down" in result.error
    assert result.latency_ms > 0


async def test_teardown_clears_client(patched_brain) -> None:
    adapter, _ = patched_brain
    await adapter.setup()
    assert adapter._client is not None
    await adapter.teardown()
    assert adapter._client is None


# ---------------------------------------------------------------------------
# AgentAdapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_agent(monkeypatch):
    """Returns a ready-to-setup Mem0AgentAdapter with mem0 + anthropic mocked."""
    fake_client = _FakeMemoryClient()
    monkeypatch.setitem(sys.modules, "mem0", _make_fake_mem0(fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", _make_fake_anthropic())

    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "test-mem0-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-ant-key")

    adapter = Mem0AgentAdapter()
    return adapter, fake_client


# ---------------------------------------------------------------------------
# AgentAdapter tests
# ---------------------------------------------------------------------------


def test_agent_adapter_name() -> None:
    """The adapter declares the canonical registry key 'mem0-agent'."""
    assert Mem0AgentAdapter.name == "mem0-agent"


async def test_agent_setup_refuses_without_mem0_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "ant-key")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(MemoryClient=object))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=object))
    adapter = Mem0AgentAdapter()
    with pytest.raises(RuntimeError, match="MEM0_API_KEY"):
        await adapter.setup()


async def test_agent_setup_refuses_without_anthropic_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "mem0-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(MemoryClient=object))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=object))
    adapter = Mem0AgentAdapter()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await adapter.setup()


async def test_agent_answer_with_oracle_context(patched_agent) -> None:
    adapter, client = patched_agent
    client._search_results = [
        {"id": "m1", "memory": "Paris is the capital of France", "score": 0.9}
    ]
    await adapter.setup()

    result = await adapter.answer(
        "What is the capital of France?", oracle_context="Paris is the capital of France."
    )

    # oracle_context triggers an initial add call.
    assert len(client.add_calls) == 1
    assert client.add_calls[0]["messages"][0]["content"] == "Paris is the capital of France."
    # Then a search was done.
    assert len(client.search_calls) == 1
    # The answer is populated.
    assert result.answer == "test answer"
    assert result.error is None
    assert result.cost_usd > 0


async def test_agent_answer_without_oracle_context(patched_agent) -> None:
    adapter, client = patched_agent
    client._search_results = []
    await adapter.setup()

    result = await adapter.answer("Some question")

    # No oracle_context → no initial add call.
    assert client.add_calls == []
    assert len(client.search_calls) == 1
    assert result.answer == "test answer"
    assert result.error is None


async def test_agent_fresh_user_id_per_answer(patched_agent) -> None:
    adapter, client = patched_agent
    client._search_results = []
    await adapter.setup()

    await adapter.answer("Q1")
    await adapter.answer("Q2")

    # Each answer produces one search call; user_ids must differ.
    ids = [c["user_id"] for c in client.search_calls]
    assert len(ids) == 2
    assert ids[0] != ids[1]


async def test_agent_cost_computed_from_tokens(monkeypatch) -> None:
    """Cost must equal (10 * 3 + 5 * 15) / 1_000_000 = 0.000105."""
    fake_client = _FakeMemoryClient()
    monkeypatch.setitem(sys.modules, "mem0", _make_fake_mem0(fake_client))
    monkeypatch.setitem(
        sys.modules, "anthropic", _make_fake_anthropic(input_tokens=10, output_tokens=5)
    )
    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "k")
    monkeypatch.setattr(settings, "anthropic_api_key", "k")

    adapter = Mem0AgentAdapter()
    await adapter.setup()
    result = await adapter.answer("q")

    expected = (10 * 3.0 + 5 * 15.0) / 1_000_000.0
    assert result.cost_usd == pytest.approx(expected)


async def test_agent_anthropic_error_wrapped(monkeypatch) -> None:
    fake_client = _FakeMemoryClient()
    monkeypatch.setitem(sys.modules, "mem0", _make_fake_mem0(fake_client))
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        _make_fake_anthropic(raise_on_create=RuntimeError("anthropic outage")),
    )
    settings = get_settings()
    monkeypatch.setattr(settings, "mem0_api_key", "k")
    monkeypatch.setattr(settings, "anthropic_api_key", "k")

    adapter = Mem0AgentAdapter()
    await adapter.setup()
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "anthropic outage" in result.error
    assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# _stable_id helper
# ---------------------------------------------------------------------------


def test_stable_id_deterministic() -> None:
    assert _stable_id("hello") == _stable_id("hello")


def test_stable_id_different_for_different_text() -> None:
    assert _stable_id("foo") != _stable_id("bar")


def test_stable_id_prefix() -> None:
    assert _stable_id("any text").startswith("mem0-")
