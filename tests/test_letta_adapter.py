"""Letta BrainAdapter — mocked unit tests.

All network calls are intercepted by patching letta_client into sys.modules
with a fake AsyncLetta whose agents.passages.create / .search / .delete mirror
the real SDK shape (as of letta-client 1.10.3).

Tests verify:
  * registered name correctness
  * setup refuses without LETTA_API_KEY
  * reset creates a fresh agent (and deletes the prior one)
  * ingest calls passages.create per doc with the '[path] body' encoding
  * ingest skips empty doc lists
  * search maps PassageSearchResponse.results to RetrievedChunk (rank/score)
  * search wraps unexpected exceptions as BrainSearchResult with error set
  * teardown deletes the current agent and closes the http client
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters.letta import LettaBrainAdapter, _split_tagged_content
from unison_evals.types import Document

# ---------------------------------------------------------------------------
# Fake SDK helpers
# ---------------------------------------------------------------------------


class _FakePassages:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self._search_results: list[Any] = []
        self._search_raises: Exception | None = None

    async def create(self, agent_id: str, *, text: str, **_kw: Any) -> list[Any]:
        self.create_calls.append({"agent_id": agent_id, "text": text})
        return []

    async def search(self, agent_id: str, *, query: str, top_k: int, **_kw: Any) -> Any:
        self.search_calls.append({"agent_id": agent_id, "query": query, "top_k": top_k})
        if self._search_raises is not None:
            raise self._search_raises
        results = self._search_results[:top_k]
        return SimpleNamespace(count=len(results), results=results)


class _FakeAgentsResource:
    def __init__(self) -> None:
        self.passages = _FakePassages()
        self.created_agents: list[dict[str, Any]] = []
        self.deleted_agents: list[str] = []
        self._next_agent_id: int = 1

    async def create(self, **kwargs: Any) -> Any:
        agent_id = f"agent-fake-{self._next_agent_id:04d}"
        self._next_agent_id += 1
        self.created_agents.append({"id": agent_id, **kwargs})
        return SimpleNamespace(id=agent_id)

    async def delete(self, agent_id: str, **_kw: Any) -> None:
        self.deleted_agents.append(agent_id)


class _FakeHttpxClient:
    async def aclose(self) -> None:
        pass


class _FakeAsyncLetta:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.agents = _FakeAgentsResource()
        self._client = _FakeHttpxClient()


def _make_result(path: str, body: str) -> Any:
    """Build a fake Passage result with the '[path] body' encoding."""
    content = f"[{path}] {body}"
    return SimpleNamespace(
        id=f"passage-{path}",
        content=content,
        timestamp="2026-01-01T00:00:00Z",
        tags=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_adapter(monkeypatch: pytest.MonkeyPatch):
    """Injects a fake letta_client module and returns (adapter, fake_client)."""
    fake_async_letta_cls = _FakeAsyncLetta
    fake_module = SimpleNamespace(AsyncLetta=fake_async_letta_cls)

    monkeypatch.setitem(sys.modules, "letta_client", fake_module)

    settings = get_settings()
    monkeypatch.setattr(settings, "letta_api_key", "test-letta-key")
    monkeypatch.setattr(settings, "letta_base_url", "")

    adapter = LettaBrainAdapter()
    return adapter


# ---------------------------------------------------------------------------
# Tests: registry
# ---------------------------------------------------------------------------


def test_registered_name() -> None:
    assert LettaBrainAdapter.name == "letta"


# ---------------------------------------------------------------------------
# Tests: setup
# ---------------------------------------------------------------------------


async def test_setup_refuses_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncLetta=_FakeAsyncLetta)
    monkeypatch.setitem(sys.modules, "letta_client", fake_module)

    settings = get_settings()
    monkeypatch.setattr(settings, "letta_api_key", "")
    adapter = LettaBrainAdapter()
    with pytest.raises(RuntimeError, match="LETTA_API_KEY"):
        await adapter.setup()


async def test_setup_instantiates_client(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    assert patched_adapter._client is not None
    assert patched_adapter._client.init_kwargs["api_key"] == "test-letta-key"


async def test_setup_passes_base_url_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncLetta=_FakeAsyncLetta)
    monkeypatch.setitem(sys.modules, "letta_client", fake_module)
    settings = get_settings()
    monkeypatch.setattr(settings, "letta_api_key", "key")
    monkeypatch.setattr(settings, "letta_base_url", "http://localhost:8283")
    adapter = LettaBrainAdapter()
    await adapter.setup()
    assert adapter._client.init_kwargs.get("base_url") == "http://localhost:8283"


# ---------------------------------------------------------------------------
# Tests: reset (create / delete agent lifecycle)
# ---------------------------------------------------------------------------


async def test_reset_creates_fresh_agent(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    fake_agents: _FakeAgentsResource = patched_adapter._client.agents

    await patched_adapter.reset()
    assert len(fake_agents.created_agents) == 1
    assert patched_adapter._agent_id == "agent-fake-0001"


async def test_reset_twice_deletes_prior_agent(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    fake_agents: _FakeAgentsResource = patched_adapter._client.agents

    await patched_adapter.reset()
    first_id = patched_adapter._agent_id

    await patched_adapter.reset()
    assert first_id in fake_agents.deleted_agents
    assert len(fake_agents.created_agents) == 2
    assert patched_adapter._agent_id == "agent-fake-0002"


# ---------------------------------------------------------------------------
# Tests: ingest
# ---------------------------------------------------------------------------


async def test_ingest_calls_passages_create_per_doc(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages

    docs = [
        Document(path="/a.md", body="apple facts"),
        Document(path="/b.md", body="banana facts"),
    ]
    await patched_adapter.ingest(docs)

    assert len(fake_passages.create_calls) == 2
    texts = [c["text"] for c in fake_passages.create_calls]
    assert "[/a.md] apple facts" in texts
    assert "[/b.md] banana facts" in texts


async def test_ingest_empty_is_noop(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages

    await patched_adapter.ingest([])
    assert fake_passages.create_calls == []


async def test_ingest_uses_correct_agent_id(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages

    await patched_adapter.ingest([Document(path="/x.md", body="x")])
    assert all(c["agent_id"] == patched_adapter._agent_id for c in fake_passages.create_calls)


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------


async def test_search_maps_results_to_chunks(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages
    fake_passages._search_results = [
        _make_result("/a.md", "apple facts"),
        _make_result("/b.md", "banana facts"),
    ]

    result = await patched_adapter.search("fruit", k=2)

    assert result.error is None
    assert len(result.chunks) == 2
    assert result.chunks[0].rank == 1
    assert result.chunks[1].rank == 2
    assert result.chunks[0].doc_path == "/a.md"
    assert result.chunks[1].doc_path == "/b.md"
    assert result.chunks[0].chunk_text == "apple facts"


async def test_search_score_descends_by_rank(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages
    fake_passages._search_results = [_make_result(f"/{i}.md", f"body {i}") for i in range(5)]

    result = await patched_adapter.search("query", k=5)
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


async def test_search_respects_k_limit(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages
    fake_passages._search_results = [_make_result(f"/{i}.md", f"body {i}") for i in range(10)]

    result = await patched_adapter.search("query", k=3)
    assert len(result.chunks) == 3
    assert fake_passages.search_calls[-1]["top_k"] == 3


async def test_search_wraps_errors(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_passages: _FakePassages = patched_adapter._client.agents.passages
    fake_passages._search_raises = RuntimeError("letta cloud outage")

    result = await patched_adapter.search("anything", k=5)
    assert result.chunks == []
    assert result.error is not None
    assert "outage" in result.error
    assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# Tests: teardown
# ---------------------------------------------------------------------------


async def test_teardown_deletes_agent(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    await patched_adapter.reset()
    fake_agents: _FakeAgentsResource = patched_adapter._client.agents
    agent_id = patched_adapter._agent_id

    await patched_adapter.teardown()
    assert agent_id in fake_agents.deleted_agents
    assert patched_adapter._agent_id is None
    assert patched_adapter._client is None


async def test_teardown_idempotent_without_agent(patched_adapter: LettaBrainAdapter) -> None:
    await patched_adapter.setup()
    # Never called reset() — no agent_id set. Should not raise.
    await patched_adapter.teardown()
    assert patched_adapter._client is None


# ---------------------------------------------------------------------------
# Tests: _split_tagged_content helper
# ---------------------------------------------------------------------------


def test_split_tagged_content_round_trip() -> None:
    path, text = _split_tagged_content("[/some/path.md] The body text here")
    assert path == "/some/path.md"
    assert text == "The body text here"


def test_split_tagged_content_no_bracket_falls_back() -> None:
    raw = "no bracket prefix at all"
    path, text = _split_tagged_content(raw)
    assert path == raw
    assert text == raw
