"""Zep BrainAdapter — mocked unit tests.

All zep_cloud calls are mocked via monkeypatch.setitem on sys.modules so
the tests run without network access or real API keys.

Verifies:
  * name is registered correctly
  * setup refuses without ZEP_API_KEY
  * setup accepts zep_base_url override and passes it to the client
  * ingest calls graph.add per doc with correct user_id and type="text"
  * empty ingest is a no-op (no graph.add calls, no sleep)
  * search maps Zep Episode result shape → RetrievedChunk (rank, score, doc_path)
  * search falls back to _stable_id when episode uuid is missing
  * search passes user_id, limit, scope="episodes" to graph.search
  * reset rotates user_id to a new "unison-evals-" prefixed value
  * search error is wrapped in BrainSearchResult with empty chunks + error field
  * teardown clears the client reference
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters.zep import ZepBrainAdapter, _stable_id
from unison_evals.types import Document

# ---------------------------------------------------------------------------
# Helpers — fake zep_cloud module
# ---------------------------------------------------------------------------


class _FakeEpisode:
    """Minimal Episode shape: content, score, uuid_."""

    def __init__(self, content: str, score: float | None, uuid_: str) -> None:
        self.content = content
        self.score = score
        self.uuid_ = uuid_


class _FakeGraphSearchResults:
    def __init__(self, episodes: list[_FakeEpisode]) -> None:
        self.episodes = episodes
        self.edges = None
        self.nodes = None
        self.observations = None
        self.thread_summaries = None


class _FakeGraphClient:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self._search_results: list[_FakeEpisode] = []

    def add(self, *, user_id: str, type: str, data: str, **_kw: Any) -> None:
        self.add_calls.append({"user_id": user_id, "type": type, "data": data})

    def search(
        self, *, user_id: str, query: str, limit: int, scope: str, **_kw: Any
    ) -> _FakeGraphSearchResults:
        self.search_calls.append(
            {"user_id": user_id, "query": query, "limit": limit, "scope": scope}
        )
        return _FakeGraphSearchResults(episodes=self._search_results[:])


class _FakeZepClient:
    def __init__(self, **kwargs: Any) -> None:
        self._init_kwargs = kwargs
        self.graph = _FakeGraphClient()


def _make_fake_zep_cloud(client_instance: _FakeZepClient) -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(Zep=lambda **kw: client_instance),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_zep_client() -> _FakeZepClient:
    return _FakeZepClient()


@pytest.fixture
def patched_brain(monkeypatch, fake_zep_client):
    """Returns a ready-to-setup ZepBrainAdapter with zep_cloud mocked."""
    # Patch sys.modules so `from zep_cloud.client import Zep` resolves to our fake.
    fake_module = SimpleNamespace(Zep=lambda **kw: fake_zep_client)
    monkeypatch.setitem(sys.modules, "zep_cloud", SimpleNamespace(client=fake_module))
    monkeypatch.setitem(sys.modules, "zep_cloud.client", fake_module)

    settings = get_settings()
    monkeypatch.setattr(settings, "zep_api_key", "test-zep-key")
    monkeypatch.setattr(settings, "zep_base_url", "")
    monkeypatch.setattr(settings, "zep_ingest_wait_seconds", 0.0)

    adapter = ZepBrainAdapter()
    return adapter, fake_zep_client


# ---------------------------------------------------------------------------
# Tests — registration and name
# ---------------------------------------------------------------------------


def test_brain_adapter_name() -> None:
    """The adapter declares the canonical registry key 'zep'."""
    assert ZepBrainAdapter.name == "zep"


def test_brain_adapter_registered() -> None:
    """ZepBrainAdapter is in BRAIN_REGISTRY under 'zep'."""
    from unison_evals.memory_evals.adapters import BRAIN_REGISTRY

    assert "zep" in BRAIN_REGISTRY
    assert BRAIN_REGISTRY["zep"] is ZepBrainAdapter


# ---------------------------------------------------------------------------
# Tests — setup
# ---------------------------------------------------------------------------


async def test_setup_refuses_without_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "zep_api_key", "")
    adapter = ZepBrainAdapter()
    with pytest.raises(RuntimeError, match="ZEP_API_KEY"):
        await adapter.setup()


async def test_setup_creates_client(patched_brain) -> None:
    adapter, _fake_client = patched_brain
    await adapter.setup()
    assert adapter._client is not None


async def test_setup_passes_base_url(monkeypatch, fake_zep_client) -> None:
    """When zep_base_url is set, it should be forwarded to the Zep constructor."""
    captured_kwargs: dict[str, Any] = {}

    def capturing_zep(**kw: Any) -> _FakeZepClient:
        captured_kwargs.update(kw)
        return fake_zep_client

    fake_module = SimpleNamespace(Zep=capturing_zep)
    monkeypatch.setitem(sys.modules, "zep_cloud", SimpleNamespace(client=fake_module))
    monkeypatch.setitem(sys.modules, "zep_cloud.client", fake_module)

    settings = get_settings()
    monkeypatch.setattr(settings, "zep_api_key", "key")
    monkeypatch.setattr(settings, "zep_base_url", "https://my.zep.server")
    monkeypatch.setattr(settings, "zep_ingest_wait_seconds", 0.0)

    adapter = ZepBrainAdapter()
    await adapter.setup()

    assert captured_kwargs.get("base_url") == "https://my.zep.server"


# ---------------------------------------------------------------------------
# Tests — ingest
# ---------------------------------------------------------------------------


async def test_ingest_calls_graph_add_per_doc(patched_brain) -> None:
    adapter, fake_client = patched_brain
    await adapter.setup()
    docs = [
        Document(path="/a.md", body="apple facts"),
        Document(path="/b.md", body="banana facts"),
    ]
    await adapter.ingest(docs)

    assert len(fake_client.graph.add_calls) == 2
    assert fake_client.graph.add_calls[0]["data"] == "apple facts"
    assert fake_client.graph.add_calls[1]["data"] == "banana facts"


async def test_ingest_uses_correct_user_id_and_type(patched_brain) -> None:
    adapter, fake_client = patched_brain
    await adapter.setup()
    await adapter.ingest([Document(path="/x.md", body="content")])

    call = fake_client.graph.add_calls[0]
    assert call["user_id"] == adapter._current_user_id
    assert call["type"] == "text"


async def test_ingest_empty_is_noop(patched_brain) -> None:
    adapter, fake_client = patched_brain
    await adapter.setup()
    await adapter.ingest([])
    assert fake_client.graph.add_calls == []


# ---------------------------------------------------------------------------
# Tests — search
# ---------------------------------------------------------------------------


async def test_search_maps_episodes_to_chunks(patched_brain) -> None:
    adapter, fake_client = patched_brain
    fake_client.graph._search_results = [
        _FakeEpisode(content="apple is a fruit", score=0.95, uuid_="ep-001"),
        _FakeEpisode(content="banana is yellow", score=0.80, uuid_="ep-002"),
    ]
    await adapter.setup()
    result = await adapter.search("fruit", k=5)

    assert result.error is None
    assert len(result.chunks) == 2
    # Rank is 1-indexed.
    assert result.chunks[0].rank == 1
    assert result.chunks[1].rank == 2
    # Score preserved.
    assert result.chunks[0].score == pytest.approx(0.95)
    assert result.chunks[1].score == pytest.approx(0.80)
    # doc_path comes from episode uuid.
    assert result.chunks[0].doc_path == "ep-001"
    assert result.chunks[1].doc_path == "ep-002"
    # chunk_text carries the episode content.
    assert result.chunks[0].chunk_text == "apple is a fruit"


async def test_search_passes_correct_args(patched_brain) -> None:
    adapter, fake_client = patched_brain
    fake_client.graph._search_results = []
    await adapter.setup()
    await adapter.search("my query", k=7)

    assert len(fake_client.graph.search_calls) == 1
    call = fake_client.graph.search_calls[0]
    assert call["user_id"] == adapter._current_user_id
    assert call["query"] == "my query"
    assert call["limit"] == 7
    assert call["scope"] == "episodes"


async def test_search_fallback_doc_path_when_no_uuid(patched_brain) -> None:
    adapter, fake_client = patched_brain
    fake_client.graph._search_results = [
        _FakeEpisode(content="episode without uuid", score=0.5, uuid_=""),
    ]
    await adapter.setup()
    result = await adapter.search("q")
    chunk = result.chunks[0]
    assert chunk.doc_path == _stable_id("episode without uuid")
    assert chunk.doc_path.startswith("zep-")


async def test_search_none_score_defaults_to_zero(patched_brain) -> None:
    adapter, fake_client = patched_brain
    fake_client.graph._search_results = [
        _FakeEpisode(content="no score", score=None, uuid_="ep-x"),
    ]
    await adapter.setup()
    result = await adapter.search("q")
    assert result.chunks[0].score == pytest.approx(0.0)


async def test_search_wraps_error(patched_brain, monkeypatch) -> None:
    adapter, fake_client = patched_brain
    await adapter.setup()

    def boom(**_kw: Any) -> Any:
        raise RuntimeError("zep is down")

    monkeypatch.setattr(fake_client.graph, "search", boom)
    result = await adapter.search("anything")
    assert result.chunks == []
    assert result.error is not None
    assert "zep is down" in result.error
    assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# Tests — reset
# ---------------------------------------------------------------------------


async def test_reset_rotates_user_id(patched_brain) -> None:
    adapter, _ = patched_brain
    await adapter.setup()
    first_id = adapter._current_user_id
    await adapter.reset()
    second_id = adapter._current_user_id
    assert first_id != second_id
    assert second_id.startswith("unison-evals-")


async def test_reset_produces_unique_ids_across_calls(patched_brain) -> None:
    adapter, _ = patched_brain
    await adapter.setup()
    ids = set()
    for _ in range(5):
        await adapter.reset()
        ids.add(adapter._current_user_id)
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# Tests — teardown
# ---------------------------------------------------------------------------


async def test_teardown_clears_client(patched_brain) -> None:
    adapter, _ = patched_brain
    await adapter.setup()
    assert adapter._client is not None
    await adapter.teardown()
    assert adapter._client is None


# ---------------------------------------------------------------------------
# Tests — _stable_id helper
# ---------------------------------------------------------------------------


def test_stable_id_deterministic() -> None:
    assert _stable_id("hello") == _stable_id("hello")


def test_stable_id_different_for_different_text() -> None:
    assert _stable_id("foo") != _stable_id("bar")


def test_stable_id_prefix() -> None:
    assert _stable_id("any text").startswith("zep-")
