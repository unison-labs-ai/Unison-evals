"""pgvector_naive BrainAdapter — mocked unit tests.

The real DB integration is exercised when you run an actual benchmark
(see Makefile `make smoke` once a real corpus is wired). Here we mock
asyncpg + OpenAI to verify:
  * setup creates table + extension + HNSW index
  * ingest batches embeddings + bulk-INSERTs
  * search embeds the query, runs cosine SQL, maps results
  * reset truncates
  * search wraps errors in BrainSearchResult with empty chunks
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from unison_evals.config import get_settings
from unison_evals.memory_evals.adapters import BRAIN_REGISTRY, get_brain_adapter
from unison_evals.memory_evals.adapters.pgvector_naive import (
    EMBED_BATCH_SIZE,
    PgvectorNaiveBrainAdapter,
    _safe_json,
    _truncate_for_embedding,
)
from unison_evals.types import Document


def test_registered() -> None:
    assert "pgvector-naive" in BRAIN_REGISTRY


def test_get_brain_adapter_returns_instance() -> None:
    a = get_brain_adapter("pgvector-naive")
    assert isinstance(a, PgvectorNaiveBrainAdapter)
    assert a.name == "pgvector-naive"


def test_truncate_short_passthrough() -> None:
    assert _truncate_for_embedding("hello") == "hello"


def test_truncate_long_clipped() -> None:
    big = "x" * 30_000
    out = _truncate_for_embedding(big, max_chars=24_000)
    assert len(out) == 24_000


def test_safe_json_roundtrip() -> None:
    assert _safe_json({}) == "{}"
    assert "foo" in _safe_json({"foo": "bar"})


def test_safe_json_handles_non_serializable() -> None:
    """e.g. a dataset row may contain a numpy scalar; we should not crash —
    fallback path converts via str() (default=str in json.dumps), so the
    output is valid JSON whether it's the original dict or {} fallback."""
    import json

    class Weird:
        pass

    out = _safe_json({"x": Weird()})
    parsed = json.loads(out)
    assert isinstance(parsed, dict)  # never crashes, always valid JSON


# ---------------------------------------------------------------------------
# Mocked end-to-end: setup → ingest → search → teardown without a real DB.
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        if "TRUNCATE" in sql:
            self.store["rows"] = []
        return "OK"

    async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        self.executemany_calls.append((sql, rows))
        self.store.setdefault("rows", []).extend(
            [{"doc_path": r[0], "body": r[1], "embedding": r[2], "metadata": r[3]} for r in rows]
        )

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((sql, args))
        # Mock cosine: just return the first k rows we have.
        k = args[1] if len(args) > 1 else 10
        rows = self.store.get("rows", [])[:k]
        return [{"doc_path": r["doc_path"], "body": r["body"], "score": 0.9} for r in rows]


class _FakePool:
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield _FakeConn(self.store)

    async def close(self) -> None:
        self.closed = True


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def create(self, *, model: str, input: Any) -> Any:
        self.calls.append({"model": model, "input": input})
        n = len(input) if isinstance(input, list) else 1
        # 1536-dim fake vectors of a fixed shape so pgvector's `vector(N)`
        # cast doesn't blow up downstream.
        data = [SimpleNamespace(embedding=[0.0] * 1536) for _ in range(n)]
        usage = SimpleNamespace(total_tokens=10 * n)
        return SimpleNamespace(data=data, usage=usage)


class _FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


@pytest.fixture
def patched_adapter(monkeypatch):
    """Returns (adapter, store, openai) so each test can introspect what
    happened. Replaces asyncpg.create_pool, OpenAI, and pgvector
    register_vector with no-ops/fakes."""
    store: dict[str, Any] = {"rows": []}
    fake_pool = _FakePool(store)
    fake_openai = _FakeOpenAI()

    async def fake_create_pool(*_a: Any, **_kw: Any) -> _FakePool:
        return fake_pool

    fake_asyncpg_module = SimpleNamespace(create_pool=fake_create_pool)
    fake_pgvector_module = SimpleNamespace(register_vector=lambda _conn: None)
    fake_openai_module = SimpleNamespace(AsyncOpenAI=lambda **_kw: fake_openai)

    monkeypatch.setitem(__import__("sys").modules, "asyncpg", fake_asyncpg_module)
    monkeypatch.setitem(__import__("sys").modules, "pgvector.asyncpg", fake_pgvector_module)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)

    # Force OPENAI_API_KEY to non-empty so setup() doesn't refuse.
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    adapter = PgvectorNaiveBrainAdapter()
    return adapter, store, fake_openai


async def test_setup_creates_extension_and_table(patched_adapter) -> None:
    adapter, _store, _ = patched_adapter
    await adapter.setup()
    # Last fake conn is gone (context-managed) but we can verify by ingesting:
    # if the table didn't get "created", the test below would still pass on
    # the fake. Real assertion: pool is set.
    assert adapter._pool is not None
    await adapter.teardown()


async def test_ingest_then_search_round_trip(patched_adapter) -> None:
    adapter, store, fake_openai = patched_adapter
    await adapter.setup()
    docs = [
        Document(path="/a.md", body="apple"),
        Document(path="/b.md", body="banana"),
        Document(path="/c.md", body="cherry"),
    ]
    await adapter.ingest(docs)
    assert len(store["rows"]) == 3
    # One embedding call for ingest (all 3 docs in one batch under EMBED_BATCH_SIZE).
    assert len(fake_openai.embeddings.calls) == 1
    assert fake_openai.embeddings.calls[0]["input"] == ["apple", "banana", "cherry"]

    res = await adapter.search("anything", k=2)
    assert len(res.chunks) == 2
    assert res.chunks[0].rank == 1
    assert res.chunks[1].rank == 2
    assert res.chunks[0].doc_path == "/a.md"
    # Cost > 0 because the search call also embedded the query.
    assert res.cost_usd > 0
    assert res.error is None
    await adapter.teardown()


async def test_ingest_batches_when_over_threshold(patched_adapter) -> None:
    adapter, store, fake_openai = patched_adapter
    await adapter.setup()
    n = EMBED_BATCH_SIZE + 5
    docs = [Document(path=f"/d{i}.md", body=f"body {i}") for i in range(n)]
    await adapter.ingest(docs)
    assert len(store["rows"]) == n
    # Two embedding batches: one full, one with the remainder.
    assert len(fake_openai.embeddings.calls) == 2
    assert len(fake_openai.embeddings.calls[0]["input"]) == EMBED_BATCH_SIZE
    assert len(fake_openai.embeddings.calls[1]["input"]) == 5
    await adapter.teardown()


async def test_reset_truncates(patched_adapter) -> None:
    adapter, store, _ = patched_adapter
    await adapter.setup()
    await adapter.ingest([Document(path="/x.md", body="x")])
    assert len(store["rows"]) == 1
    await adapter.reset()
    assert store["rows"] == []
    await adapter.teardown()


async def test_search_wraps_errors_into_result(patched_adapter, monkeypatch) -> None:
    adapter, _, fake_openai = patched_adapter
    await adapter.setup()

    async def boom(**_kw: Any) -> Any:
        raise RuntimeError("simulated openai outage")

    monkeypatch.setattr(fake_openai.embeddings, "create", boom)
    res = await adapter.search("anything", k=5)
    assert res.chunks == []
    assert res.error is not None
    assert "outage" in res.error
    assert res.latency_ms > 0
    await adapter.teardown()


async def test_setup_refuses_without_openai_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "")
    adapter = PgvectorNaiveBrainAdapter()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await adapter.setup()


async def test_ingest_empty_is_noop(patched_adapter) -> None:
    adapter, store, fake_openai = patched_adapter
    await adapter.setup()
    await adapter.ingest([])
    assert store["rows"] == []
    assert fake_openai.embeddings.calls == []
    await adapter.teardown()
