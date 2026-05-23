"""BrainAdapter contract — abstract enforcement + types + registry."""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.adapters import BRAIN_REGISTRY, BrainAdapter, get_brain_adapter
from unison_evals.memory_evals.adapters.base import BrainAdapter as BaseBrainAdapter
from unison_evals.types import BrainSearchResult, Document, RetrievedChunk


def test_brain_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        BrainAdapter()  # type: ignore[abstract]


def test_brain_adapter_re_export_matches_base() -> None:
    """The package __init__ should re-export the same class as base.py."""
    assert BrainAdapter is BaseBrainAdapter


def test_brain_registry_has_built_ins() -> None:
    """At least one brain adapter must be registered (v0.1+)."""
    assert len(BRAIN_REGISTRY) >= 1


def test_get_brain_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown brain adapter"):
        get_brain_adapter("nope")


def test_document_minimal() -> None:
    doc = Document(path="/notes/a.md", body="hello world")
    assert doc.path == "/notes/a.md"
    assert doc.body == "hello world"
    assert doc.metadata == {}


def test_document_with_metadata() -> None:
    doc = Document(
        path="/sessions/2026-01-05.md",
        body="conversation transcript",
        metadata={"date": "2026-01-05", "session_id": "s1"},
    )
    assert doc.metadata["date"] == "2026-01-05"
    assert doc.metadata["session_id"] == "s1"


def test_retrieved_chunk_defaults() -> None:
    c = RetrievedChunk(doc_path="/x.md", chunk_text="text")
    assert c.score == 0.0
    assert c.rank == 0
    assert c.raw == {}


def test_retrieved_chunk_full() -> None:
    c = RetrievedChunk(
        doc_path="/wiki/yashraj.md",
        chunk_text="Yashraj is a partner at Lightspeed.",
        score=0.87,
        rank=1,
        raw={"source": "bm25"},
    )
    assert c.doc_path == "/wiki/yashraj.md"
    assert c.score == 0.87
    assert c.rank == 1


def test_brain_search_result_empty_default() -> None:
    res = BrainSearchResult()
    assert res.chunks == []
    assert res.latency_ms == 0.0
    assert res.cost_usd == 0.0
    assert res.error is None


def test_brain_search_result_with_chunks() -> None:
    chunks = [
        RetrievedChunk(doc_path="/a.md", chunk_text="alpha", score=0.9, rank=1),
        RetrievedChunk(doc_path="/b.md", chunk_text="beta", score=0.8, rank=2),
    ]
    res = BrainSearchResult(chunks=chunks, latency_ms=42.0, cost_usd=0.001)
    assert len(res.chunks) == 2
    assert res.chunks[0].doc_path == "/a.md"
    assert res.latency_ms == 42.0


class _FakeBrain(BrainAdapter):
    """In-memory stub used to verify the contract is satisfiable."""

    name = "fake-brain"

    def __init__(self) -> None:
        self.docs: list[Document] = []

    async def ingest(self, docs: list[Document]) -> None:
        self.docs.extend(docs)

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        # Trivial substring rank — only here to test the contract.
        scored = [
            (d.body.lower().count(query.lower()), d)
            for d in self.docs
            if query.lower() in d.body.lower()
        ]
        scored.sort(key=lambda x: (-x[0], x[1].path))
        chunks = [
            RetrievedChunk(doc_path=d.path, chunk_text=d.body, score=float(score), rank=i + 1)
            for i, (score, d) in enumerate(scored[:k])
        ]
        return BrainSearchResult(chunks=chunks)


async def test_fake_brain_round_trip() -> None:
    brain = _FakeBrain()
    await brain.setup()
    await brain.ingest(
        [
            Document(path="/a.md", body="apple banana cherry"),
            Document(path="/b.md", body="banana banana"),
            Document(path="/c.md", body="cherry"),
        ]
    )
    res = await brain.search("banana", k=5)
    assert len(res.chunks) == 2
    # b.md has 2 hits → ranks first
    assert res.chunks[0].doc_path == "/b.md"
    assert res.chunks[0].rank == 1
    assert res.chunks[1].doc_path == "/a.md"
    assert res.chunks[1].rank == 2
    await brain.teardown()


async def test_fake_brain_reset_default_noop() -> None:
    """The default reset() is a no-op — adapters that need it must override."""
    brain = _FakeBrain()
    await brain.ingest([Document(path="/a.md", body="x")])
    await brain.reset()  # default no-op leaves state intact
    assert len(brain.docs) == 1


async def test_fake_brain_search_misses_returns_empty() -> None:
    brain = _FakeBrain()
    await brain.ingest([Document(path="/a.md", body="hello")])
    res = await brain.search("nonexistent")
    assert res.chunks == []
    assert res.error is None
