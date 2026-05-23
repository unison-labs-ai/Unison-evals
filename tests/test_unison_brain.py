"""unison_brain BrainAdapter — mocked HTTP tests."""

from __future__ import annotations

import re

import pytest

from unison_evals.memory_evals.adapters import BRAIN_REGISTRY, get_brain_adapter
from unison_evals.memory_evals.adapters._url_utils import is_localhost_url
from unison_evals.memory_evals.adapters.unison_brain import (
    UnisonBrainAdapter,
    _extract_trpc_hits,
    _hits_to_chunks,
)
from unison_evals.types import Document

SEARCH_URL_RE = re.compile(r"http://localhost:3001/trpc/agents\.cortex\.search\?.*")


# ---------------------------------------------------------------------------
# is_localhost_url (brain-side mirror — canonical tests are in test_adapters.py)
# ---------------------------------------------------------------------------


def test_is_localhost_url_true_cases_brain() -> None:
    assert is_localhost_url("http://localhost:3001")
    assert is_localhost_url("http://127.0.0.1:3001")
    assert is_localhost_url("http://host.docker.internal:3001")


def test_is_localhost_url_false_cases_brain() -> None:
    assert not is_localhost_url("https://api.eval.unison.ai")
    assert not is_localhost_url("")


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


def test_registered() -> None:
    assert "unison-brain" in BRAIN_REGISTRY


def test_get_brain_adapter_returns_instance() -> None:
    a = get_brain_adapter("unison-brain")
    assert isinstance(a, UnisonBrainAdapter)
    assert a.name == "unison-brain"


def test_extract_trpc_v10_wrapped() -> None:
    payload = {
        "result": {
            "data": {
                "json": [
                    {"doc": {"path": "/a.md", "bodyMd": "alpha"}, "score": 0.9},
                    {"doc": {"path": "/b.md", "bodyMd": "beta"}, "score": 0.8},
                ]
            }
        }
    }
    hits = _extract_trpc_hits(payload)
    assert len(hits) == 2
    assert hits[0]["doc"]["path"] == "/a.md"


def test_extract_handles_unwrapped_list() -> None:
    """Defensive — accept the already-unwrapped shape too."""
    hits = _extract_trpc_hits([{"doc": {"path": "/x.md", "bodyMd": "x"}}])  # type: ignore[arg-type]
    # Function expects dict but is robust to non-dict; check it doesn't crash.
    assert isinstance(hits, list)


def test_extract_empty_returns_empty() -> None:
    assert _extract_trpc_hits({}) == []
    assert _extract_trpc_hits({"result": {}}) == []
    assert _extract_trpc_hits({"result": {"data": {"json": "not a list"}}}) == []


def test_hits_to_chunks_assigns_ranks() -> None:
    hits = [
        {"doc": {"path": "/a.md", "bodyMd": "alpha"}, "score": 0.9, "sources": ["bm25"]},
        {"doc": {"path": "/b.md", "bodyMd": "beta"}, "score": 0.8, "sources": ["dense"]},
        {"doc": {"path": "/c.md", "bodyMd": "gamma"}, "score": 0.7},
    ]
    chunks = _hits_to_chunks(hits)
    assert len(chunks) == 3
    assert chunks[0].rank == 1
    assert chunks[1].rank == 2
    assert chunks[2].rank == 3
    assert chunks[0].doc_path == "/a.md"
    assert chunks[0].score == 0.9
    assert chunks[0].raw["sources"] == ["bm25"]


def test_hits_to_chunks_falls_back_to_inverse_rank_score() -> None:
    """When a hit has no score, fall back to 1/(rank+1)."""
    hits = [{"doc": {"path": "/a.md", "bodyMd": "alpha"}}]
    chunks = _hits_to_chunks(hits)
    assert chunks[0].score == 1.0  # 1 / (0 + 1)


def test_hits_to_chunks_handles_id_only_doc() -> None:
    """Some shapes use 'id' as the path; we should still produce a chunk."""
    hits = [{"doc": {"id": "doc-123", "text": "content"}}]
    chunks = _hits_to_chunks(hits)
    assert chunks[0].doc_path == "doc-123"
    assert chunks[0].chunk_text == "content"


async def test_setup_refuses_without_jwt_on_remote(monkeypatch) -> None:
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "https://api.eval.unison.ai")
    adapter = UnisonBrainAdapter()
    with pytest.raises(RuntimeError, match="UNISON_JWT"):
        await adapter.setup()


async def test_setup_succeeds_localhost_no_jwt(httpx_mock, monkeypatch) -> None:
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    await adapter.teardown()


async def test_no_auth_header_when_localhost_no_jwt(httpx_mock, monkeypatch) -> None:
    """Requests sent without JWT on localhost must not include Authorization."""
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    httpx_mock.add_response(
        method="GET",
        url=SEARCH_URL_RE,
        status_code=200,
        json={"result": {"data": {"json": []}}},
    )

    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        await adapter.search("anything")
    finally:
        await adapter.teardown()

    for req in httpx_mock.get_requests():
        assert "authorization" not in req.headers


async def test_auth_header_included_when_jwt_provided_on_localhost(httpx_mock, monkeypatch) -> None:
    """When JWT is set even on localhost, it must be forwarded."""
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "my-brain-token")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    httpx_mock.add_response(
        method="GET",
        url=SEARCH_URL_RE,
        status_code=200,
        json={"result": {"data": {"json": []}}},
    )

    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        await adapter.search("anything")
    finally:
        await adapter.teardown()

    search_req = next(r for r in httpx_mock.get_requests() if "cortex.search" in str(r.url))
    assert search_req.headers["authorization"] == "Bearer my-brain-token"


async def test_search_round_trip(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    httpx_mock.add_response(
        method="GET",
        url=SEARCH_URL_RE,
        status_code=200,
        json={
            "result": {
                "data": {
                    "json": [
                        {
                            "doc": {"path": "/wiki/yashraj.md", "bodyMd": "Yashraj is a partner"},
                            "score": 0.92,
                            "sources": ["bm25", "dense"],
                        },
                        {
                            "doc": {
                                "path": "/notes/2026-04-12.md",
                                "bodyMd": "Met Yashraj at YC dinner",
                            },
                            "score": 0.81,
                            "sources": ["dense"],
                        },
                    ]
                }
            }
        },
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        result = await adapter.search("who is yashraj", k=2)
        assert len(result.chunks) == 2
        assert result.chunks[0].doc_path == "/wiki/yashraj.md"
        assert result.chunks[0].rank == 1
        assert result.chunks[0].score == 0.92
        assert result.error is None
        assert result.latency_ms > 0
    finally:
        await adapter.teardown()


async def test_search_handles_500(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    httpx_mock.add_response(
        method="GET",
        url=SEARCH_URL_RE,
        status_code=500,
        text="boom",
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        result = await adapter.search("anything")
        assert result.chunks == []
        assert result.error is not None
        assert "500" in result.error
    finally:
        await adapter.teardown()


async def test_setup_fails_when_health_unreachable(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=503,
        text="down",
    )
    adapter = UnisonBrainAdapter()
    with pytest.raises(RuntimeError, match="health check failed"):
        await adapter.setup()


async def test_ingest_raises_with_helpful_message(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        with pytest.raises(NotImplementedError, match="brain-cli import"):
            await adapter.ingest([Document(path="/a.md", body="x")])
    finally:
        await adapter.teardown()


async def test_ingest_empty_is_silent_noop(httpx_mock) -> None:
    """Empty list should NOT raise — runner may call ingest([]) on the
    'no documents this question' path."""
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        await adapter.ingest([])  # no raise
    finally:
        await adapter.teardown()


async def test_reset_is_noop(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"ok": True},
    )
    adapter = UnisonBrainAdapter()
    await adapter.setup()
    try:
        await adapter.reset()  # no-op, no error
    finally:
        await adapter.teardown()
