"""Track 3 tests for the Unison agent adapter.

Covers:
  * seed_docs are serialised with kind="raw" in the POST body
  * seedDocsCount / seedEmbedMs from the response surface in AdapterResult.raw
  * Providing BOTH oracle_context + seed_docs returns an error without making an HTTP call
  * No seed_docs → backward-compatible behavior (omitting seed_docs is valid)
"""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.adapters.unison_agent import UnisonAgentAdapter
from unison_evals.types import Document


@pytest.fixture()
def health_mock(httpx_mock):
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"status": "ok"},
    )
    return httpx_mock


async def test_seed_docs_sent_with_kind_raw(health_mock) -> None:
    """POST body must include seedDocs with kind='raw' for each doc."""
    health_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={
            "answer": "Paris",
            "sessionId": "s1",
            "totalCostUsd": 0.005,
            "seedDocsCount": 2,
            "seedEmbedMs": 123.4,
        },
    )

    docs = [
        Document(path="/sessions/day1.md", body="Alice visited Paris on Monday."),
        Document(path="/sessions/day2.md", body="She had croissants for breakfast."),
    ]

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("Where did Alice visit?", seed_docs=docs)
    finally:
        await adapter.teardown()

    assert result.answer == "Paris"
    assert result.error is None

    # Verify the request body that was sent.
    requests = health_mock.get_requests()
    post_req = next(r for r in requests if r.method == "POST")
    import json

    body = json.loads(post_req.content)
    assert "seedDocs" in body
    assert "oracleContext" not in body
    assert len(body["seedDocs"]) == 2
    for sent_doc, original_doc in zip(body["seedDocs"], docs, strict=True):
        assert sent_doc["path"] == original_doc.path
        assert sent_doc["body"] == original_doc.body
        assert sent_doc["kind"] == "raw"  # critical — skips Unison extract pipeline


async def test_seed_embed_telemetry_surfaced(health_mock) -> None:
    """seedDocsCount and seedEmbedMs from the response appear in AdapterResult.raw."""
    health_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={
            "answer": "42",
            "totalCostUsd": 0.001,
            "seedDocsCount": 5,
            "seedEmbedMs": 88.0,
        },
    )

    docs = [Document(path="/doc.md", body="The answer is 42.")]
    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("What is the answer?", seed_docs=docs)
    finally:
        await adapter.teardown()

    assert result.raw.get("seed_docs_count") == 5
    assert result.raw.get("seed_embed_ms") == pytest.approx(88.0)


async def test_both_oracle_and_seed_docs_returns_error(health_mock) -> None:
    """Providing BOTH oracle_context and seed_docs must return an error result
    without making any HTTP call to /eval-turn."""
    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer(
            "Q",
            oracle_context="some gold context",
            seed_docs=[Document(path="/d.md", body="body")],
        )
    finally:
        await adapter.teardown()

    assert result.error == "seed_docs and oracle_context are mutually exclusive"
    assert result.answer == ""
    # Only the health-check GET was made; no POST.
    requests = health_mock.get_requests()
    post_reqs = [r for r in requests if r.method == "POST"]
    assert post_reqs == [], "No POST should be made when both args are set"


async def test_no_seed_docs_backward_compat(health_mock) -> None:
    """When seed_docs is None, POST body must NOT include 'seedDocs'."""
    health_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={"answer": "ok", "totalCostUsd": 0.0},
    )

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("ping")
    finally:
        await adapter.teardown()

    assert result.error is None
    import json

    requests = health_mock.get_requests()
    post_req = next(r for r in requests if r.method == "POST")
    body = json.loads(post_req.content)
    assert "seedDocs" not in body
    assert "oracleContext" not in body


async def test_oracle_context_still_works(health_mock) -> None:
    """Existing Track 2 (oracle_context only) path remains unchanged."""
    health_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={"answer": "9:45 PM", "totalCostUsd": 0.002},
    )

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("What time?", oracle_context="Flight lands at 9:45 PM.")
    finally:
        await adapter.teardown()

    assert result.error is None
    import json

    requests = health_mock.get_requests()
    post_req = next(r for r in requests if r.method == "POST")
    body = json.loads(post_req.content)
    assert body["oracleContext"] == "Flight lands at 9:45 PM."
    assert "seedDocs" not in body
