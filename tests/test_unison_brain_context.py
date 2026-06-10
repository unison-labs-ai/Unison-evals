"""Tests for unison-brain-context adapter (mocked HTTP)."""

from __future__ import annotations

import json
import time

import pytest

from unison_evals.memory_evals.adapters import REGISTRY, get_adapter
from unison_evals.memory_evals.adapters.unison_brain_context import (
    UnisonBrainContextAdapter,
    _mint_hs256_jwt,
)
from unison_evals.types import Document

# ---------------------------------------------------------------------------
# JWT minting
# ---------------------------------------------------------------------------


def test_mint_hs256_jwt_structure() -> None:
    token = _mint_hs256_jwt("user-123", "tenant-456", "secret", ttl=300)
    parts = token.split(".")
    assert len(parts) == 3
    # Decode header + payload (no padding needed — b64url decode adds it)
    import base64

    def decode(s: str) -> dict:
        pad = 4 - len(s) % 4
        return json.loads(base64.urlsafe_b64decode(s + "=" * pad))

    header = decode(parts[0])
    payload = decode(parts[1])
    assert header["alg"] == "HS256"
    assert payload["sub"] == "user-123"
    assert payload["role"] == "authenticated"
    assert payload["app_metadata"]["tenant_id"] == "tenant-456"
    # Token should not be expired right after minting.
    assert payload["exp"] > int(time.time())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_brain_context_in_registry() -> None:
    assert "unison-brain-context" in REGISTRY


def test_get_adapter_brain_context() -> None:
    a = get_adapter("unison-brain-context")
    assert isinstance(a, UnisonBrainContextAdapter)
    assert a.name == "unison-brain-context"


# ---------------------------------------------------------------------------
# setup() auth guards
# ---------------------------------------------------------------------------


async def test_setup_fails_remote_no_eval_secret(monkeypatch) -> None:
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "")
    monkeypatch.setattr(s, "unison_api_url", "https://api.example.com")

    adapter = UnisonBrainContextAdapter()
    with pytest.raises(RuntimeError, match="UNISON_EVAL_SECRET is required"):
        await adapter.setup()


async def test_setup_fails_no_jwt_source(monkeypatch) -> None:
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "secret")
    monkeypatch.setattr(s, "unison_api_url", "http://localhost:3001")
    monkeypatch.setattr(s, "unison_jwt", "")
    monkeypatch.setattr(s, "unison_eval_jwt", "")
    monkeypatch.setattr(s, "supabase_jwt_secret", "")
    monkeypatch.setattr(s, "unison_brain_machine_key", "")

    adapter = UnisonBrainContextAdapter()
    with pytest.raises(RuntimeError, match="needs auth for GET /v1/brain/context"):
        await adapter.setup()


# ---------------------------------------------------------------------------
# Full happy path (mocked HTTP)
# ---------------------------------------------------------------------------


async def test_answer_e2e_flow(httpx_mock, monkeypatch) -> None:
    """Provision → seed → brain/context → reader LLM → teardown."""
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "test-secret")
    monkeypatch.setattr(s, "unison_api_url", "http://localhost:3001")
    monkeypatch.setattr(
        s, "supabase_jwt_secret", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    monkeypatch.setattr(s, "unison_jwt", "")
    monkeypatch.setattr(s, "unison_eval_jwt", "")
    monkeypatch.setattr(s, "openai_api_key", "sk-test")
    # Force the per-question provision path (not the shared-tenant machine-key path).
    monkeypatch.setattr(s, "unison_brain_machine_key", "")
    monkeypatch.setattr(s, "unison_eval_tenant_id", "")
    monkeypatch.setattr(s, "unison_eval_user_id", "")

    # health check
    httpx_mock.add_response(url="http://localhost:3001/health", status_code=200, json={"ok": True})
    # provision
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/provision",
        status_code=200,
        json={"tenantId": "tid-1", "userId": "uid-1"},
    )
    # seed
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/seed",
        status_code=200,
        json={"docsWritten": 2, "embedDurationMs": 42.0},
    )
    # brain/context — match by URL prefix + match_params so query string is handled
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/v1/brain/context",
        match_params={"q": "How old is Alice?"},
        status_code=200,
        json={
            "contextMd": "## Session 1\nAlice is 30 years old.",
            "hits": [{"doc": {"path": "/sessions/1.md"}, "score": 0.9}],
            "entities": [],
            "weakEvidence": False,
        },
    )
    # teardown
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/teardown",
        status_code=200,
        json={"ok": True},
    )

    # Patch the reader LLM to avoid real OpenAI call.
    async def _fake_reader(self: UnisonBrainContextAdapter, question: str, context_md: str) -> str:
        assert "Alice is 30" in context_md
        return "Alice is 30 years old."

    monkeypatch.setattr(UnisonBrainContextAdapter, "_reader_llm", _fake_reader)

    adapter = UnisonBrainContextAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer(
            "How old is Alice?",
            seed_docs=[
                Document(path="/sessions/1.md", body="## Session 1\nAlice is 30 years old."),
                Document(path="/sessions/2.md", body="## Session 2\nBob is 25 years old."),
            ],
            question_id="q001",
        )
        assert result.answer == "Alice is 30 years old."
        assert result.error is None
        assert result.latency_ms > 0
        assert result.raw["seed_docs_count"] == 2
        assert result.raw["hits"] == 1
        assert result.raw["weak_evidence"] is False
    finally:
        await adapter.teardown()


# ---------------------------------------------------------------------------
# Oracle track (no provision/seed needed)
# ---------------------------------------------------------------------------


async def test_answer_oracle_track(httpx_mock, monkeypatch) -> None:
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "test-secret")
    monkeypatch.setattr(s, "unison_api_url", "http://localhost:3001")
    monkeypatch.setattr(
        s, "supabase_jwt_secret", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    monkeypatch.setattr(s, "unison_jwt", "")
    monkeypatch.setattr(s, "unison_eval_jwt", "")
    monkeypatch.setattr(s, "unison_brain_machine_key", "")
    monkeypatch.setattr(s, "unison_eval_tenant_id", "")
    monkeypatch.setattr(s, "unison_eval_user_id", "")

    httpx_mock.add_response(url="http://localhost:3001/health", status_code=200, json={"ok": True})

    async def _fake_reader(self: UnisonBrainContextAdapter, question: str, context_md: str) -> str:
        return "42"

    monkeypatch.setattr(UnisonBrainContextAdapter, "_reader_llm", _fake_reader)

    adapter = UnisonBrainContextAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer(
            "What is the answer?",
            oracle_context="The answer is 42.",
        )
        assert result.answer == "42"
        assert result.error is None
        assert result.raw["mode"] == "oracle"
    finally:
        await adapter.teardown()


# ---------------------------------------------------------------------------
# Mutual-exclusion guard
# ---------------------------------------------------------------------------


async def test_answer_rejects_both_oracle_and_seed(httpx_mock, monkeypatch) -> None:
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "test-secret")
    monkeypatch.setattr(s, "unison_api_url", "http://localhost:3001")
    monkeypatch.setattr(
        s, "supabase_jwt_secret", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    monkeypatch.setattr(s, "unison_jwt", "")
    monkeypatch.setattr(s, "unison_eval_jwt", "")
    monkeypatch.setattr(s, "unison_brain_machine_key", "")
    monkeypatch.setattr(s, "unison_eval_tenant_id", "")
    monkeypatch.setattr(s, "unison_eval_user_id", "")

    httpx_mock.add_response(url="http://localhost:3001/health", status_code=200, json={"ok": True})

    adapter = UnisonBrainContextAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer(
            "Q",
            oracle_context="ctx",
            seed_docs=[Document(path="/x.md", body="body")],
        )
        assert result.error == "seed_docs and oracle_context are mutually exclusive"
    finally:
        await adapter.teardown()


# ---------------------------------------------------------------------------
# brain/context 4xx is surfaced as an error (not a crash)
# ---------------------------------------------------------------------------


async def test_answer_brain_context_401(httpx_mock, monkeypatch) -> None:
    from unison_evals.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "unison_eval_secret", "test-secret")
    monkeypatch.setattr(s, "unison_api_url", "http://localhost:3001")
    monkeypatch.setattr(
        s, "supabase_jwt_secret", "super-secret-jwt-token-with-at-least-32-characters-long"
    )
    monkeypatch.setattr(s, "unison_jwt", "")
    monkeypatch.setattr(s, "unison_eval_jwt", "")
    # Force the per-question provision path (not the shared-tenant machine-key path).
    monkeypatch.setattr(s, "unison_brain_machine_key", "")
    monkeypatch.setattr(s, "unison_eval_tenant_id", "")
    monkeypatch.setattr(s, "unison_eval_user_id", "")

    httpx_mock.add_response(url="http://localhost:3001/health", status_code=200, json={"ok": True})
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/provision",
        status_code=200,
        json={"tenantId": "t1", "userId": "u1"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/seed",
        status_code=200,
        json={"docsWritten": 1, "embedDurationMs": 0.0},
    )
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/v1/brain/context",
        match_params={"q": "Q"},
        status_code=401,
        text="Unauthorized",
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/v1/eval/teardown",
        status_code=200,
        json={"ok": True},
    )

    adapter = UnisonBrainContextAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer(
            "Q",
            seed_docs=[Document(path="/x.md", body="body")],
            question_id="q1",
        )
        assert result.answer == ""
        assert result.error is not None
        assert "401" in result.error
    finally:
        await adapter.teardown()
