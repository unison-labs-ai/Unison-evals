"""Adapter contract + Unison HTTP adapter shape (mocked)."""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.adapters import REGISTRY, get_adapter
from unison_evals.memory_evals.adapters._url_utils import is_localhost_url
from unison_evals.memory_evals.adapters.base import AgentAdapter
from unison_evals.memory_evals.adapters.claude_code import ClaudeCodeAdapter
from unison_evals.memory_evals.adapters.unison_agent import UnisonAgentAdapter

# ---------------------------------------------------------------------------
# is_localhost_url
# ---------------------------------------------------------------------------


def test_is_localhost_url_true_cases() -> None:
    assert is_localhost_url("http://localhost:3001")
    assert is_localhost_url("http://127.0.0.1:3001")
    assert is_localhost_url("http://host.docker.internal:3001")
    assert is_localhost_url("http://0.0.0.0:3001")
    assert is_localhost_url("http://[::1]:3001")


def test_is_localhost_url_false_cases() -> None:
    assert not is_localhost_url("https://api.eval.unison.ai")
    assert not is_localhost_url("https://example.com")
    assert not is_localhost_url("")


# ---------------------------------------------------------------------------
# setup() auth logic
# ---------------------------------------------------------------------------


async def test_setup_succeeds_localhost_no_jwt(httpx_mock, monkeypatch) -> None:
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"status": "ok"},
    )
    adapter = UnisonAgentAdapter()
    await adapter.setup()
    await adapter.teardown()


async def test_setup_fails_remote_no_jwt(monkeypatch) -> None:
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "https://api.eval.unison.ai")

    adapter = UnisonAgentAdapter()
    with pytest.raises(RuntimeError, match="UNISON_JWT not set"):
        await adapter.setup()


async def test_setup_no_auth_header_when_localhost_no_jwt(httpx_mock, monkeypatch) -> None:
    """When targeting localhost without JWT, requests must not include Authorization."""
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"status": "ok"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={"answer": "pong", "totalCostUsd": 0.0},
    )

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        await adapter.answer("ping")
    finally:
        await adapter.teardown()

    requests = httpx_mock.get_requests()
    for req in requests:
        assert "authorization" not in req.headers


async def test_setup_includes_auth_header_when_jwt_provided(httpx_mock, monkeypatch) -> None:
    """When JWT is provided even on localhost, it must be forwarded."""
    from unison_evals.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "unison_jwt", "my-token")
    monkeypatch.setattr(settings, "unison_api_url", "http://localhost:3001")

    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"status": "ok"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={"answer": "pong", "totalCostUsd": 0.0},
    )

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        await adapter.answer("ping")
    finally:
        await adapter.teardown()

    post_req = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    assert post_req.headers["authorization"] == "Bearer my-token"


# ---------------------------------------------------------------------------
# Existing tests below (unchanged)
# ---------------------------------------------------------------------------


def test_registry_has_built_ins() -> None:
    assert "unison-agent" in REGISTRY
    assert "claude-code" in REGISTRY


def test_get_adapter_returns_instance() -> None:
    a = get_adapter("unison-agent")
    assert isinstance(a, UnisonAgentAdapter)
    assert a.name == "unison-agent"


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_adapter("nonexistent")


def test_adapters_implement_contract() -> None:
    for name, cls in REGISTRY.items():
        instance = cls()
        assert isinstance(instance, AgentAdapter)
        assert instance.name == name


async def test_unison_adapter_http_shape(httpx_mock) -> None:
    """Verifies the Unison adapter sends the right POST body and parses
    the response shape from the eval-turn endpoint."""
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:3001/health",
        status_code=200,
        json={"status": "ok"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=200,
        json={
            "answer": "9:45 PM",
            "sessionId": "fake-session",
            "totalCostUsd": 0.0123,
            "totalSteps": 2,
            "finishReason": "no-more-tool-use",
        },
    )

    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("What time?", oracle_context="some context")
        assert result.answer == "9:45 PM"
        assert result.cost_usd == 0.0123
        assert result.latency_ms > 0
        assert result.error is None
    finally:
        await adapter.teardown()


async def test_unison_adapter_handles_500(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET", url="http://localhost:3001/health", status_code=200, json={"ok": True}
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:3001/api/rest/agents/eval-turn",
        status_code=500,
        text="server error",
    )
    adapter = UnisonAgentAdapter()
    await adapter.setup()
    try:
        result = await adapter.answer("Q")
        assert result.answer == ""
        assert result.error is not None
        assert "500" in result.error
    finally:
        await adapter.teardown()


def test_claude_code_prompt_with_oracle() -> None:
    prompt = ClaudeCodeAdapter._build_prompt("What time?", "Some context here.")
    assert "What time?" in prompt
    assert "Some context here." in prompt
    assert "<context>" in prompt


def test_claude_code_prompt_without_oracle() -> None:
    prompt = ClaudeCodeAdapter._build_prompt("What time?", None)
    assert prompt == "What time?"


def test_claude_code_parse_json_output() -> None:
    raw = '{"result": "The flight lands at 9:45 PM.", "usage": {"input_tokens": 100, "output_tokens": 30}}'
    answer, usage = ClaudeCodeAdapter._parse_output(raw)
    assert answer == "The flight lands at 9:45 PM."
    assert usage["input_tokens"] == 100


def test_claude_code_parse_plain_text_fallback() -> None:
    answer, usage = ClaudeCodeAdapter._parse_output("Plain text response\n")
    assert answer == "Plain text response"
    assert usage == {}


def test_claude_code_cost_estimate_with_usage() -> None:
    cost = ClaudeCodeAdapter._estimate_cost(
        usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        prompt="x",
        answer="y",
    )
    # Sonnet 4.5: $3 + $15 per Mtok = $18 total for 1M+1M
    assert abs(cost - 18.0) < 0.01


def test_claude_code_cost_estimate_fallback_to_chars() -> None:
    cost = ClaudeCodeAdapter._estimate_cost(
        usage={},
        prompt="x" * 4_000_000,  # ~1M tokens
        answer="y" * 4_000_000,
    )
    # Approximation: 1M tokens each, same pricing as above
    assert abs(cost - 18.0) < 1.0
