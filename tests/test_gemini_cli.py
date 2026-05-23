"""GeminiCliAdapter — mocked unit tests.

All subprocess calls are mocked via monkeypatch so the tests run without
the `gemini` binary installed.

Verifies:
  * registered name is 'gemini-cli'
  * _build_prompt with and without oracle context
  * _parse_output with valid JSON, malformed JSON, plain text fallback
  * _estimate_cost with usage stats and char-count fallback
  * _format_seed_docs empty + populated
  * mutual exclusion check returns error
  * subprocess args include '-p', '--output-format', 'json'
  * timeout path returns error with latency > 0
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.gemini_cli import (
    GEMINI_INPUT_USD_PER_MTOK,
    GEMINI_OUTPUT_USD_PER_MTOK,
    GeminiCliAdapter,
    _build_prompt,
    _estimate_cost,
    _format_seed_docs,
    _parse_output,
)
from unison_evals.types import Document

# ---------------------------------------------------------------------------
# Registry + name
# ---------------------------------------------------------------------------


def test_adapter_name_registered() -> None:
    assert "gemini-cli" in REGISTRY
    assert REGISTRY["gemini-cli"] is GeminiCliAdapter


def test_adapter_name_class_attr() -> None:
    assert GeminiCliAdapter.name == "gemini-cli"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_no_context() -> None:
    assert _build_prompt("What time is it?", None) == "What time is it?"


def test_build_prompt_with_oracle() -> None:
    prompt = _build_prompt("What is the capital?", "Paris is the capital.")
    assert "What is the capital?" in prompt
    assert "Paris is the capital." in prompt
    assert "<context>" in prompt
    assert prompt.index("<context>") < prompt.index("What is the capital?")


# ---------------------------------------------------------------------------
# _parse_output
# ---------------------------------------------------------------------------


def test_parse_output_valid_json() -> None:
    raw = json.dumps(
        {
            "response": "The answer is 42.",
            "stats": {"input_token_count": 100, "output_token_count": 30},
        }
    )
    answer, usage = _parse_output(raw)
    assert answer == "The answer is 42."
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 30


def test_parse_output_json_missing_stats() -> None:
    raw = json.dumps({"response": "Just text, no stats."})
    answer, usage = _parse_output(raw)
    assert answer == "Just text, no stats."
    assert usage == {"input_tokens": 0, "output_tokens": 0}


def test_parse_output_json_result_key_fallback() -> None:
    raw = json.dumps({"result": "Alternate key answer."})
    answer, _usage = _parse_output(raw)
    assert answer == "Alternate key answer."


def test_parse_output_malformed_json_plain_text() -> None:
    answer, usage = _parse_output("Plain text response here\n")
    assert answer == "Plain text response here"
    assert usage == {}


def test_parse_output_empty_string() -> None:
    answer, usage = _parse_output("")
    assert answer == ""
    assert usage == {}


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_with_usage() -> None:
    cost = _estimate_cost(
        usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        prompt="x",
        answer="y",
    )
    expected = (
        1_000_000 * GEMINI_INPUT_USD_PER_MTOK + 1_000_000 * GEMINI_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    assert abs(cost - expected) < 0.01
    assert abs(expected - 11.25) < 0.01


def test_estimate_cost_fallback_to_chars() -> None:
    cost = _estimate_cost(
        usage={},
        prompt="x" * 4_000_000,
        answer="y" * 4_000_000,
    )
    assert abs(cost - 11.25) < 1.0


# ---------------------------------------------------------------------------
# _format_seed_docs
# ---------------------------------------------------------------------------


def test_format_seed_docs_empty() -> None:
    result = _format_seed_docs([])
    assert result == ""


def test_format_seed_docs_populated() -> None:
    docs = [
        Document(path="/a.md", body="Body A"),
        Document(path="/b.md", body="Body B"),
    ]
    result = _format_seed_docs(docs)
    assert "[Document: /a.md]" in result
    assert "Body A" in result
    assert "[Document: /b.md]" in result
    assert "Body B" in result


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


async def test_mutual_exclusion_returns_error() -> None:
    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer(
        "q",
        oracle_context="ctx",
        seed_docs=[Document(path="/x.md", body="body")],
    )
    assert result.error == "seed_docs and oracle_context are mutually exclusive"
    assert result.answer == ""


# ---------------------------------------------------------------------------
# Subprocess — happy path (mocked)
# ---------------------------------------------------------------------------


async def test_answer_calls_correct_subprocess_args(monkeypatch) -> None:
    """Verify `gemini -p <prompt> --output-format json` is invoked."""
    response_json = json.dumps(
        {
            "response": "Mocked gemini answer.",
            "stats": {"input_token_count": 50, "output_token_count": 20},
        }
    )
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    captured: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer("What time?")

    assert result.error is None
    assert result.answer == "Mocked gemini answer."
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "/fake/gemini"
    assert "-p" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "What time?" in cmd


async def test_answer_with_oracle_context(monkeypatch) -> None:
    response_json = json.dumps(
        {
            "response": "Context-based answer.",
            "stats": {"input_token_count": 80, "output_token_count": 25},
        }
    )
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    captured: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer("What?", oracle_context="The context.")

    assert result.error is None
    assert result.answer == "Context-based answer."
    # The prompt arg (after -p) should contain <context>
    cmd = captured[0]
    p_idx = list(cmd).index("-p")
    prompt_arg = cmd[p_idx + 1]
    assert "<context>" in prompt_arg
    assert "The context." in prompt_arg


async def test_answer_with_seed_docs(monkeypatch) -> None:
    response_json = json.dumps({"response": "Seed-doc answer."})
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    captured: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    docs = [Document(path="/doc.md", body="Some doc body.")]
    result = await adapter.answer("What?", seed_docs=docs)

    assert result.error is None
    cmd = captured[0]
    p_idx = list(cmd).index("-p")
    prompt_arg = cmd[p_idx + 1]
    assert "[Document: /doc.md]" in prompt_arg
    assert "Some doc body." in prompt_arg


async def test_answer_non_zero_exit(monkeypatch) -> None:
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    fake_proc.communicate = AsyncMock(return_value=(b"", b"auth error"))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc))

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "exit 1" in result.error


async def test_answer_plain_text_fallback(monkeypatch) -> None:
    """Plain text stdout (no JSON) should return raw text as the answer."""
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"Plain text answer.", b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc))

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer("q")

    assert result.error is None
    assert result.answer == "Plain text answer."


async def test_answer_timeout(monkeypatch) -> None:
    fake_proc = MagicMock()
    fake_proc.returncode = None
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    async def fast_wait_for(coro, timeout):
        raise TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc))

    adapter = GeminiCliAdapter()
    adapter._gemini_bin = "/fake/gemini"
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "timeout" in result.error
    assert result.latency_ms > 0
