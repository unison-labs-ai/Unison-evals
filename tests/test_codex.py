"""CodexAdapter — mocked unit tests.

All subprocess calls are mocked via monkeypatch so the tests run without
the `codex` binary installed.

Verifies:
  * registered name is 'codex'
  * _build_prompt with and without oracle context
  * _parse_output with valid JSONL, plain text fallback, single JSON object
  * _estimate_cost with usage stats and char-count fallback
  * _format_seed_docs empty + populated
  * mutual exclusion check returns error
  * subprocess args include 'exec', '--json', '--ephemeral'
  * timeout path returns error with latency > 0
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.codex import (
    CODEX_INPUT_USD_PER_MTOK,
    CODEX_OUTPUT_USD_PER_MTOK,
    CodexAdapter,
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
    assert "codex" in REGISTRY
    assert REGISTRY["codex"] is CodexAdapter


def test_adapter_name_class_attr() -> None:
    assert CodexAdapter.name == "codex"


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


def test_parse_output_jsonl_item_completed() -> None:
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"role": "assistant", "content": "The answer is 42."},
        }
    )
    answer, usage = _parse_output(event + "\n")
    assert answer == "The answer is 42."
    assert usage == {}


def test_parse_output_jsonl_item_completed_content_blocks() -> None:
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello world."}],
            },
        }
    )
    answer, _usage = _parse_output(event + "\n")
    assert answer == "Hello world."


def test_parse_output_jsonl_turn_completed_with_usage() -> None:
    lines = [
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"role": "assistant", "content": "My final answer."},
            }
        ),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 200, "output_tokens": 50},
            }
        ),
    ]
    answer, usage = _parse_output("\n".join(lines))
    assert answer == "My final answer."
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 50


def test_parse_output_single_json_object_fallback() -> None:
    raw = json.dumps({"result": "Single JSON answer.", "usage": {}})
    answer, _usage = _parse_output(raw)
    assert answer == "Single JSON answer."


def test_parse_output_plain_text_fallback() -> None:
    answer, usage = _parse_output("Just plain text response\n")
    assert answer == "Just plain text response"
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
        1_000_000 * CODEX_INPUT_USD_PER_MTOK + 1_000_000 * CODEX_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    assert abs(cost - expected) < 0.01
    assert abs(expected - 20.0) < 0.01


def test_estimate_cost_fallback_to_chars() -> None:
    cost = _estimate_cost(
        usage={},
        prompt="x" * 4_000_000,
        answer="y" * 4_000_000,
    )
    assert abs(cost - 20.0) < 1.0


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
    adapter = CodexAdapter()
    adapter._codex_bin = "/fake/codex"
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
    """Verify `codex exec --json --ephemeral <prompt>` is invoked."""
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"role": "assistant", "content": "Mocked answer."},
        }
    )
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(event.encode(), b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    captured: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexAdapter()
    adapter._codex_bin = "/fake/codex"
    result = await adapter.answer("What time?")

    assert result.error is None
    assert result.answer == "Mocked answer."
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "/fake/codex"
    assert "exec" in cmd
    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert "What time?" in cmd


async def test_answer_with_oracle_context(monkeypatch) -> None:
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"role": "assistant", "content": "Oracle-based answer."},
        }
    )
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(event.encode(), b""))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    captured: list[tuple] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexAdapter()
    adapter._codex_bin = "/fake/codex"
    result = await adapter.answer("What?", oracle_context="The context.")

    assert result.error is None
    assert result.answer == "Oracle-based answer."
    # The prompt arg should contain <context>
    cmd = captured[0]
    prompt_arg = cmd[-1]
    assert "<context>" in prompt_arg
    assert "The context." in prompt_arg


async def test_answer_non_zero_exit(monkeypatch) -> None:
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    fake_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc))

    adapter = CodexAdapter()
    adapter._codex_bin = "/fake/codex"
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "exit 1" in result.error


async def test_answer_timeout(monkeypatch) -> None:
    fake_proc = MagicMock()
    fake_proc.returncode = None
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock()

    async def slow_communicate():
        raise TimeoutError()

    fake_proc.communicate = slow_communicate

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc))

    async def fast_wait_for(coro, timeout):
        raise TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

    adapter = CodexAdapter()
    adapter._codex_bin = "/fake/codex"
    result = await adapter.answer("q")

    assert result.answer == ""
    assert result.error is not None
    assert "timeout" in result.error
    assert result.latency_ms > 0
