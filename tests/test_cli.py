"""Tests for the CLI — track dispatch and validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from unison_evals.cli import main
from unison_evals.memory_evals.adapters import REGISTRY as ADAPTER_REGISTRY
from unison_evals.memory_evals.adapters.base import AgentAdapter
from unison_evals.types import AdapterResult


class _FakeAgentAdapter(AgentAdapter):
    name = "fake-agent-cli"

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: object = None,
    ) -> AdapterResult:
        return AdapterResult(answer="fake", cost_usd=0.0, latency_ms=1.0)


@pytest.fixture(autouse=True)
def _register_fake_agent():
    ADAPTER_REGISTRY["fake-agent-cli"] = _FakeAgentAdapter
    yield
    ADAPTER_REGISTRY.pop("fake-agent-cli", None)


# ---------------------------------------------------------------------------
# --track validation
# ---------------------------------------------------------------------------


def test_run_rejects_unknown_track() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "fake-agent-cli",
            "--dataset",
            "longmemeval",
            "--track",
            "bogus-track",
        ],
    )
    assert result.exit_code != 0
    assert "bogus-track" in result.output or "Invalid value" in result.output


def test_run_accepts_agent_oracle_track() -> None:
    runner = CliRunner()
    with patch("unison_evals.cli._run_async", new_callable=AsyncMock) as mock_run:
        result = runner.invoke(
            main,
            [
                "run",
                "--systems",
                "fake-agent-cli",
                "--dataset",
                "longmemeval",
                "--track",
                "agent-oracle",
            ],
        )
    assert result.exit_code == 0 or mock_run.called


def test_run_accepts_agent_e2e_track() -> None:
    runner = CliRunner()
    with patch("unison_evals.cli._run_async", new_callable=AsyncMock):
        result = runner.invoke(
            main,
            [
                "run",
                "--systems",
                "fake-agent-cli",
                "--dataset",
                "longmemeval",
                "--track",
                "agent-e2e",
            ],
        )
    assert "Invalid value" not in (result.output or "")


def test_agent_oracle_rejects_unknown_system() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "no-such-system",
            "--dataset",
            "longmemeval",
            "--track",
            "agent-oracle",
        ],
    )
    assert result.exit_code != 0


def test_no_judge_rejected_for_agent_e2e() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "fake-agent-cli",
            "--dataset",
            "longmemeval",
            "--track",
            "agent-e2e",
            "--no-judge",
        ],
    )
    assert result.exit_code != 0
    assert "no-judge" in result.output.lower() or "not supported" in result.output.lower()
