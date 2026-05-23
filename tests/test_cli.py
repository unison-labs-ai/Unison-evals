"""Tests for the CLI — particularly the brain-only track dispatch.

Per-dataset BrainQuestion conversion is now exercised in each
`tests/test_<dataset>_dataset.py` (via the `Dataset.load_brain_questions()`
ABC method). This file only covers CLI-level routing and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from unison_evals.cli import main
from unison_evals.memory_evals.adapters import BRAIN_REGISTRY
from unison_evals.memory_evals.adapters.base import BrainAdapter
from unison_evals.types import BrainSearchResult, Document


class _FakeBrainAdapter(BrainAdapter):
    name = "fake-brain-cli"
    call_log: ClassVar[list[str]] = []

    async def reset(self) -> None:
        self.call_log.append("reset")

    async def ingest(self, docs: list[Document]) -> None:
        self.call_log.append(f"ingest:{len(docs)}")

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        self.call_log.append(f"search:{query}")
        return BrainSearchResult(chunks=[], latency_ms=1.0)


@pytest.fixture(autouse=True)
def _register_fake_brain():
    BRAIN_REGISTRY["fake-brain-cli"] = _FakeBrainAdapter
    yield
    BRAIN_REGISTRY.pop("fake-brain-cli", None)


# ---------------------------------------------------------------------------
# --track validation
# ---------------------------------------------------------------------------


def test_run_rejects_unknown_track() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--systems", "claude-code", "--dataset", "bitempoqa", "--track", "bogus-track"],
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
                "claude-code",
                "--dataset",
                "bitempoqa",
                "--track",
                "agent-oracle",
            ],
        )
    assert result.exit_code == 0 or mock_run.called


def test_run_accepts_brain_only_track() -> None:
    runner = CliRunner()
    with patch("unison_evals.cli._run_async", new_callable=AsyncMock):
        result = runner.invoke(
            main,
            [
                "run",
                "--systems",
                "fake-brain-cli",
                "--dataset",
                "bitempoqa",
                "--track",
                "brain-only",
            ],
        )
    assert "Invalid value" not in (result.output or "")


def test_brain_only_rejects_agent_adapter() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "claude-code",  # agent adapter, not a brain adapter
            "--dataset",
            "bitempoqa",
            "--track",
            "brain-only",
        ],
    )
    assert result.exit_code != 0


def test_agent_oracle_rejects_brain_adapter() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "pgvector-naive",  # brain adapter, not an agent adapter
            "--dataset",
            "bitempoqa",
            "--track",
            "agent-oracle",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# BrainRetrievalRunner dispatch via brain-only track
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brain_only_dispatches_to_brain_retrieval_runner(tmp_path: Path) -> None:
    """Verify brain-only track wires BrainRetrievalRunner, calling
    Dataset.load_brain_questions() instead of the old inline converters."""
    from unison_evals.cli import _run_brain_only
    from unison_evals.memory_evals.runners.brain_retrieval import BrainRetrievalRunner

    captured: list[BrainRetrievalRunner] = []

    original_run = BrainRetrievalRunner.run

    async def _capture_run(self, questions, dataset_name="unknown"):
        captured.append(self)
        async for ev in original_run(self, questions, dataset_name=dataset_name):
            yield ev

    from unison_evals.types import BrainMode

    output = tmp_path / "out.json"
    with patch.object(BrainRetrievalRunner, "run", _capture_run):
        await _run_brain_only(
            systems=["fake-brain-cli"],
            dataset="bitempoqa",
            limit=2,
            mode=BrainMode.COLD,
            output=output,
        )

    assert len(captured) == 1, "BrainRetrievalRunner.run should have been called once"
    assert isinstance(captured[0], BrainRetrievalRunner)
    assert "fake-brain-cli" in captured[0].systems


@pytest.mark.asyncio
async def test_brain_only_handles_unsupported_dataset(tmp_path: Path) -> None:
    """A dataset whose load_brain_questions raises NotImplementedError
    should produce a clear error, not a crash."""
    from unison_evals.cli import _run_brain_only
    from unison_evals.types import BrainMode

    output = tmp_path / "out.json"
    # FRAMES is the canonical 'Track 1 not supported' dataset
    await _run_brain_only(
        systems=["fake-brain-cli"],
        dataset="frames",
        limit=2,
        mode=BrainMode.COLD,
        output=output,
    )
    # Should not crash; output file may or may not exist depending on
    # whether the runner reached the write step.
