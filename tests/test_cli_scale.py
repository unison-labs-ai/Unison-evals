"""Tests for --track scale CLI dispatch and validation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from unison_evals.cli import main
from unison_evals.memory_evals.adapters import BRAIN_REGISTRY
from unison_evals.memory_evals.adapters.base import BrainAdapter
from unison_evals.memory_evals.runners.scale_retrieval import ScaleRetrievalRunner
from unison_evals.types import BrainSearchResult, Document, Track

# ---------------------------------------------------------------------------
# Fake brain adapter
# ---------------------------------------------------------------------------


class _FakeScaleCLIBrain(BrainAdapter):
    name = "fake-brain-scale-cli"
    call_log: ClassVar[list[str]] = []

    async def reset(self) -> None:
        self.call_log.append("reset")

    async def ingest(self, docs: list[Document]) -> None:
        self.call_log.append(f"ingest:{len(docs)}")

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        self.call_log.append(f"search:{query}")
        return BrainSearchResult(chunks=[], latency_ms=1.0)


@pytest.fixture(autouse=True)
def _register_fake_scale_brain():
    BRAIN_REGISTRY["fake-brain-scale-cli"] = _FakeScaleCLIBrain
    yield
    BRAIN_REGISTRY.pop("fake-brain-scale-cli", None)


# ---------------------------------------------------------------------------
# Track choice validation
# ---------------------------------------------------------------------------


def test_scale_track_accepted_by_cli() -> None:
    """--track scale is a valid choice and does not produce an 'Invalid value' error."""
    runner = CliRunner()
    with patch("unison_evals.cli._run_async", new_callable=AsyncMock):
        result = runner.invoke(
            main,
            [
                "run",
                "--systems",
                "fake-brain-scale-cli",
                "--dataset",
                "msmarco",
                "--track",
                "scale",
                "--corpus",
                "test-corpus",
            ],
        )
    assert "Invalid value" not in (result.output or "")


def test_scale_track_requires_corpus_flag() -> None:
    """--track scale without --corpus raises UsageError."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "fake-brain-scale-cli",
            "--dataset",
            "msmarco",
            "--track",
            "scale",
            # No --corpus flag
        ],
    )
    assert result.exit_code != 0
    assert "corpus" in result.output.lower()


def test_scale_track_rejects_agent_adapter() -> None:
    """--track scale requires a BrainAdapter, not an agent adapter."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "claude-code",  # agent adapter, not brain adapter
            "--dataset",
            "msmarco",
            "--track",
            "scale",
            "--corpus",
            "test-corpus",
        ],
    )
    assert result.exit_code != 0


def test_unknown_track_rejected() -> None:
    """Completely unknown track values are rejected."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--systems",
            "fake-brain-scale-cli",
            "--dataset",
            "msmarco",
            "--track",
            "not-a-track",
            "--corpus",
            "test-corpus",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "not-a-track" in result.output


# ---------------------------------------------------------------------------
# Dispatch — scale track routes to ScaleRetrievalRunner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scale_track_dispatches_to_scale_runner(tmp_path: Path) -> None:
    """_run_scale wires ScaleRetrievalRunner, not BrainRetrievalRunner."""
    from unison_evals.cli import _run_scale
    from unison_evals.memory_evals.datasets.msmarco import _EMBEDDED_SMOKE_ROWS, MsMarcoDataset

    captured_runners: list[ScaleRetrievalRunner] = []
    original_run = ScaleRetrievalRunner.run

    async def _capture_run(self, questions, dataset_name="unknown"):
        captured_runners.append(self)
        async for ev in original_run(self, questions, dataset_name=dataset_name):
            yield ev

    output = tmp_path / "scale-out.json"

    with (
        patch.object(MsMarcoDataset, "_load_raw_rows", return_value=_EMBEDDED_SMOKE_ROWS),
        patch.object(ScaleRetrievalRunner, "run", _capture_run),
    ):
        await _run_scale(
            systems=["fake-brain-scale-cli"],
            dataset="msmarco",
            limit=2,
            corpus_label="test-corpus-label",
            output=output,
        )

    assert len(captured_runners) == 1
    assert isinstance(captured_runners[0], ScaleRetrievalRunner)
    assert captured_runners[0].corpus_label == "test-corpus-label"
    assert "fake-brain-scale-cli" in captured_runners[0].systems


@pytest.mark.asyncio
async def test_scale_track_raises_usage_error_for_non_scale_dataset(tmp_path: Path) -> None:
    """_run_scale raises UsageError when the dataset has no load_scale_questions()."""
    import click

    from unison_evals.cli import _run_scale

    with pytest.raises(click.UsageError, match="does not support --track scale"):
        await _run_scale(
            systems=["fake-brain-scale-cli"],
            dataset="bitempoqa",  # Does not implement load_scale_questions
            limit=5,
            corpus_label="irrelevant",
            output=tmp_path / "out.json",
        )


# ---------------------------------------------------------------------------
# Output JSON structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scale_run_writes_json_output(tmp_path: Path) -> None:
    """_run_scale writes a valid JSON artifact with summary + results keys."""
    import json

    from unison_evals.cli import _run_scale
    from unison_evals.memory_evals.datasets.msmarco import _EMBEDDED_SMOKE_ROWS, MsMarcoDataset

    output = tmp_path / "scale-run.json"

    with patch.object(MsMarcoDataset, "_load_raw_rows", return_value=_EMBEDDED_SMOKE_ROWS):
        await _run_scale(
            systems=["fake-brain-scale-cli"],
            dataset="msmarco",
            limit=3,
            corpus_label="test-corpus",
            output=output,
        )

    assert output.exists()
    payload = json.loads(output.read_text())
    assert "summary" in payload
    assert "results" in payload
    assert "exported_at" in payload

    summary = payload["summary"]
    assert summary["track"] == Track.SCALE.value
    assert summary["corpus_label"] == "test-corpus"
    assert summary["n_questions"] == 3
    assert len(summary["summaries"]) == 1

    sys_sum = summary["summaries"][0]
    assert "p99_latency_ms" in sys_sum
    assert "mean_recall_at_10" in sys_sum
    assert "mean_ndcg_at_10" in sys_sum
    assert "mean_mrr" in sys_sum
    assert "mean_hit_at_1" in sys_sum
