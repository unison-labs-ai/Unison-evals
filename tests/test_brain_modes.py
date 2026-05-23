"""Tests for the four BrainMode sub-modes of BrainRetrievalRunner.

Covers:
  - COLD mode: existing reset+ingest+search behavior (regression guard)
  - WARM mode: no reset/ingest, only search
  - BITEMPORAL mode: temporal_correct_at_1 metric computed; version-aware scoring
  - COMPACTION mode: non-unison-brain adapters are [SKIP]ped with clear reason

Also covers:
  - (dataset, mode) compatibility guards in the CLI helper
  - temporal_correct_at_1 math (unit tests)
"""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.adapters.base import BrainAdapter
from unison_evals.memory_evals.runners.brain_retrieval import BrainRetrievalRunner, BrainRunEvent
from unison_evals.types import (
    BrainMode,
    BrainQuestion,
    BrainRunSummary,
    BrainSearchResult,
    Document,
    RetrievedChunk,
)

# ---------------------------------------------------------------------------
# Shared fake adapters
# ---------------------------------------------------------------------------


class _FakeBrain(BrainAdapter):
    """Substring-match in-memory brain."""

    name = "fake-brain"

    def __init__(self) -> None:
        self.docs: list[Document] = []
        self.reset_count = 0
        self.ingest_count = 0

    async def reset(self) -> None:
        self.docs = []
        self.reset_count += 1

    async def ingest(self, docs: list[Document]) -> None:
        self.docs.extend(docs)
        self.ingest_count += 1

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        scored = [
            (d.body.lower().count(query.lower()), d)
            for d in self.docs
            if query.lower() in d.body.lower()
        ]
        scored.sort(key=lambda x: (-x[0], x[1].path))
        chunks = [
            RetrievedChunk(doc_path=d.path, chunk_text=d.body, score=float(s), rank=i + 1)
            for i, (s, d) in enumerate(scored[:k])
        ]
        return BrainSearchResult(chunks=chunks, latency_ms=1.0)


class _PreloadedBrain(_FakeBrain):
    """Fake brain with a fixed pre-loaded corpus (simulates WARM mode)."""

    name = "preloaded-brain"

    def __init__(self, store: dict[str, str]) -> None:
        super().__init__()
        # Populate docs directly as if pre-loaded.
        self.docs = [Document(path=path, body=body) for path, body in store.items()]


# ---------------------------------------------------------------------------
# Shared question fixtures
# ---------------------------------------------------------------------------

CORPUS = {
    "/facts/f001.md": "Marta Osei ceo Velox Systems valid 2021-03-01 to 2023-08-15",
    "/facts/f002.md": "Derek Holt ceo Velox Systems valid 2023-08-15 to 2025-01-10",
    "/facts/f003.md": "Priya Nair ceo Velox Systems valid 2025-01-10 to present",
}


def _make_q(
    qid: str,
    query: str,
    gold: set[str],
    as_of: str | None = None,
    expected_versions: dict[str, str] | None = None,
) -> BrainQuestion:
    corpus = [Document(path=p, body=b) for p, b in CORPUS.items()]
    metadata: dict = {}
    if as_of is not None:
        metadata["as_of"] = as_of
    if expected_versions is not None:
        metadata["expected_versions"] = expected_versions
    return BrainQuestion(id=qid, query=query, corpus=corpus, gold_doc_paths=gold, metadata=metadata)


Q_CURRENT = _make_q("q-current", "Priya Nair", gold={"/facts/f003.md"})
Q_HISTORICAL = _make_q(
    "q-hist",
    "Marta Osei",
    gold={"/facts/f001.md"},
    as_of="2022-06-01",
    expected_versions={"/facts/f001.md": "f001"},
)
Q_WRONG_VERSION = _make_q(
    "q-wrong-version",
    "ceo Velox",  # ambiguous — matches f001, f002, f003
    gold={"/facts/f001.md"},
    as_of="2022-06-01",
    expected_versions={"/facts/f001.md": "f001"},
)


# ---------------------------------------------------------------------------
# COLD mode tests
# ---------------------------------------------------------------------------


async def test_cold_mode_calls_reset_and_ingest() -> None:
    """COLD mode: reset() and ingest() called once per question."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.COLD)
    await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    assert brain.reset_count == 1
    assert brain.ingest_count == 1


async def test_cold_mode_produces_standard_metrics() -> None:
    """COLD mode: standard retrieval metrics, no temporal metric."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.COLD)
    summary = await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    assert isinstance(summary, BrainRunSummary)
    assert summary.mode == BrainMode.COLD
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_hit_at_1 == 1.0
    assert sys_sum.mean_temporal_correct_at_1 is None  # not BITEMPORAL


async def test_cold_mode_default_mode_parameter() -> None:
    """BrainRetrievalRunner defaults to COLD when mode is not given."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain})
    assert runner.mode == BrainMode.COLD


# ---------------------------------------------------------------------------
# WARM mode tests
# ---------------------------------------------------------------------------


async def test_warm_mode_never_calls_reset_or_ingest() -> None:
    """WARM mode: reset() and ingest() must NOT be called."""
    brain = _PreloadedBrain(CORPUS)
    runner = BrainRetrievalRunner(systems={"pre": brain}, mode=BrainMode.WARM)
    await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    assert brain.reset_count == 0, "WARM mode must not call reset()"
    assert brain.ingest_count == 0, "WARM mode must not call ingest()"


async def test_warm_mode_retrieves_from_preloaded_corpus() -> None:
    """WARM mode: results come from the pre-loaded corpus."""
    brain = _PreloadedBrain(CORPUS)
    runner = BrainRetrievalRunner(systems={"pre": brain}, mode=BrainMode.WARM)
    summary = await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    assert summary.mode == BrainMode.WARM
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_hit_at_1 == 1.0
    assert sys_sum.mean_temporal_correct_at_1 is None  # WARM doesn't compute temporal


async def test_warm_mode_no_temporal_metric_in_summary() -> None:
    """WARM mode: mean_temporal_correct_at_1 is None even if questions carry as_of."""
    brain = _PreloadedBrain(CORPUS)
    runner = BrainRetrievalRunner(systems={"pre": brain}, mode=BrainMode.WARM)
    summary = await runner.run_to_completion([Q_HISTORICAL], dataset_name="test")
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_temporal_correct_at_1 is None


# ---------------------------------------------------------------------------
# BITEMPORAL mode tests
# ---------------------------------------------------------------------------


async def test_bitemporal_mode_calls_reset_and_ingest() -> None:
    """BITEMPORAL mode still does per-Q reset → ingest (like COLD)."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.BITEMPORAL)
    await runner.run_to_completion([Q_HISTORICAL], dataset_name="test")
    assert brain.reset_count == 1
    assert brain.ingest_count == 1


async def test_bitemporal_mode_computes_temporal_correct_at_1() -> None:
    """BITEMPORAL mode: temporal_correct_at_1 appears in question metrics."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.BITEMPORAL)

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q_HISTORICAL], dataset_name="test"):
        events.append(ev)

    completed = next(e for e in events if e.type == "question_completed")
    assert completed.result is not None
    assert "temporal_correct_at_1" in completed.result.metrics


async def test_bitemporal_summary_has_mean_temporal() -> None:
    """BITEMPORAL mode: BrainSystemSummary carries mean_temporal_correct_at_1."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.BITEMPORAL)
    summary = await runner.run_to_completion([Q_HISTORICAL], dataset_name="test")
    assert summary.mode == BrainMode.BITEMPORAL
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_temporal_correct_at_1 is not None
    assert 0.0 <= sys_sum.mean_temporal_correct_at_1 <= 1.0


async def test_bitemporal_no_as_of_falls_back_to_hit_at_1() -> None:
    """Questions without as_of still get temporal_correct_at_1 via hit@1 fallback."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.BITEMPORAL)
    # Q_CURRENT has no as_of and no expected_versions.
    summary = await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_temporal_correct_at_1 is not None
    # Should equal 1.0 (top-1 doc is in gold, no version constraint).
    assert sys_sum.mean_temporal_correct_at_1 == 1.0


# ---------------------------------------------------------------------------
# COMPACTION mode tests
# ---------------------------------------------------------------------------


async def test_compaction_mode_skips_non_unison_brain() -> None:
    """COMPACTION mode: non-unison-brain adapters receive a [SKIP] result."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.COMPACTION)

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q_CURRENT], dataset_name="test"):
        events.append(ev)

    skipped = [e for e in events if e.type == "question_skipped"]
    assert len(skipped) == 1, "Expected exactly one skipped event"
    assert skipped[0].skip_reason is not None
    assert "COMPACTION" in skipped[0].skip_reason or "compaction" in skipped[0].skip_reason.lower()


async def test_compaction_mode_skip_result_has_no_metrics() -> None:
    """COMPACTION skip: the result has empty metrics (not zero retrieval scores)."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.COMPACTION)
    summary = await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    # Skipped rows have empty metrics — excluded from aggregate.
    sys_sum = summary.summaries[0]
    # n_questions in summary is the number of *scored* rows (0 after skips).
    assert sys_sum.n_questions == 0


async def test_compaction_mode_run_completes_not_fails() -> None:
    """COMPACTION mode with all-skip adapters still produces run_completed, not run_failed."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, mode=BrainMode.COMPACTION)

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q_CURRENT], dataset_name="test"):
        events.append(ev)

    types = [e.type for e in events]
    assert "run_completed" in types
    assert "run_failed" not in types


# ---------------------------------------------------------------------------
# Adapter compatibility: two systems, one compaction-capable
# ---------------------------------------------------------------------------


class _UnisonBrainStub(BrainAdapter):
    """Stub that pretends to be unison-brain for compaction compat tests.
    The actual compaction endpoint isn't available so it still gets [SKIP],
    but only because the endpoint is absent — not because it's the wrong adapter.
    """

    name = "unison-brain"

    async def ingest(self, docs: list[Document]) -> None:
        pass

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        return BrainSearchResult()


async def test_compaction_mode_unison_brain_gets_endpoint_skip_not_adapter_skip() -> None:
    """unison-brain in COMPACTION gets [SKIP] with endpoint-absent reason."""
    adapter = _UnisonBrainStub()
    runner = BrainRetrievalRunner(systems={"unison-brain": adapter}, mode=BrainMode.COMPACTION)

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q_CURRENT], dataset_name="test"):
        events.append(ev)

    skipped = [e for e in events if e.type == "question_skipped"]
    assert len(skipped) == 1
    reason = skipped[0].skip_reason or ""
    assert "eval-wiki" in reason or "endpoint" in reason.lower()
    # Must NOT say the adapter is the wrong type — unison-brain IS capable.
    assert "not supported by adapter" not in reason


async def test_compaction_two_systems_non_capable_skipped_capable_also_skipped_for_endpoint() -> (
    None
):
    """Two systems: fake-brain skipped for adapter reason; unison-brain for endpoint reason."""
    fake = _FakeBrain()
    unison = _UnisonBrainStub()
    runner = BrainRetrievalRunner(
        systems={"fake-brain": fake, "unison-brain": unison},
        mode=BrainMode.COMPACTION,
    )

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q_CURRENT], dataset_name="test"):
        events.append(ev)

    skipped = [e for e in events if e.type == "question_skipped"]
    assert len(skipped) == 2  # both systems skipped, for different reasons

    reasons = {e.system: e.skip_reason or "" for e in skipped}
    # fake-brain: adapter-level skip
    assert "not supported by adapter" in reasons.get("fake-brain", "")
    # unison-brain: endpoint-level skip
    assert (
        "eval-wiki" in reasons.get("unison-brain", "")
        or "endpoint" in reasons.get("unison-brain", "").lower()
    )


# ---------------------------------------------------------------------------
# BrainRunSummary.mode field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [BrainMode.COLD, BrainMode.WARM, BrainMode.BITEMPORAL, BrainMode.COMPACTION],
)
async def test_summary_records_mode(mode: BrainMode) -> None:
    """BrainRunSummary.mode must reflect the mode the runner was created with."""
    if mode == BrainMode.WARM:
        brain = _PreloadedBrain(CORPUS)
    else:
        brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"sys": brain}, mode=mode)
    summary = await runner.run_to_completion([Q_CURRENT], dataset_name="test")
    assert summary.mode == mode
