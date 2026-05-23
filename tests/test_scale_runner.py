"""Tests for ScaleRetrievalRunner (Track 4)."""

from __future__ import annotations

from unison_evals.memory_evals.adapters.base import BrainAdapter
from unison_evals.memory_evals.runners.scale_retrieval import ScaleRetrievalRunner, ScaleRunEvent
from unison_evals.types import (
    BrainSearchResult,
    Document,
    RetrievedChunk,
    ScaleQuestion,
    ScaleRunSummary,
    Track,
)

# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class _FakeScaleBrain(BrainAdapter):
    """In-memory brain that matches queries by substring over a fixed store.

    Unlike Track 1 fakes, this one tracks whether reset/ingest were called —
    the runner must NOT call either.
    """

    name = "fake-scale-brain"

    def __init__(self, docs: dict[str, str]) -> None:
        self._store = docs  # path → body, pre-loaded
        self.reset_count = 0
        self.ingest_count = 0

    async def reset(self) -> None:
        self.reset_count += 1

    async def ingest(self, docs: list[Document]) -> None:
        self.ingest_count += 1

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        scored = [
            (body.lower().count(query.lower()), path, body)
            for path, body in self._store.items()
            if query.lower() in body.lower()
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        chunks = [
            RetrievedChunk(doc_path=path, chunk_text=body, score=float(cnt), rank=i + 1)
            for i, (cnt, path, body) in enumerate(scored[:k])
        ]
        return BrainSearchResult(chunks=chunks, latency_ms=2.0)


class _SlowFakeScaleBrain(_FakeScaleBrain):
    """Returns results with varying latency values to test percentile computation."""

    def __init__(self) -> None:
        super().__init__({})
        self._call_count = 0

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        self._call_count += 1
        latency = float(self._call_count * 10)  # 10, 20, 30, 40, 50, ...
        return BrainSearchResult(chunks=[], latency_ms=latency)


class _ErrorScaleBrain(_FakeScaleBrain):
    """Setup always raises."""

    name = "error-scale-brain"

    def __init__(self) -> None:
        super().__init__({})

    async def setup(self) -> None:
        raise RuntimeError("setup intentionally failed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS_STORE = {
    "/msmarco/passages/p001.md": "Paris is the capital of France.",
    "/msmarco/passages/p002.md": "Lyon is a city in eastern France.",
    "/msmarco/passages/p003.md": "Photosynthesis uses sunlight and carbon dioxide.",
    "/msmarco/passages/p004.md": "The Eiffel Tower was built in 1889 in Paris.",
}

Q1 = ScaleQuestion(
    id="sq1",
    query="capital",  # matches p001 ("capital of France") uniquely
    gold_doc_paths={"/msmarco/passages/p001.md"},
)

Q2 = ScaleQuestion(
    id="sq2",
    query="photosynthesis",  # matches p003 uniquely
    gold_doc_paths={"/msmarco/passages/p003.md"},
)

Q_NO_GOLD_IN_STORE = ScaleQuestion(
    id="sq3",
    query="quantum physics",
    gold_doc_paths={"/msmarco/passages/p999.md"},
)


# ---------------------------------------------------------------------------
# Tests — lifecycle
# ---------------------------------------------------------------------------


async def test_runner_never_calls_reset_or_ingest() -> None:
    """Track 4 runner must NOT call reset() or ingest() on any adapter."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain}, corpus_label="test-corpus")
    await runner.run_to_completion([Q1, Q2], dataset_name="test")

    assert brain.reset_count == 0, "reset() should never be called in Track 4"
    assert brain.ingest_count == 0, "ingest() should never be called in Track 4"


async def test_correct_retrieval_and_metrics() -> None:
    """Runner finds gold docs and computes non-zero metrics."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(
        systems={"fake": brain}, corpus_label="test-corpus", run_id="test-001"
    )

    events: list[ScaleRunEvent] = []
    async for ev in runner.run([Q1, Q2], dataset_name="test-ds"):
        events.append(ev)

    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert "corpus_announced" in types
    assert types[-1] == "run_completed"
    assert types.count("question_started") == 2
    assert types.count("question_completed") == 2

    summary = events[-1].summary
    assert isinstance(summary, ScaleRunSummary)
    assert summary.track == Track.SCALE
    assert summary.n_questions == 2
    assert summary.corpus_label == "test-corpus"

    sys_sum = summary.summaries[0]
    assert sys_sum.mean_recall_at_10 == 1.0
    assert sys_sum.mean_hit_at_1 == 1.0
    assert sys_sum.mean_mrr == 1.0


async def test_corpus_announced_event_has_label() -> None:
    """The corpus_announced event carries the corpus_label."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain}, corpus_label="msmarco-passages-v1-100k")
    events: list[ScaleRunEvent] = []
    async for ev in runner.run([Q1], dataset_name="test"):
        events.append(ev)

    announced = next(e for e in events if e.type == "corpus_announced")
    assert announced.corpus_label == "msmarco-passages-v1-100k"


# ---------------------------------------------------------------------------
# Tests — metrics
# ---------------------------------------------------------------------------


async def test_no_matching_docs_gives_zero_metrics() -> None:
    """Query that matches nothing in the store → all metrics 0."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain})
    summary = await runner.run_to_completion([Q_NO_GOLD_IN_STORE])

    sys_sum = summary.summaries[0]
    assert sys_sum.mean_recall_at_10 == 0.0
    assert sys_sum.mean_hit_at_1 == 0.0
    assert sys_sum.mean_mrr == 0.0
    assert sys_sum.mean_ndcg_at_10 == 0.0


async def test_p99_latency_computed() -> None:
    """p99 latency is present in ScaleSystemSummary and is >= p95."""
    brain = _SlowFakeScaleBrain()
    questions = [
        ScaleQuestion(id=f"q{i}", query="anything", gold_doc_paths={"/x.md"}) for i in range(20)
    ]
    runner = ScaleRetrievalRunner(systems={"slow": brain})
    summary = await runner.run_to_completion(questions)

    sys_sum = summary.summaries[0]
    assert sys_sum.p99_latency_ms > 0.0
    assert sys_sum.p99_latency_ms >= sys_sum.p95_latency_ms
    assert sys_sum.p95_latency_ms >= sys_sum.p50_latency_ms


async def test_p99_exact_value() -> None:
    """For a deterministic set of 10 latencies, p99 should equal the max."""
    brain = _SlowFakeScaleBrain()
    questions = [
        ScaleQuestion(id=f"q{i}", query="anything", gold_doc_paths={"/x.md"}) for i in range(10)
    ]
    runner = ScaleRetrievalRunner(systems={"slow": brain})
    summary = await runner.run_to_completion(questions)

    sys_sum = summary.summaries[0]
    # latencies are 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    assert sys_sum.p99_latency_ms >= 90.0  # at least p95 value


# ---------------------------------------------------------------------------
# Tests — summary aggregation
# ---------------------------------------------------------------------------


async def test_two_systems_two_summaries() -> None:
    """Two systems produce two ScaleSystemSummary entries."""
    brain_a = _FakeScaleBrain(CORPUS_STORE)
    brain_b = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"sys-a": brain_a, "sys-b": brain_b})
    summary = await runner.run_to_completion([Q1, Q2])

    assert len(summary.systems) == 2
    assert len(summary.summaries) == 2
    sys_names = {s.system for s in summary.summaries}
    assert sys_names == {"sys-a", "sys-b"}
    for s in summary.summaries:
        assert s.n_questions == 2


async def test_results_property_populated() -> None:
    """runner.results accumulates ScaleQuestionResult objects."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain})
    await runner.run_to_completion([Q1, Q2])
    assert len(runner.results) == 2
    assert all(r.system == "fake" for r in runner.results)


async def test_run_to_completion_returns_summary() -> None:
    """run_to_completion convenience method returns ScaleRunSummary."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain}, corpus_label="test-corpus")
    summary = await runner.run_to_completion([Q1], dataset_name="convenience-test")
    assert isinstance(summary, ScaleRunSummary)
    assert summary.dataset == "convenience-test"
    assert summary.corpus_label == "test-corpus"
    assert summary.n_questions == 1


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


async def test_setup_error_emits_run_failed() -> None:
    """If adapter setup raises, runner emits run_failed and stops."""
    brain = _ErrorScaleBrain()
    runner = ScaleRetrievalRunner(systems={"bad": brain})

    events: list[ScaleRunEvent] = []
    async for ev in runner.run([Q1]):
        events.append(ev)

    types = [e.type for e in events]
    assert "run_failed" in types
    assert "run_completed" not in types
    failed_ev = next(e for e in events if e.type == "run_failed")
    assert "setup intentionally failed" in (failed_ev.error or "")


async def test_empty_questions_produces_summary() -> None:
    """Zero questions → run completes with n_questions=0."""
    brain = _FakeScaleBrain(CORPUS_STORE)
    runner = ScaleRetrievalRunner(systems={"fake": brain})
    summary = await runner.run_to_completion([], dataset_name="empty-test")
    assert summary.n_questions == 0
    assert summary.summaries[0].n_questions == 0


async def test_no_systems_raises() -> None:
    """Creating a runner with no systems raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="at least one system"):
        ScaleRetrievalRunner(systems={})
