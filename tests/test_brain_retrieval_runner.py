"""End-to-end tests for BrainRetrievalRunner (Track 1)."""

from __future__ import annotations

from unison_evals.memory_evals.adapters.base import BrainAdapter
from unison_evals.memory_evals.runners.brain_retrieval import BrainRetrievalRunner, BrainRunEvent
from unison_evals.types import (
    BrainQuestion,
    BrainRunSummary,
    BrainSearchResult,
    Document,
    RetrievedChunk,
)

# ---------------------------------------------------------------------------
# Helpers — fake adapters
# ---------------------------------------------------------------------------


class _FakeBrain(BrainAdapter):
    """In-memory substring-match brain adapter for testing."""

    name = "fake-brain"

    def __init__(self) -> None:
        self.docs: list[Document] = []
        self._reset_count = 0

    async def reset(self) -> None:
        self.docs = []
        self._reset_count += 1

    async def ingest(self, docs: list[Document]) -> None:
        self.docs.extend(docs)

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        scored = [
            (d.body.lower().count(query.lower()), d)
            for d in self.docs
            if query.lower() in d.body.lower()
        ]
        scored.sort(key=lambda x: (-x[0], x[1].path))
        chunks = [
            RetrievedChunk(doc_path=d.path, chunk_text=d.body, score=float(score), rank=i + 1)
            for i, (score, d) in enumerate(scored[:k])
        ]
        return BrainSearchResult(chunks=chunks, latency_ms=1.0)


class _ErrorBrain(BrainAdapter):
    """Brain adapter whose setup raises an error."""

    name = "error-brain"

    async def setup(self) -> None:
        raise RuntimeError("setup intentionally failed")

    async def ingest(self, docs: list[Document]) -> None:
        pass

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        return BrainSearchResult()


class _SearchErrorBrain(_FakeBrain):
    """Brain adapter whose search always raises."""

    name = "search-error-brain"

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        raise RuntimeError("search intentionally failed")


class _EmptyBrain(_FakeBrain):
    """Brain that ingests nothing and always returns empty results."""

    name = "empty-brain"

    async def ingest(self, docs: list[Document]) -> None:
        pass

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        return BrainSearchResult(latency_ms=0.5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question(
    qid: str, query: str, corpus_bodies: dict[str, str], gold: set[str]
) -> BrainQuestion:
    corpus = [Document(path=path, body=body) for path, body in corpus_bodies.items()]
    return BrainQuestion(id=qid, query=query, corpus=corpus, gold_doc_paths=gold)


Q1 = _make_question(
    "q1",
    "apple",
    {"/a.md": "apple pie recipe", "/b.md": "banana bread", "/c.md": "apple cider"},
    gold={"/a.md", "/c.md"},
)

Q2 = _make_question(
    "q2",
    "banana",
    {"/a.md": "banana split dessert", "/b.md": "cherry tart"},
    gold={"/a.md"},
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_end_to_end_two_questions_one_system() -> None:
    """Both questions run, result list has 2 entries, summary aggregates correctly."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain}, run_id="test-run-001")

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q1, Q2], dataset_name="test-ds"):
        events.append(ev)

    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    assert types.count("question_started") == 2
    assert types.count("question_completed") == 2

    summary = events[-1].summary
    assert isinstance(summary, BrainRunSummary)
    assert summary.n_questions == 2
    assert len(summary.summaries) == 1
    sys_sum = summary.summaries[0]
    assert sys_sum.system == "fake"
    assert sys_sum.n_questions == 2
    # Q1: retrieved [/a.md, /c.md] → hit@1=1, recall@10=1, mrr=1
    # Q2: retrieved [/a.md] → hit@1=1, recall@10=1, mrr=1
    assert sys_sum.mean_hit_at_1 == 1.0
    assert sys_sum.mean_recall_at_10 == 1.0
    assert sys_sum.mean_mrr == 1.0


async def test_end_to_end_two_systems() -> None:
    """Two systems, two questions → 4 (system, question) pairs, two system summaries."""
    brain_a = _FakeBrain()
    brain_b = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"sys-a": brain_a, "sys-b": brain_b})

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q1, Q2]):
        events.append(ev)

    summary = events[-1].summary
    assert isinstance(summary, BrainRunSummary)
    assert len(summary.systems) == 2
    assert len(summary.summaries) == 2
    # Both fake brains behave identically → same metrics
    for sys_sum in summary.summaries:
        assert sys_sum.n_questions == 2


async def test_adapter_setup_error_emits_run_failed() -> None:
    """If setup raises, runner emits run_failed and stops early."""
    brain = _ErrorBrain()
    runner = BrainRetrievalRunner(systems={"bad": brain})

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q1]):
        events.append(ev)

    types = [e.type for e in events]
    assert "run_failed" in types
    assert "run_completed" not in types
    failed_ev = next(e for e in events if e.type == "run_failed")
    assert "setup intentionally failed" in (failed_ev.error or "")


async def test_search_error_scores_zero() -> None:
    """If search raises, the question still completes with all-zero metrics."""
    brain = _SearchErrorBrain()
    runner = BrainRetrievalRunner(systems={"search-error": brain})

    events: list[BrainRunEvent] = []
    async for ev in runner.run([Q1]):
        events.append(ev)

    types = [e.type for e in events]
    assert "run_completed" in types
    assert "question_completed" in types

    completed = next(e for e in events if e.type == "question_completed")
    assert completed.result is not None
    m = completed.result.metrics
    assert m["recall_at_10"] == 0.0
    assert m["ndcg_at_10"] == 0.0
    assert m["mrr"] == 0.0
    assert m["hit_at_1"] == 0.0
    assert completed.result.error is not None


async def test_empty_corpus_recall_zero() -> None:
    """Empty corpus → ingest is a no-op, search returns empty, all metrics zero."""
    q_empty = BrainQuestion(
        id="qe",
        query="anything",
        corpus=[],
        gold_doc_paths={"/relevant.md"},
    )
    brain = _EmptyBrain()
    runner = BrainRetrievalRunner(systems={"empty": brain})

    events: list[BrainRunEvent] = []
    async for ev in runner.run([q_empty]):
        events.append(ev)

    summary = events[-1].summary
    assert isinstance(summary, BrainRunSummary)
    sys_sum = summary.summaries[0]
    assert sys_sum.mean_recall_at_10 == 0.0
    assert sys_sum.mean_hit_at_1 == 0.0
    assert sys_sum.mean_mrr == 0.0


async def test_run_to_completion_convenience() -> None:
    """run_to_completion returns BrainRunSummary without streaming."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain})
    summary = await runner.run_to_completion([Q1], dataset_name="convenience-test")
    assert isinstance(summary, BrainRunSummary)
    assert summary.dataset == "convenience-test"
    assert summary.n_questions == 1


async def test_reset_called_between_questions() -> None:
    """The adapter's reset() is called once per question (not once globally)."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain})
    await runner.run_to_completion([Q1, Q2])
    # reset() is called before each (system, question) pair → should be 2
    assert brain._reset_count == 2


async def test_results_property_after_run() -> None:
    """runner.results gives the accumulated BrainQuestionResult list."""
    brain = _FakeBrain()
    runner = BrainRetrievalRunner(systems={"fake": brain})
    await runner.run_to_completion([Q1, Q2])
    assert len(runner.results) == 2
    assert all(r.system == "fake" for r in runner.results)
