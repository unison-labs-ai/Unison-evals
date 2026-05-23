"""Track 4 (scale) runner — queries a pre-loaded large corpus, no per-question
ingest or reset. Mirrors BrainRetrievalRunner's event/summary shape but skips
the adapter.reset() / adapter.ingest() lifecycle entirely.

Designed for large reference corpora (1M-10M docs) loaded once via a bulk
loader script, then queried thousands of times across multiple eval runs.

Extra compared to Track 1:
  - `corpus_announced` event emitted right after `run_started` so the UI can
    display "Querying msmarco-passages-v1 with N docs preloaded".
  - p99 latency in addition to p50/p95 (statistically reliable at scale).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from ...config import get_settings
from ...types import (
    BrainSearchResult,
    RunStatus,
    ScaleQuestion,
    ScaleQuestionResult,
    ScaleRunSummary,
    ScaleSystemSummary,
    Track,
)
from ..adapters.base import BrainAdapter
from ..metrics.retrieval import hit_at_k, mrr, ndcg_at_k, recall_at_k


class ScaleRunEvent(BaseModel):
    """One progress event during a Track 4 run, streamed via SSE to the UI."""

    type: Literal[
        "run_started",
        "corpus_announced",
        "question_started",
        "question_completed",
        "question_failed",
        "run_completed",
        "run_failed",
    ]
    run_id: str
    system: str | None = None
    question_id: str | None = None
    questions_total: int | None = None
    questions_done: int | None = None
    corpus_label: str | None = None
    result: ScaleQuestionResult | None = None
    summary: ScaleRunSummary | None = None
    error: str | None = None


class ScaleRetrievalRunner:
    """Track 4 — query-only retrieval against a pre-loaded corpus.

    .. deprecated::
        Use ``BrainRetrievalRunner(mode=BrainMode.WARM)`` instead.
        ScaleRetrievalRunner is kept for back-compat only and will not receive
        new features. The WARM sub-mode of BrainRetrievalRunner is the
        canonical path forward for pre-loaded corpus benchmarks.

    The corpus MUST already be loaded into the adapter's backing store before
    calling run(). This runner never calls adapter.reset() or adapter.ingest().

    Use:
        runner = ScaleRetrievalRunner(
            systems={"pgvector-naive": adapter},
            corpus_label="msmarco-passages-v1-100k",
        )
        async for event in runner.run(questions):
            ...
        # or:
        summary = await runner.run_to_completion(questions)
    """

    def __init__(
        self,
        systems: dict[str, BrainAdapter],
        corpus_label: str = "unknown-corpus",
        run_id: str | None = None,
    ) -> None:
        if not systems:
            raise ValueError("ScaleRetrievalRunner needs at least one system")
        self.systems = systems
        self.corpus_label = corpus_label
        self.run_id = run_id or _new_run_id()
        self.settings = get_settings()
        self._results: list[ScaleQuestionResult] = []

    async def run(
        self,
        questions: Iterable[ScaleQuestion],
        dataset_name: str = "unknown",
    ) -> AsyncIterator[ScaleRunEvent]:
        questions = list(questions)
        n_q = len(questions)
        n_systems = len(self.systems)
        total_pairs = n_q * n_systems
        started_at = datetime.now(UTC)

        yield ScaleRunEvent(
            type="run_started",
            run_id=self.run_id,
            questions_total=total_pairs,
            questions_done=0,
        )

        # Announce corpus so the UI can display the label.
        yield ScaleRunEvent(
            type="corpus_announced",
            run_id=self.run_id,
            corpus_label=self.corpus_label,
        )

        # One-time setup for every adapter (open connections, etc.).
        try:
            for sys_name, adapter in self.systems.items():
                try:
                    await adapter.setup()
                except Exception as e:
                    yield ScaleRunEvent(
                        type="run_failed",
                        run_id=self.run_id,
                        error=f"Setup failed for {sys_name}: {e}",
                    )
                    return
        except Exception as e:
            yield ScaleRunEvent(
                type="run_failed",
                run_id=self.run_id,
                error=f"Unexpected setup error: {e}",
            )
            return

        # Bounded-concurrency loop over (question, system) pairs.
        sem = asyncio.Semaphore(self.settings.max_concurrent_questions)
        done_count = 0

        # Emit question_started placeholders up front so the UI renders rows.
        for q in questions:
            for sys_name in self.systems:
                yield ScaleRunEvent(
                    type="question_started",
                    run_id=self.run_id,
                    system=sys_name,
                    question_id=q.id,
                    questions_total=total_pairs,
                    questions_done=done_count,
                )

        async def _one(
            q: ScaleQuestion, sys_name: str, adapter: BrainAdapter
        ) -> tuple[ScaleQuestion, str, ScaleQuestionResult]:
            async with sem:
                search_result: BrainSearchResult
                try:
                    # Track 4: NO reset(), NO ingest() — corpus is pre-loaded.
                    search_result = await adapter.search(q.query, k=10)
                except Exception as e:
                    search_result = BrainSearchResult(error=str(e))

                if search_result.error:
                    m = _zero_metrics()
                else:
                    retrieved_paths = [c.doc_path for c in search_result.chunks]
                    m = {
                        "recall_at_10": recall_at_k(retrieved_paths, q.gold_doc_paths, 10),
                        "ndcg_at_10": ndcg_at_k(retrieved_paths, q.gold_doc_paths, 10),
                        "mrr": mrr(retrieved_paths, q.gold_doc_paths),
                        "hit_at_1": hit_at_k(retrieved_paths, q.gold_doc_paths, 1),
                    }

                qr = ScaleQuestionResult(
                    question_id=q.id,
                    system=sys_name,
                    search_result=search_result,
                    metrics=m,
                    error=search_result.error,
                )
                return q, sys_name, qr

        tasks = [
            asyncio.create_task(_one(q, sys_name, adapter))
            for q in questions
            for sys_name, adapter in self.systems.items()
        ]

        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    q, sys_name, qr = await coro
                    self._results.append(qr)
                    done_count += 1
                    yield ScaleRunEvent(
                        type="question_completed",
                        run_id=self.run_id,
                        system=sys_name,
                        question_id=q.id,
                        result=qr,
                        questions_total=total_pairs,
                        questions_done=done_count,
                    )
                except Exception as e:
                    logger.exception("Scale question task crashed")
                    done_count += 1
                    yield ScaleRunEvent(
                        type="question_failed",
                        run_id=self.run_id,
                        questions_total=total_pairs,
                        questions_done=done_count,
                        error=str(e),
                    )
        finally:
            for sys_name, adapter in self.systems.items():
                try:
                    await adapter.teardown()
                except Exception as e:
                    logger.warning("teardown failed for {}: {}", sys_name, e)

        finished_at = datetime.now(UTC)
        summary = self._build_summary(
            dataset_name=dataset_name,
            n_questions=n_q,
            started_at=started_at,
            finished_at=finished_at,
        )
        yield ScaleRunEvent(
            type="run_completed",
            run_id=self.run_id,
            summary=summary,
        )

    async def run_to_completion(
        self,
        questions: Iterable[ScaleQuestion],
        dataset_name: str = "unknown",
    ) -> ScaleRunSummary:
        """Convenience for callers that don't want event streaming."""
        summary: ScaleRunSummary | None = None
        async for ev in self.run(questions, dataset_name=dataset_name):
            if ev.type == "run_completed" and ev.summary is not None:
                summary = ev.summary
            elif ev.type == "run_failed":
                raise RuntimeError(ev.error or "run failed")
        if summary is None:
            raise RuntimeError("Run produced no summary")
        return summary

    def _build_summary(
        self,
        dataset_name: str,
        n_questions: int,
        started_at: datetime,
        finished_at: datetime,
    ) -> ScaleRunSummary:
        per_system: dict[str, list[ScaleQuestionResult]] = {s: [] for s in self.systems}
        for r in self._results:
            per_system.setdefault(r.system, []).append(r)

        summaries: list[ScaleSystemSummary] = []
        total_cost = 0.0

        for sys_name, rows in per_system.items():
            latencies = [r.search_result.latency_ms for r in rows] or [0.0]
            cost_for_sys = sum(r.search_result.cost_usd for r in rows)
            total_cost += cost_for_sys

            _rows = rows

            def _mean(key: str, _r: list[ScaleQuestionResult] = _rows) -> float:
                if not _r:
                    return 0.0
                return sum(r.metrics.get(key, 0.0) for r in _r) / len(_r)

            summaries.append(
                ScaleSystemSummary(
                    system=sys_name,
                    n_questions=len(rows),
                    mean_recall_at_10=_mean("recall_at_10"),
                    mean_ndcg_at_10=_mean("ndcg_at_10"),
                    mean_mrr=_mean("mrr"),
                    mean_hit_at_1=_mean("hit_at_1"),
                    total_cost_usd=cost_for_sys,
                    avg_latency_ms=sum(latencies) / len(latencies),
                    p50_latency_ms=_percentile(latencies, 50),
                    p95_latency_ms=_percentile(latencies, 95),
                    p99_latency_ms=_percentile(latencies, 99),
                )
            )

        return ScaleRunSummary(
            run_id=self.run_id,
            dataset=dataset_name,
            track=Track.SCALE,
            systems=list(self.systems.keys()),
            n_questions=n_questions,
            corpus_label=self.corpus_label,
            started_at=started_at,
            finished_at=finished_at,
            total_cost_usd=total_cost,
            summaries=summaries,
        )

    @property
    def results(self) -> list[ScaleQuestionResult]:
        return list(self._results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_metrics() -> dict[str, float]:
    return {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0}


def _new_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"scale-{ts}-{uuid.uuid4().hex[:6]}"


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# Re-export for type checkers
_ = RunStatus
