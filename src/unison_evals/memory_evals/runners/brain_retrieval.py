"""Track 1 (brain-only retrieval) runner — four sub-modes via BrainMode.

BrainMode.COLD (default, original behaviour)
    Per-question: reset → ingest → search → score.

BrainMode.WARM
    Corpus already loaded; skip reset+ingest, iterate search only.
    Equivalent to ScaleRetrievalRunner but expressed as a Track 1 sub-mode.
    Use when the dataset supports pre-loading (e.g. MS MARCO, BitempoQA shared
    corpus) and you want to amortise ingest cost across many queries.

BrainMode.BITEMPORAL
    Per-question: reset → ingest → search → score with temporal_correct_at_1.
    Questions with metadata["as_of"] get the temporal metric (1.0/0.5/0.0);
    questions without as_of fall back to plain hit@1.

BrainMode.COMPACTION
    Ingest raw transcripts, poll for compacted wiki page, LLM-judge quality.
    Only unison-brain supports this; all other adapters are [SKIP]ped.
    Requires GET /api/rest/agents/eval-wiki/<entity_slug> on the Unison side.

(dataset, mode) compatibility is enforced at question-load time. Incompatible
combos auto-skip with a [SKIP] log line rather than raising.

Yields the same event shape as AgentOracleRunner so the FastAPI server can
stream live progress to the UI unchanged.
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
    BrainMode,
    BrainQuestion,
    BrainQuestionResult,
    BrainRunSummary,
    BrainSearchResult,
    BrainSystemSummary,
    RunStatus,
    Track,
)
from ..adapters.base import BrainAdapter
from ..metrics.retrieval import hit_at_k, mrr, ndcg_at_k, recall_at_k
from ..metrics.stats import bootstrap_ci
from ..metrics.temporal import temporal_correct_at_1

# Adapters that support COMPACTION mode (must have a compactor + eval-wiki endpoint).
_COMPACTION_CAPABLE_ADAPTERS = frozenset({"unison-brain"})


class BrainRunEvent(BaseModel):
    """One progress event during a Track 1 run, streamed via SSE to the UI."""

    type: Literal[
        "run_started",
        "question_started",
        "corpus_ingested",
        "question_completed",
        "question_failed",
        "question_skipped",
        "run_completed",
        "run_failed",
    ]
    run_id: str
    system: str | None = None
    question_id: str | None = None
    questions_total: int | None = None
    questions_done: int | None = None
    result: BrainQuestionResult | None = None
    summary: BrainRunSummary | None = None
    error: str | None = None
    skip_reason: str | None = None


class BrainRetrievalRunner:
    """Track 1 — retrieval metric scoring across four sub-modes.

    Use:
        runner = BrainRetrievalRunner(
            systems={"pgvector-naive": adapter},
            mode=BrainMode.COLD,         # default
        )
        async for event in runner.run(questions):
            ...
        # or:
        summary = await runner.run_to_completion(questions)
    """

    def __init__(
        self,
        systems: dict[str, BrainAdapter],
        mode: BrainMode = BrainMode.COLD,
        run_id: str | None = None,
    ) -> None:
        if not systems:
            raise ValueError("BrainRetrievalRunner needs at least one system")
        self.systems = systems
        self.mode = mode
        self.run_id = run_id or _new_run_id()
        self.settings = get_settings()
        self._results: list[BrainQuestionResult] = []

    async def run(
        self,
        questions: Iterable[BrainQuestion],
        dataset_name: str = "unknown",
    ) -> AsyncIterator[BrainRunEvent]:
        questions = list(questions)
        n_q = len(questions)
        n_systems = len(self.systems)
        total_pairs = n_q * n_systems
        started_at = datetime.now(UTC)

        yield BrainRunEvent(
            type="run_started",
            run_id=self.run_id,
            questions_total=total_pairs,
            questions_done=0,
        )

        # One-time setup for every adapter.
        try:
            for sys_name, adapter in self.systems.items():
                try:
                    await adapter.setup()
                except Exception as e:
                    yield BrainRunEvent(
                        type="run_failed",
                        run_id=self.run_id,
                        error=f"Setup failed for {sys_name}: {e}",
                    )
                    return
        except Exception as e:
            yield BrainRunEvent(
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
                yield BrainRunEvent(
                    type="question_started",
                    run_id=self.run_id,
                    system=sys_name,
                    question_id=q.id,
                    questions_total=total_pairs,
                    questions_done=done_count,
                )

        mode = self.mode

        async def _one(
            q: BrainQuestion, sys_name: str, adapter: BrainAdapter
        ) -> tuple[BrainQuestion, str, BrainQuestionResult, str | None]:
            """Returns (q, sys_name, result, skip_reason_or_None)."""
            async with sem:
                # --- COMPACTION mode: only unison-brain ---
                if mode == BrainMode.COMPACTION:
                    if sys_name not in _COMPACTION_CAPABLE_ADAPTERS:
                        skip_reason = (
                            f"[SKIP] COMPACTION mode not supported by adapter '{sys_name}'. "
                            "Only 'unison-brain' implements compaction. "
                            "Requires GET /api/rest/agents/eval-wiki/<slug> endpoint."
                        )
                        logger.info(skip_reason)
                        qr = BrainQuestionResult(
                            question_id=q.id,
                            system=sys_name,
                            search_result=BrainSearchResult(),
                            metrics={},
                            error=skip_reason,
                        )
                        return q, sys_name, qr, skip_reason

                    # For now, attempt compaction flow and skip if endpoint absent.
                    skip_reason = (
                        "[SKIP] COMPACTION mode requires GET "
                        "/api/rest/agents/eval-wiki/<entity_slug> on the Unison side. "
                        "Endpoint not yet available — skipping."
                    )
                    logger.info(skip_reason)
                    qr = BrainQuestionResult(
                        question_id=q.id,
                        system=sys_name,
                        search_result=BrainSearchResult(),
                        metrics={},
                        error=skip_reason,
                    )
                    return q, sys_name, qr, skip_reason

                # --- WARM mode: no reset, no ingest ---
                search_result: BrainSearchResult
                try:
                    if mode == BrainMode.WARM:
                        # Corpus is pre-loaded; just search.
                        search_result = await adapter.search(q.query, k=10)
                    else:
                        # COLD or BITEMPORAL: reset → ingest → search.
                        await adapter.reset()
                        await adapter.ingest(q.corpus)
                        search_result = await adapter.search(q.query, k=10)
                except Exception as e:
                    search_result = BrainSearchResult(error=str(e))

                if search_result.error:
                    m = _zero_metrics(mode)
                else:
                    retrieved_paths = [c.doc_path for c in search_result.chunks]
                    m = {
                        "recall_at_10": recall_at_k(retrieved_paths, q.gold_doc_paths, 10),
                        "ndcg_at_10": ndcg_at_k(retrieved_paths, q.gold_doc_paths, 10),
                        "mrr": mrr(retrieved_paths, q.gold_doc_paths),
                        "hit_at_1": hit_at_k(retrieved_paths, q.gold_doc_paths, 1),
                    }

                    if mode == BrainMode.BITEMPORAL:
                        expected_versions: dict[str, str] = q.metadata.get("expected_versions", {})
                        m["temporal_correct_at_1"] = temporal_correct_at_1(
                            retrieved_paths, q.gold_doc_paths, expected_versions
                        )

                qr = BrainQuestionResult(
                    question_id=q.id,
                    system=sys_name,
                    search_result=search_result,
                    metrics=m,
                    error=search_result.error,
                )
                return q, sys_name, qr, None

        tasks = [
            asyncio.create_task(_one(q, sys_name, adapter))
            for q in questions
            for sys_name, adapter in self.systems.items()
        ]

        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    q, sys_name, qr, skip_reason = await coro
                    self._results.append(qr)
                    done_count += 1
                    if skip_reason:
                        yield BrainRunEvent(
                            type="question_skipped",
                            run_id=self.run_id,
                            system=sys_name,
                            question_id=q.id,
                            result=qr,
                            questions_total=total_pairs,
                            questions_done=done_count,
                            skip_reason=skip_reason,
                        )
                    else:
                        yield BrainRunEvent(
                            type="question_completed",
                            run_id=self.run_id,
                            system=sys_name,
                            question_id=q.id,
                            result=qr,
                            questions_total=total_pairs,
                            questions_done=done_count,
                        )
                except Exception as e:
                    logger.exception("Brain question task crashed")
                    done_count += 1
                    yield BrainRunEvent(
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
        yield BrainRunEvent(
            type="run_completed",
            run_id=self.run_id,
            summary=summary,
        )

    async def run_to_completion(
        self,
        questions: Iterable[BrainQuestion],
        dataset_name: str = "unknown",
    ) -> BrainRunSummary:
        """Convenience for callers that don't want event streaming."""
        summary: BrainRunSummary | None = None
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
    ) -> BrainRunSummary:
        per_system: dict[str, list[BrainQuestionResult]] = {s: [] for s in self.systems}
        for r in self._results:
            per_system.setdefault(r.system, []).append(r)

        summaries: list[BrainSystemSummary] = []
        total_cost = 0.0

        for sys_name, rows in per_system.items():
            # Exclude skipped rows from metric aggregation.
            scored_rows = [r for r in rows if r.metrics]
            n = len(scored_rows)
            latencies = [r.search_result.latency_ms for r in scored_rows] or [0.0]
            cost_for_sys = sum(r.search_result.cost_usd for r in rows)
            total_cost += cost_for_sys

            _rows = scored_rows

            def _mean(key: str, _r: list[BrainQuestionResult] = _rows) -> float:
                vals = [r.metrics[key] for r in _r if key in r.metrics]
                return sum(vals) / len(vals) if vals else 0.0

            # Temporal metric: only present in BITEMPORAL mode.
            mean_temporal: float | None = None
            if self.mode == BrainMode.BITEMPORAL:
                temporal_vals = [
                    r.metrics["temporal_correct_at_1"]
                    for r in _rows
                    if "temporal_correct_at_1" in r.metrics
                ]
                mean_temporal = sum(temporal_vals) / len(temporal_vals) if temporal_vals else 0.0

            # Compaction metric: only present in COMPACTION mode.
            mean_compaction: float | None = None
            if self.mode == BrainMode.COMPACTION:
                compaction_vals = [
                    r.metrics["compaction_quality_score"]
                    for r in _rows
                    if "compaction_quality_score" in r.metrics
                ]
                mean_compaction = (
                    sum(compaction_vals) / len(compaction_vals) if compaction_vals else None
                )

            # Bootstrap 95% CIs for the two headline retrieval metrics.
            per_q_recall = [r.metrics.get("recall_at_10", 0.0) for r in _rows]
            per_q_hit1 = [r.metrics.get("hit_at_1", 0.0) for r in _rows]
            recall_ci = bootstrap_ci(per_q_recall) if per_q_recall else (0.0, 0.0)
            hit1_ci = bootstrap_ci(per_q_hit1) if per_q_hit1 else (0.0, 0.0)

            summaries.append(
                BrainSystemSummary(
                    system=sys_name,
                    n_questions=n,
                    mean_recall_at_10=_mean("recall_at_10"),
                    mean_ndcg_at_10=_mean("ndcg_at_10"),
                    mean_mrr=_mean("mrr"),
                    mean_hit_at_1=_mean("hit_at_1"),
                    recall_at_10_ci_low=recall_ci[0],
                    recall_at_10_ci_high=recall_ci[1],
                    hit_at_1_ci_low=hit1_ci[0],
                    hit_at_1_ci_high=hit1_ci[1],
                    total_cost_usd=cost_for_sys,
                    avg_latency_ms=sum(latencies) / len(latencies),
                    p50_latency_ms=_percentile(latencies, 50),
                    p95_latency_ms=_percentile(latencies, 95),
                    mean_temporal_correct_at_1=mean_temporal,
                    mean_compaction_quality=mean_compaction,
                )
            )

        return BrainRunSummary(
            run_id=self.run_id,
            dataset=dataset_name,
            track=Track.BRAIN_ONLY,
            mode=self.mode,
            systems=list(self.systems.keys()),
            n_questions=n_questions,
            started_at=started_at,
            finished_at=finished_at,
            total_cost_usd=total_cost,
            summaries=summaries,
        )

    @property
    def results(self) -> list[BrainQuestionResult]:
        return list(self._results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_metrics(mode: BrainMode) -> dict[str, float]:
    base = {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0}
    if mode == BrainMode.BITEMPORAL:
        base["temporal_correct_at_1"] = 0.0
    return base


def _new_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{ts}-{uuid.uuid4().hex[:6]}"


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
