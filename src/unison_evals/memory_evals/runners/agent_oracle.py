"""Track 2 (agent oracle) runner — feeds gold context to each system, scores
the answer, accumulates cost/latency.

Yields events as it goes so the FastAPI server can stream live progress
to the UI. The CLI just consumes the events for terminal output.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from ...config import get_settings
from ...results import new_run_id
from ...types import (
    JudgeResult,
    Question,
    QuestionResult,
    RunStatus,
    RunSummary,
    SystemSummary,
    Track,
)
from ..adapters import AgentAdapter, get_adapter
from ..metrics.llm_judge import LLMJudge
from ..metrics.stats import bootstrap_ci


class RunEvent(BaseModel):
    """One progress event during a run, streamed via SSE to the UI."""

    type: Literal[
        "run_started",
        "question_started",
        "question_completed",
        "question_failed",
        "run_completed",
        "run_failed",
    ]
    run_id: str
    # Optional fields by event type:
    system: str | None = None
    question_id: str | None = None
    questions_total: int | None = None
    questions_done: int | None = None
    result: QuestionResult | None = None
    summary: RunSummary | None = None
    error: str | None = None


class AgentOracleRunner:
    """Track 2 — feed each question's oracle_context to each adapter,
    score the answer with the LLM judge.

    When repeat > 1, each (system, question) pair is run N times and
    pass^k is computed: the fraction of questions where ALL k runs passed.

    Use:
        runner = AgentOracleRunner(systems=["unison-agent", "claude-code"], judge=LLMJudge())
        async for event in runner.run(questions):
            ...
        # or just:
        summary = await runner.run_to_completion(questions)
    """

    def __init__(
        self,
        systems: list[str],
        judge: LLMJudge | None = None,
        run_id: str | None = None,
        repeat: int = 1,
    ) -> None:
        if not systems:
            raise ValueError("AgentOracleRunner needs at least one system")
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.systems = systems
        self.judge = judge or LLMJudge()
        self.run_id = run_id or new_run_id("run")
        self.repeat = repeat
        self.settings = get_settings()
        self._results: list[QuestionResult] = []

    async def run(
        self,
        questions: Iterable[Question],
        dataset_name: str = "unknown",
    ) -> AsyncIterator[RunEvent]:
        questions = list(questions)
        n_q = len(questions)
        n_systems = len(self.systems)
        # Total work = questions x systems x repeat
        total_pairs = n_q * n_systems * self.repeat
        started_at = datetime.now(UTC)

        yield RunEvent(
            type="run_started",
            run_id=self.run_id,
            questions_total=total_pairs,
            questions_done=0,
        )

        # Set up adapters once each.
        adapters: dict[str, AgentAdapter] = {}
        try:
            for sys_name in self.systems:
                a = get_adapter(sys_name)
                try:
                    await a.setup()
                except Exception as e:
                    yield RunEvent(
                        type="run_failed",
                        run_id=self.run_id,
                        error=f"Setup failed for {sys_name}: {e}",
                    )
                    return
                adapters[sys_name] = a

            # Bounded-concurrency loop over (question, system, repeat_index) triples.
            sem = asyncio.Semaphore(self.settings.max_concurrent_questions)
            done_count = 0

            async def _one(
                q: Question, sys_name: str, rep: int
            ) -> tuple[Question, str, int, QuestionResult]:
                async with sem:
                    adapter = adapters[sys_name]
                    answer = await adapter.answer(q.question, oracle_context=q.oracle_context)
                    if answer.error:
                        judge_res = JudgeResult(
                            score=0.0,
                            passed=False,
                            verdict="WRONG",
                            confidence=1.0,
                            reasoning=f"Adapter error: {answer.error}",
                            cost_usd=0.0,
                        )
                    else:
                        judge_res = await self.judge.judge(
                            q.question, q.expected_answer, answer.answer
                        )
                    qr = QuestionResult(
                        question_id=q.id,
                        system=sys_name,
                        adapter=answer,
                        judge=judge_res,
                    )
                    return q, sys_name, rep, qr

            tasks = [
                asyncio.create_task(_one(q, sys_name, rep))
                for q in questions
                for sys_name in self.systems
                for rep in range(self.repeat)
            ]

            # Emit question_started events up front so the UI can render placeholders.
            for q in questions:
                for sys_name in self.systems:
                    yield RunEvent(
                        type="question_started",
                        run_id=self.run_id,
                        system=sys_name,
                        question_id=q.id,
                        questions_total=total_pairs,
                        questions_done=done_count,
                    )

            for coro in asyncio.as_completed(tasks):
                try:
                    q, sys_name, _rep, qr = await coro
                    self._results.append(qr)
                    done_count += 1
                    yield RunEvent(
                        type="question_completed",
                        run_id=self.run_id,
                        system=sys_name,
                        question_id=q.id,
                        result=qr,
                        questions_total=total_pairs,
                        questions_done=done_count,
                    )
                except Exception as e:
                    logger.exception("Question task crashed")
                    done_count += 1
                    yield RunEvent(
                        type="question_failed",
                        run_id=self.run_id,
                        questions_total=total_pairs,
                        questions_done=done_count,
                        error=str(e),
                    )

            finished_at = datetime.now(UTC)
            summary = self._build_summary(
                dataset_name=dataset_name,
                n_questions=n_q,
                started_at=started_at,
                finished_at=finished_at,
            )
            yield RunEvent(
                type="run_completed",
                run_id=self.run_id,
                summary=summary,
            )
        finally:
            for a in adapters.values():
                try:
                    await a.teardown()
                except Exception as e:  # pragma: no cover
                    logger.warning("teardown failed for {}: {}", a.name, e)

    async def run_to_completion(
        self,
        questions: Iterable[Question],
        dataset_name: str = "unknown",
    ) -> RunSummary:
        """Convenience for callers that don't want event streaming."""
        summary: RunSummary | None = None
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
    ) -> RunSummary:
        # Group results by (system, question_id) for pass^k computation.
        # Structure: {sys_name: {q_id: [QuestionResult, ...]}}
        per_system_q: dict[str, dict[str, list[QuestionResult]]] = {sys: {} for sys in self.systems}
        for r in self._results:
            sys_bucket = per_system_q.setdefault(r.system, {})
            sys_bucket.setdefault(r.question_id, []).append(r)

        summaries: list[SystemSummary] = []
        total_cost = 0.0
        for sys_name in self.systems:
            q_buckets = per_system_q.get(sys_name, {})
            # Flatten all rows for aggregate stats.
            rows = [r for bucket in q_buckets.values() for r in bucket]
            n = len(rows)

            # Verdict counts.
            n_correct = sum(1 for r in rows if r.judge and r.judge.verdict == "CORRECT")
            n_wrong = sum(1 for r in rows if r.judge and r.judge.verdict == "WRONG")
            n_correct_abstain = sum(
                1 for r in rows if r.judge and r.judge.verdict == "CORRECT_ABSTAIN"
            )
            n_incorrect_abstain = sum(
                1 for r in rows if r.judge and r.judge.verdict == "INCORRECT_ABSTAIN"
            )
            n_passed = n_correct + n_correct_abstain

            # pass^k: for each question, did ALL k runs pass?
            # This is the TAU-bench definition: reliability at k independent trials.
            if self.repeat > 1 and q_buckets:
                n_q_all_passed = sum(
                    1
                    for bucket in q_buckets.values()
                    if all(r.judge and r.judge.passed for r in bucket)
                )
                pass_at_k: float | None = n_q_all_passed / len(q_buckets)
            else:
                pass_at_k = None

            # Cost.
            adapter_cost = sum(r.adapter.cost_usd for r in rows)
            judge_cost = sum(r.judge.cost_usd if r.judge else 0.0 for r in rows)
            total_for_sys = adapter_cost + judge_cost
            total_cost += total_for_sys

            # Latency — use one result per (system, question) pair for latency to avoid
            # double-counting when repeat > 1 (we report per-question latency, not per-trial).
            latencies = [r.adapter.latency_ms for r in rows] or [0.0]

            # Token efficiency.
            mean_input = _mean([r.adapter.input_tokens for r in rows])
            mean_output = _mean([r.adapter.output_tokens for r in rows])
            mean_cache_read = _mean([r.adapter.cache_read_tokens for r in rows])
            mean_cache_creation = _mean([r.adapter.cache_creation_tokens for r in rows])
            total_cache_denom = mean_cache_read + mean_cache_creation + mean_input
            cache_hit_rate = (mean_cache_read / total_cache_denom) if total_cache_denom > 0 else 0.0
            mean_tool_calls = _mean([r.adapter.tool_calls for r in rows])
            mean_memory_ops = _mean([r.adapter.memory_ops for r in rows])

            # Derived quality metrics.
            hallucination_rate = (n_wrong / n) if n else 0.0
            total_abstain = n_correct_abstain + n_incorrect_abstain
            abstention_precision: float | None = (
                (n_correct_abstain / total_abstain) if total_abstain > 0 else None
            )

            # pass_rate is computed over unique (question, trial) pairs; when
            # repeat > 1 it averages over all trials (legacy metric).
            pass_rate = (n_passed / n) if n else 0.0

            # Bootstrap 95% CI over per-row pass outcomes (one entry per trial).
            per_q_pass = [1.0 if r.judge and r.judge.passed else 0.0 for r in rows]
            ci_low, ci_high = bootstrap_ci(per_q_pass) if per_q_pass else (0.0, 0.0)

            summaries.append(
                SystemSummary(
                    system=sys_name,
                    n_questions=n,
                    n_passed=n_passed,
                    n_correct=n_correct,
                    n_wrong=n_wrong,
                    n_correct_abstain=n_correct_abstain,
                    n_incorrect_abstain=n_incorrect_abstain,
                    pass_rate=pass_rate,
                    pass_rate_ci_low=ci_low,
                    pass_rate_ci_high=ci_high,
                    pass_at_k=pass_at_k,
                    repeat=self.repeat,
                    hallucination_rate=hallucination_rate,
                    abstention_precision=abstention_precision,
                    total_cost_usd=total_for_sys,
                    cost_per_question_usd=(total_for_sys / n) if n else 0.0,
                    cost_per_solved_usd=(total_for_sys / n_passed) if n_passed else None,
                    avg_latency_ms=sum(latencies) / len(latencies),
                    p50_latency_ms=_percentile(latencies, 50),
                    p95_latency_ms=_percentile(latencies, 95),
                    mean_input_tokens=mean_input,
                    mean_output_tokens=mean_output,
                    mean_cache_read_tokens=mean_cache_read,
                    mean_cache_creation_tokens=mean_cache_creation,
                    cache_hit_rate=cache_hit_rate,
                    mean_tool_calls=mean_tool_calls,
                    mean_memory_ops=mean_memory_ops,
                )
            )

        return RunSummary(
            run_id=self.run_id,
            dataset=dataset_name,
            track=Track.AGENT_ORACLE,
            systems=self.systems,
            judge_model=self.judge.model,
            n_questions=n_questions,
            repeat=self.repeat,
            started_at=started_at,
            finished_at=finished_at,
            total_cost_usd=total_cost,
            summaries=summaries,
        )

    @property
    def results(self) -> list[QuestionResult]:
        return list(self._results)


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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
_ = (RunStatus, statistics)  # silence unused-import noise; both used in adjacent files
