"""Track 3 (agent E2E) runner — per-question corpus ingest via seed_docs, then judge.

Each BrainQuestion carries its own corpus. The runner calls
  adapter.answer(q.query, seed_docs=q.corpus)
for every (question, system) pair, then scores the answer with the LLM judge.

Yields RunEvents identically to AgentOracleRunner so the CLI and server can use
either runner interchangeably.

Brain-efficiency headline
-------------------------
The canonical Track 3 story is "given the same per-question corpus, Unison uses
Nx fewer tokens than long-context inlining at equivalent accuracy."  This is
surfaced as:

  SystemSummary.mean_input_tokens_per_q   — mean prompt tokens actually consumed
  SystemSummary.efficiency_ratio          — baseline_tokens / system_tokens
  RunSummary.efficiency_narrative         — human-readable summary string

The baseline system is ``EFFICIENCY_BASELINE`` (default: "anthropic-raw") because
that adapter inlines the full corpus into its prompt, making it the most
token-hungry naive approach. If the baseline system is not in the systems list,
efficiency_ratio is omitted for all systems.

Comprehensive matrix support
-----------------------------
Every agent adapter in REGISTRY is accepted. When adapter.setup() raises, the
system is skipped with a [SKIP] log rather than aborting the whole run. Datasets
that don't implement load_brain_questions() are expected to raise
NotImplementedError at the call site (in the CLI / server); the runner itself
never calls that method.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from ...config import get_settings
from ...results import new_run_id
from ...types import (
    BrainQuestion,
    JudgeResult,
    QuestionResult,
    RunStatus,
    RunSummary,
    SystemSummary,
    Track,
)
from ..adapters import AgentAdapter, get_adapter
from ..metrics.llm_judge import LLMJudge
from ..metrics.stats import bootstrap_ci

# The baseline system for efficiency_ratio. anthropic-raw inlines the full corpus
# into its prompt — it is the "stuff everything in" naive approach.
EFFICIENCY_BASELINE = "anthropic-raw"


class E2ERunEvent(BaseModel):
    """One progress event during an E2E run, streamed via SSE to the UI."""

    type: Literal[
        "run_started",
        "question_started",
        "question_completed",
        "question_failed",
        "run_completed",
        "run_failed",
        "system_skipped",
    ]
    run_id: str
    system: str | None = None
    question_id: str | None = None
    questions_total: int | None = None
    questions_done: int | None = None
    result: QuestionResult | None = None
    summary: RunSummary | None = None
    error: str | None = None
    skip_reason: str | None = None


class AgentE2ERunner:
    """Track 3 — for each BrainQuestion, pass its corpus as seed_docs to each
    adapter, then score the answer with the LLM judge.

    All 8 agent adapters in REGISTRY are accepted. Adapters whose setup() fails
    are skipped with a [SKIP] log rather than aborting the run.

    Use:
        runner = AgentE2ERunner(systems=["unison-agent"], judge=LLMJudge())
        async for event in runner.run(brain_questions):
            ...
        # or just:
        summary = await runner.run_to_completion(brain_questions)
    """

    def __init__(
        self,
        systems: list[str],
        judge: LLMJudge | None = None,
        run_id: str | None = None,
    ) -> None:
        if not systems:
            raise ValueError("AgentE2ERunner needs at least one system")
        self.systems = systems
        self.judge = judge or LLMJudge()
        self.run_id = run_id or new_run_id("run")
        self.settings = get_settings()
        self._results: list[QuestionResult] = []
        self._skipped_systems: list[str] = []

    async def run(
        self,
        questions: Iterable[BrainQuestion],
        dataset_name: str = "unknown",
    ) -> AsyncIterator[E2ERunEvent]:
        questions = list(questions)
        n_q = len(questions)
        started_at = datetime.now(UTC)

        # --- adapter setup (with graceful skip) ---
        adapters: dict[str, AgentAdapter] = {}
        skip_events: list[E2ERunEvent] = []
        for sys_name in self.systems:
            a = get_adapter(sys_name)
            try:
                await a.setup()
                adapters[sys_name] = a
            except Exception as e:
                reason = f"{sys_name}: setup() failed — {e}"
                logger.warning("[SKIP] {}", reason)
                self._skipped_systems.append(sys_name)
                skip_events.append(
                    E2ERunEvent(
                        type="system_skipped",
                        run_id=self.run_id,
                        system=sys_name,
                        skip_reason=reason,
                    )
                )

        active_systems = list(adapters.keys())
        n_systems = len(active_systems)
        total_pairs = n_q * n_systems

        yield E2ERunEvent(
            type="run_started",
            run_id=self.run_id,
            questions_total=total_pairs,
            questions_done=0,
        )
        for ev in skip_events:
            yield ev

        try:
            sem = asyncio.Semaphore(self.settings.max_concurrent_questions)
            done_count = 0

            async def _one(
                q: BrainQuestion, sys_name: str
            ) -> tuple[BrainQuestion, str, QuestionResult]:
                async with sem:
                    adapter = adapters[sys_name]
                    # Track 3: pass per-question corpus as seed_docs; no oracle_context.
                    answer = await adapter.answer(q.query, seed_docs=q.corpus)
                    if answer.error:
                        judge_res = JudgeResult(
                            score=0.0,
                            passed=False,
                            confidence=1.0,
                            reasoning=f"Adapter error: {answer.error}",
                            cost_usd=0.0,
                        )
                    else:
                        # BrainQuestion uses `query` not `question`; gold answer is
                        # inferred from gold_doc_paths — but for LLM judge scoring we
                        # need a free-text expected answer. Datasets that support Track 3
                        # should store it in metadata["expected_answer"]; fall back to
                        # a joined list of gold paths so the judge still fires.
                        expected = str(
                            q.metadata.get("expected_answer") or "; ".join(sorted(q.gold_doc_paths))
                        )
                        judge_res = await self.judge.judge(q.query, expected, answer.answer)
                    qr = QuestionResult(
                        question_id=q.id,
                        system=sys_name,
                        adapter=answer,
                        judge=judge_res,
                    )
                    return q, sys_name, qr

            tasks = [
                asyncio.create_task(_one(q, sys_name))
                for q in questions
                for sys_name in active_systems
            ]

            for q in questions:
                for sys_name in active_systems:
                    yield E2ERunEvent(
                        type="question_started",
                        run_id=self.run_id,
                        system=sys_name,
                        question_id=q.id,
                        questions_total=total_pairs,
                        questions_done=done_count,
                    )

            for coro in asyncio.as_completed(tasks):
                try:
                    q, sys_name, qr = await coro
                    self._results.append(qr)
                    done_count += 1
                    yield E2ERunEvent(
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
                    yield E2ERunEvent(
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
                active_systems=active_systems,
                started_at=started_at,
                finished_at=finished_at,
            )
            yield E2ERunEvent(
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
        questions: Iterable[BrainQuestion],
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
        active_systems: list[str],
        started_at: datetime,
        finished_at: datetime,
    ) -> RunSummary:
        per_system: dict[str, list[QuestionResult]] = {sys: [] for sys in active_systems}
        for r in self._results:
            per_system.setdefault(r.system, []).append(r)

        # Compute mean_input_tokens_per_q for all active systems first (needed for
        # efficiency_ratio calculation).
        mean_tokens: dict[str, float] = {}
        for sys_name, rows in per_system.items():
            if rows:
                mean_tokens[sys_name] = sum(r.adapter.input_tokens for r in rows) / len(rows)
            else:
                mean_tokens[sys_name] = 0.0

        baseline_tokens = mean_tokens.get(EFFICIENCY_BASELINE)

        summaries: list[SystemSummary] = []
        total_cost = 0.0
        for sys_name, rows in per_system.items():
            n = len(rows)
            n_passed = sum(1 for r in rows if r.judge and r.judge.passed)
            adapter_cost = sum(r.adapter.cost_usd for r in rows)
            judge_cost = sum(r.judge.cost_usd if r.judge else 0.0 for r in rows)
            total_for_sys = adapter_cost + judge_cost
            total_cost += total_for_sys
            latencies = [r.adapter.latency_ms for r in rows] or [0.0]
            tokens_unavail = any(r.adapter.tokens_unavailable for r in rows)
            mean_tok = mean_tokens[sys_name]

            # efficiency_ratio = baseline_tokens / system_tokens. >1 means this system
            # consumed fewer tokens than the baseline.
            efficiency_ratio: float | None = None
            if (
                baseline_tokens is not None
                and not tokens_unavail
                and mean_tok > 0
                and sys_name != EFFICIENCY_BASELINE
            ):
                efficiency_ratio = baseline_tokens / mean_tok

            # Bootstrap 95% CI over per-question pass outcomes.
            per_q_pass = [1.0 if r.judge and r.judge.passed else 0.0 for r in rows]
            ci_low, ci_high = bootstrap_ci(per_q_pass) if per_q_pass else (0.0, 0.0)

            summaries.append(
                SystemSummary(
                    system=sys_name,
                    n_questions=n,
                    n_passed=n_passed,
                    pass_rate=(n_passed / n) if n else 0.0,
                    pass_rate_ci_low=ci_low,
                    pass_rate_ci_high=ci_high,
                    total_cost_usd=total_for_sys,
                    cost_per_question_usd=(total_for_sys / n) if n else 0.0,
                    cost_per_solved_usd=(total_for_sys / n_passed) if n_passed else None,
                    avg_latency_ms=sum(latencies) / len(latencies),
                    p50_latency_ms=_percentile(latencies, 50),
                    p95_latency_ms=_percentile(latencies, 95),
                    mean_input_tokens_per_q=mean_tok,
                    efficiency_ratio=efficiency_ratio,
                    tokens_unavailable=tokens_unavail,
                )
            )

        # Build headline efficiency narrative.
        efficiency_narrative = _build_efficiency_narrative(summaries, baseline_tokens)

        return RunSummary(
            run_id=self.run_id,
            dataset=dataset_name,
            track=Track.AGENT_E2E,
            systems=self.systems,
            judge_model=self.judge.model,
            n_questions=n_questions,
            started_at=started_at,
            finished_at=finished_at,
            total_cost_usd=total_cost,
            summaries=summaries,
            efficiency_narrative=efficiency_narrative,
        )

    @property
    def results(self) -> list[QuestionResult]:
        return list(self._results)

    @property
    def skipped_systems(self) -> list[str]:
        return list(self._skipped_systems)


def _build_efficiency_narrative(
    summaries: list[SystemSummary],
    baseline_tokens: float | None,
) -> str | None:
    """Build a human-readable efficiency narrative for the run summary.

    Example:
      "unison-agent achieved 83.0% pass-rate at 1 420 mean input tokens
       (8.2x more efficient than anthropic-raw at 11 640 tokens)"
    """
    if baseline_tokens is None or baseline_tokens == 0:
        return None

    lines: list[str] = []
    for s in summaries:
        if s.system == EFFICIENCY_BASELINE or s.tokens_unavailable:
            continue
        if s.efficiency_ratio is not None and s.mean_input_tokens_per_q > 0:
            lines.append(
                f"{s.system} achieved {s.pass_rate * 100:.1f}% pass-rate at "
                f"{s.mean_input_tokens_per_q:,.0f} mean input tokens "
                f"({s.efficiency_ratio:.1f}x more efficient than {EFFICIENCY_BASELINE} "
                f"at {baseline_tokens:,.0f} tokens)"
            )
    return "; ".join(lines) if lines else None


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
