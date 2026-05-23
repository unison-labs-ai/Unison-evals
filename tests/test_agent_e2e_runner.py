"""Tests for AgentE2ERunner (Track 3).

Covers:
  * Runner passes per-question corpus as seed_docs to the adapter
  * Events fire in the expected order
  * Summary is built correctly (n_passed, pass_rate, cost aggregation)
  * Adapter errors result in judge score=0 (no crash)
  * Adapters whose setup() fails are skipped (not fatal)
  * Brain-efficiency metrics (mean_input_tokens_per_q, efficiency_ratio) computed correctly
  * Efficiency narrative is built when baseline system is present
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.base import AgentAdapter
from unison_evals.memory_evals.runners.agent_e2e import EFFICIENCY_BASELINE, AgentE2ERunner
from unison_evals.types import AdapterResult, BrainQuestion, Document, JudgeResult


class _RecordingAdapter(AgentAdapter):
    """Fake adapter that records every call's seed_docs for assertion."""

    name = "recording"
    calls: ClassVar[list[dict]] = []

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        self.calls.append(
            {
                "question": question,
                "oracle_context": oracle_context,
                "seed_docs": list(seed_docs) if seed_docs is not None else None,
            }
        )
        # Simulate returning known token counts for efficiency metric tests.
        return AdapterResult(
            answer="recorded-answer",
            cost_usd=0.001,
            latency_ms=10.0,
            input_tokens=100,
            output_tokens=20,
        )


class _PassingJudge:
    model = "fake-judge"
    pass_threshold = 1.0

    async def judge(self, question: str, expected: str, actual: str) -> JudgeResult:
        return JudgeResult(score=1.0, passed=True, confidence=1.0, reasoning="ok", cost_usd=0.0)


class _FailingJudge:
    model = "fake-judge"
    pass_threshold = 1.0

    async def judge(self, question: str, expected: str, actual: str) -> JudgeResult:
        return JudgeResult(score=0.0, passed=False, confidence=1.0, reasoning="no", cost_usd=0.0)


@pytest.fixture(autouse=True)
def _register_recording():
    _RecordingAdapter.calls = []
    REGISTRY["recording"] = _RecordingAdapter
    yield
    REGISTRY.pop("recording", None)


def _make_brain_question(qid: str, n_docs: int = 2) -> BrainQuestion:
    return BrainQuestion(
        id=qid,
        query=f"question-{qid}",
        corpus=[
            Document(path=f"/doc-{qid}-{i}.md", body=f"body {i} for {qid}") for i in range(n_docs)
        ],
        gold_doc_paths={f"/doc-{qid}-0.md"},
        metadata={"expected_answer": f"answer-{qid}"},
    )


async def test_runner_passes_seed_docs_to_adapter() -> None:
    """The runner must pass q.corpus as seed_docs, not oracle_context."""
    q1 = _make_brain_question("q1", n_docs=3)
    q2 = _make_brain_question("q2", n_docs=1)

    runner = AgentE2ERunner(systems=["recording"], judge=_PassingJudge())  # type: ignore[arg-type]
    await runner.run_to_completion([q1, q2], dataset_name="test")

    assert len(_RecordingAdapter.calls) == 2
    for call in _RecordingAdapter.calls:
        assert call["oracle_context"] is None, "oracle_context must not be set for Track 3"
        assert call["seed_docs"] is not None, "seed_docs must be provided"

    call_by_q = {c["question"]: c for c in _RecordingAdapter.calls}
    assert len(call_by_q["question-q1"]["seed_docs"]) == 3
    assert len(call_by_q["question-q2"]["seed_docs"]) == 1


async def test_runner_events_in_order() -> None:
    q = _make_brain_question("q1")
    runner = AgentE2ERunner(systems=["recording"], judge=_PassingJudge())  # type: ignore[arg-type]
    event_types = []
    async for ev in runner.run([q], dataset_name="test"):
        event_types.append(ev.type)

    assert event_types[0] == "run_started"
    assert "question_started" in event_types
    assert "question_completed" in event_types
    assert event_types[-1] == "run_completed"


async def test_runner_summary_aggregation() -> None:
    questions = [_make_brain_question(f"q{i}") for i in range(4)]
    runner = AgentE2ERunner(systems=["recording"], judge=_PassingJudge(), run_id="test-run")  # type: ignore[arg-type]
    summary = await runner.run_to_completion(questions, dataset_name="test-ds")

    assert summary.n_questions == 4
    assert summary.track.value == "agent-e2e"
    assert summary.dataset == "test-ds"
    assert len(summary.summaries) == 1
    s = summary.summaries[0]
    assert s.n_passed == 4
    assert s.pass_rate == pytest.approx(1.0)
    assert s.n_questions == 4
    # 4 questions x $0.001 adapter cost each, judge is free
    assert abs(s.total_cost_usd - 0.004) < 1e-9


async def test_runner_adapter_error_gives_judge_score_zero() -> None:
    class _ErrorAdapter(AgentAdapter):
        name = "error-adapter"

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
        ) -> AdapterResult:
            return AdapterResult(answer="", cost_usd=0.0, latency_ms=5.0, error="simulated error")

    REGISTRY["error-adapter"] = _ErrorAdapter
    try:
        q = _make_brain_question("q1")
        runner = AgentE2ERunner(systems=["error-adapter"], judge=_PassingJudge())  # type: ignore[arg-type]
        summary = await runner.run_to_completion([q], dataset_name="test")
        s = summary.summaries[0]
        assert s.n_passed == 0
    finally:
        REGISTRY.pop("error-adapter", None)


async def test_runner_skips_adapter_that_fails_setup() -> None:
    """Adapter whose setup() fails should be skipped, not abort the whole run."""

    class _BadSetupAdapter(AgentAdapter):
        name = "bad-setup"

        async def setup(self) -> None:
            raise RuntimeError("simulated setup failure")

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
        ) -> AdapterResult:
            return AdapterResult(answer="x", cost_usd=0.0, latency_ms=1.0)

    REGISTRY["bad-setup"] = _BadSetupAdapter
    REGISTRY["recording"] = _RecordingAdapter

    try:
        q = _make_brain_question("q1")
        # Both systems requested; bad-setup should be skipped, recording should run.
        runner = AgentE2ERunner(
            systems=["bad-setup", "recording"],
            judge=_PassingJudge(),  # type: ignore[arg-type]
        )
        skip_events = []
        async for ev in runner.run([q], dataset_name="test"):
            if ev.type == "system_skipped":
                skip_events.append(ev)

        assert len(skip_events) == 1
        assert "bad-setup" in (skip_events[0].system or "")
        assert "bad-setup" in runner.skipped_systems
        # recording adapter should still have been called
        assert any(c["question"] == "question-q1" for c in _RecordingAdapter.calls)
    finally:
        REGISTRY.pop("bad-setup", None)


async def test_brain_efficiency_metrics_computed() -> None:
    """efficiency_ratio = baseline_tokens / system_tokens for non-baseline systems."""

    class _BaselineAdapter(AgentAdapter):
        name = EFFICIENCY_BASELINE  # "anthropic-raw"

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
        ) -> AdapterResult:
            return AdapterResult(
                answer="baseline",
                cost_usd=0.0,
                latency_ms=5.0,
                input_tokens=1000,
                output_tokens=50,
            )

    class _EfficientAdapter(AgentAdapter):
        name = "efficient-system"

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
        ) -> AdapterResult:
            return AdapterResult(
                answer="efficient",
                cost_usd=0.0,
                latency_ms=5.0,
                input_tokens=100,
                output_tokens=20,
            )

    # Temporarily override/add adapters to registry; always restore on exit.
    _orig_baseline = REGISTRY.get(EFFICIENCY_BASELINE)
    REGISTRY[EFFICIENCY_BASELINE] = _BaselineAdapter  # type: ignore[assignment]
    REGISTRY["efficient-system"] = _EfficientAdapter

    try:
        questions = [_make_brain_question(f"q{i}") for i in range(3)]
        runner = AgentE2ERunner(
            systems=[EFFICIENCY_BASELINE, "efficient-system"],
            judge=_PassingJudge(),  # type: ignore[arg-type]
        )
        summary = await runner.run_to_completion(questions, dataset_name="test")

        sys_map = {s.system: s for s in summary.summaries}

        baseline_s = sys_map.get(EFFICIENCY_BASELINE)
        assert baseline_s is not None
        assert baseline_s.mean_input_tokens_per_q == pytest.approx(1000.0)
        assert baseline_s.efficiency_ratio is None  # baseline has no ratio vs itself

        efficient_s = sys_map.get("efficient-system")
        assert efficient_s is not None
        assert efficient_s.mean_input_tokens_per_q == pytest.approx(100.0)
        assert efficient_s.efficiency_ratio == pytest.approx(10.0)  # 1000/100

        assert summary.efficiency_narrative is not None
        assert "10.0x" in summary.efficiency_narrative
    finally:
        # Restore original registry state precisely.
        if _orig_baseline is not None:
            REGISTRY[EFFICIENCY_BASELINE] = _orig_baseline
        else:
            REGISTRY.pop(EFFICIENCY_BASELINE, None)
        REGISTRY.pop("efficient-system", None)


async def test_brain_efficiency_metrics_tokens_unavailable() -> None:
    """When all adapters return tokens_unavailable=True, efficiency_ratio is None."""

    class _NoTokenAdapter(AgentAdapter):
        name = "no-tokens"

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
        ) -> AdapterResult:
            return AdapterResult(
                answer="x",
                cost_usd=0.0,
                latency_ms=5.0,
                input_tokens=0,
                output_tokens=0,
                tokens_unavailable=True,
            )

    REGISTRY["no-tokens"] = _NoTokenAdapter
    try:
        q = _make_brain_question("q1")
        runner = AgentE2ERunner(systems=["no-tokens"], judge=_PassingJudge())  # type: ignore[arg-type]
        summary = await runner.run_to_completion([q], dataset_name="test")
        s = summary.summaries[0]
        assert s.tokens_unavailable is True
        assert s.efficiency_ratio is None
        assert summary.efficiency_narrative is None
    finally:
        REGISTRY.pop("no-tokens", None)
