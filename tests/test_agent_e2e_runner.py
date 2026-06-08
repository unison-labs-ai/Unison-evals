"""Tests for AgentE2ERunner (Track 3).

Covers:
  * Runner passes per-question corpus as seed_docs to the adapter
  * Events fire in the expected order
  * Summary is built correctly (n_passed, pass_rate, cost aggregation)
  * Adapter errors result in judge score=0 (no crash)
  * Adapters whose setup() fails are skipped (not fatal)
  * mean_input_tokens_per_q is computed from adapter input_tokens
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.base import AgentAdapter
from unison_evals.memory_evals.runners.agent_e2e import AgentE2ERunner
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
        question_id: str | None = None,
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
            question_id: str | None = None,
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
            question_id: str | None = None,
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


async def test_mean_input_tokens_per_q_computed() -> None:
    """mean_input_tokens_per_q is the mean of adapter.input_tokens across questions."""

    class _TokenAdapter(AgentAdapter):
        name = "token-adapter"

        async def answer(
            self,
            question: str,
            oracle_context: str | None = None,
            seed_docs: list[Document] | None = None,
            question_id: str | None = None,
        ) -> AdapterResult:
            return AdapterResult(
                answer="answer",
                cost_usd=0.0,
                latency_ms=5.0,
                input_tokens=200,
                output_tokens=30,
            )

    REGISTRY["token-adapter"] = _TokenAdapter
    try:
        questions = [_make_brain_question(f"q{i}") for i in range(3)]
        runner = AgentE2ERunner(
            systems=["token-adapter"],
            judge=_PassingJudge(),  # type: ignore[arg-type]
        )
        summary = await runner.run_to_completion(questions, dataset_name="test")
        s = summary.summaries[0]
        assert s.mean_input_tokens_per_q == pytest.approx(200.0)
        assert s.efficiency_ratio is None  # always None (no baseline adapter)
    finally:
        REGISTRY.pop("token-adapter", None)
