"""End-to-end runner test using a mock adapter + mock judge.

Proves the runner loop wires up correctly: events fire in order, summaries
aggregate, cost/latency math is right.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from unison_evals.memory_evals.adapters import REGISTRY
from unison_evals.memory_evals.adapters.base import AgentAdapter
from unison_evals.memory_evals.metrics.llm_judge import LLMJudge
from unison_evals.memory_evals.runners.agent_oracle import AgentOracleRunner, _percentile
from unison_evals.types import AdapterResult, JudgeResult, Question


class _FakeAdapter(AgentAdapter):
    name = "fake"
    answers: ClassVar[dict[str, str]] = {}

    async def answer(self, question: str, oracle_context: str | None = None) -> AdapterResult:
        return AdapterResult(
            answer=self.answers.get(question, "fake-answer"),
            cost_usd=0.001,
            latency_ms=42.0,
            input_tokens=100,
            output_tokens=50,
        )


class _FakeJudge:
    """Stand-in for LLMJudge — passes if answer == expected, else fails."""

    model = "fake-judge"
    pass_threshold = 1.0

    async def judge(self, question: str, expected: str, actual: str) -> JudgeResult:
        passed = actual.strip() == expected.strip()
        verdict = "CORRECT" if passed else "WRONG"
        return JudgeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            verdict=verdict,
            confidence=1.0,
            reasoning="fake",
            cost_usd=0.0,
        )


@pytest.fixture(autouse=True)
def _register_fake_adapter():
    REGISTRY["fake"] = _FakeAdapter
    yield
    REGISTRY.pop("fake", None)


async def test_runner_loop_end_to_end() -> None:
    _FakeAdapter.answers = {
        "Q1": "right",
        "Q2": "wrong",
    }
    questions = [
        Question(id="q1", question="Q1", expected_answer="right", oracle_context="ctx"),
        Question(id="q2", question="Q2", expected_answer="right", oracle_context="ctx"),
    ]

    runner = AgentOracleRunner(systems=["fake"], judge=_FakeJudge())  # type: ignore[arg-type]
    summary = await runner.run_to_completion(questions, dataset_name="unit")

    assert summary.n_questions == 2
    assert summary.repeat == 1
    assert len(summary.summaries) == 1
    s = summary.summaries[0]
    assert s.system == "fake"
    assert s.n_questions == 2
    assert s.n_passed == 1
    assert s.n_correct == 1
    assert s.n_wrong == 1
    assert s.pass_rate == 0.5
    assert s.pass_at_k is None  # repeat=1, not computed
    assert abs(s.hallucination_rate - 0.5) < 1e-9
    assert abs(s.total_cost_usd - 0.002) < 1e-9  # 2 questions x $0.001 (judge is free)
    # Token fields populated from FakeAdapter
    assert s.mean_input_tokens == 100.0
    assert s.mean_output_tokens == 50.0


async def test_runner_emits_events_in_order() -> None:
    _FakeAdapter.answers = {"Q": "right"}
    questions = [Question(id="q", question="Q", expected_answer="right", oracle_context="ctx")]

    runner = AgentOracleRunner(systems=["fake"], judge=_FakeJudge())  # type: ignore[arg-type]
    events = []
    async for ev in runner.run(questions, dataset_name="unit"):
        events.append(ev.type)

    assert events[0] == "run_started"
    assert "question_started" in events
    assert "question_completed" in events
    assert events[-1] == "run_completed"


async def test_runner_handles_adapter_error() -> None:
    class _BrokenAdapter(AgentAdapter):
        name = "broken"

        async def answer(self, question: str, oracle_context: str | None = None) -> AdapterResult:
            return AdapterResult(answer="", cost_usd=0.0, latency_ms=10.0, error="kaboom")

    REGISTRY["broken"] = _BrokenAdapter
    try:
        questions = [Question(id="q", question="Q", expected_answer="x", oracle_context="ctx")]
        runner = AgentOracleRunner(systems=["broken"], judge=_FakeJudge())  # type: ignore[arg-type]
        summary = await runner.run_to_completion(questions, dataset_name="unit")
        s = summary.summaries[0]
        assert s.n_passed == 0  # adapter errored → judge marked it failed
    finally:
        REGISTRY.pop("broken", None)


def test_percentile_edge_cases() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([100], 50) == 100.0
    assert _percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert _percentile([1, 2, 3, 4, 5], 100) == 5.0
    assert _percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 95) == pytest.approx(95.5)


async def test_runner_pass_at_k_all_pass() -> None:
    """When repeat=3 and all runs pass, pass^3 = 1.0."""
    _FakeAdapter.answers = {"Q1": "right"}
    questions = [Question(id="q1", question="Q1", expected_answer="right", oracle_context="ctx")]

    runner = AgentOracleRunner(systems=["fake"], judge=_FakeJudge(), repeat=3)  # type: ignore[arg-type]
    summary = await runner.run_to_completion(questions, dataset_name="unit")

    assert summary.repeat == 3
    s = summary.summaries[0]
    assert s.repeat == 3
    assert s.n_questions == 3  # 1 question x 3 repeats
    assert s.n_passed == 3
    assert s.pass_at_k == 1.0


async def test_runner_pass_at_k_partial() -> None:
    """With repeat=2, if one repeat fails, pass^2 for that question = 0. With 2 questions
    where Q1 always passes and Q2 always fails, pass^2 = 0.5 (only Q1 has all runs passing)."""
    _FakeAdapter.answers = {"Q1": "right", "Q2": "wrong"}
    questions = [
        Question(id="q1", question="Q1", expected_answer="right", oracle_context="ctx"),
        Question(id="q2", question="Q2", expected_answer="right", oracle_context="ctx"),
    ]

    runner = AgentOracleRunner(systems=["fake"], judge=_FakeJudge(), repeat=2)  # type: ignore[arg-type]
    summary = await runner.run_to_completion(questions, dataset_name="unit")

    s = summary.summaries[0]
    assert s.repeat == 2
    # Q1 passes both runs; Q2 fails both. 1 out of 2 questions has all runs passing.
    assert s.pass_at_k == 0.5


async def test_runner_abstention_counts() -> None:
    """Judge returning CORRECT_ABSTAIN / INCORRECT_ABSTAIN is tracked separately."""

    class _AbstainJudge:
        model = "abstain-judge"
        pass_threshold = 1.0

        async def judge(self, question: str, expected: str, actual: str) -> JudgeResult:
            if actual == "I don't know":
                verdict = "CORRECT_ABSTAIN"
                passed = True
            elif actual == "wrong abstain":
                verdict = "INCORRECT_ABSTAIN"
                passed = False
            else:
                verdict = "CORRECT"
                passed = True
            return JudgeResult(
                score=1.0 if passed else 0.0,
                passed=passed,
                verdict=verdict,
                confidence=1.0,
                reasoning="abstain-judge",
                cost_usd=0.0,
            )

    class _AbstainAdapter(AgentAdapter):
        name = "abstain-fake"

        async def answer(self, question: str, oracle_context: str | None = None) -> AdapterResult:
            answers = {"Q1": "I don't know", "Q2": "wrong abstain", "Q3": "correct answer"}
            return AdapterResult(answer=answers.get(question, ""), cost_usd=0.0, latency_ms=1.0)

    REGISTRY["abstain-fake"] = _AbstainAdapter
    try:
        questions = [
            Question(id="q1", question="Q1", expected_answer="x", oracle_context="ctx"),
            Question(id="q2", question="Q2", expected_answer="x", oracle_context="ctx"),
            Question(id="q3", question="Q3", expected_answer="x", oracle_context="ctx"),
        ]
        runner = AgentOracleRunner(systems=["abstain-fake"], judge=_AbstainJudge())  # type: ignore[arg-type]
        summary = await runner.run_to_completion(questions, dataset_name="unit")
        s = summary.summaries[0]
        assert s.n_correct == 1
        assert s.n_correct_abstain == 1
        assert s.n_incorrect_abstain == 1
        assert s.n_wrong == 0
        assert s.n_passed == 2  # CORRECT + CORRECT_ABSTAIN
        assert s.abstention_precision == 0.5  # 1 correct abstain / (1 + 1) total abstains
    finally:
        REGISTRY.pop("abstain-fake", None)


def test_runner_repeat_validation() -> None:
    with pytest.raises(ValueError, match="repeat"):
        AgentOracleRunner(systems=["fake"], repeat=0)


def test_unused_imports_silenced() -> None:
    # Just to keep import graph honest
    assert LLMJudge.__name__ == "LLMJudge"
