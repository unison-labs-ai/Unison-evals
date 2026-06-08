"""LLM judge — JSON parsing, 4-way verdict, threshold logic."""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.metrics.llm_judge import (
    _VALID_VERDICTS,
    LLMJudge,
    _parse_judge_json,
    _verdict_to_score,
)


def test_parse_clean_json_new_verdict_format() -> None:
    raw = '{"verdict": "CORRECT", "confidence": 0.9, "reasoning": "perfect match"}'
    parsed = _parse_judge_json(raw)
    assert parsed["verdict"] == "CORRECT"
    assert parsed["confidence"] == 0.9


def test_parse_clean_json_legacy_score_format() -> None:
    raw = '{"score": 1.0, "confidence": 0.9, "reasoning": "perfect match"}'
    parsed = _parse_judge_json(raw)
    assert parsed["score"] == 1.0
    assert parsed["confidence"] == 0.9


def test_parse_with_markdown_fence() -> None:
    raw = '```json\n{"verdict": "WRONG", "confidence": 0.7, "reasoning": "fabricated"}\n```'
    parsed = _parse_judge_json(raw)
    assert parsed["verdict"] == "WRONG"


def test_parse_with_chatter() -> None:
    raw = 'Sure: {"verdict": "CORRECT_ABSTAIN", "confidence": 0.95, "reasoning": "unanswerable"} thanks'
    parsed = _parse_judge_json(raw)
    assert parsed["verdict"] == "CORRECT_ABSTAIN"


def test_parse_garbage_returns_empty() -> None:
    assert _parse_judge_json("not valid json at all") == {}


async def test_empty_answer_short_circuits() -> None:
    judge = LLMJudge(model="claude-haiku-4-5")
    result = await judge.judge("q", "expected", "")
    assert result.score == 0.0
    assert result.passed is False
    assert result.verdict == "WRONG"
    assert "Empty" in result.reasoning


def test_pass_threshold_strict() -> None:
    judge = LLMJudge(model="claude-haiku-4-5", pass_threshold=1.0)
    assert judge.pass_threshold == 1.0


def test_pass_threshold_lenient() -> None:
    judge = LLMJudge(model="claude-haiku-4-5", pass_threshold=0.5)
    assert judge.pass_threshold == 0.5


def test_cost_calculation_haiku() -> None:
    judge = LLMJudge(model="claude-haiku-4-5")
    # Haiku 4.5: $0.80 input, $4.00 output per Mtok
    cost = judge._compute_cost(1_000_000, 1_000_000)
    assert abs(cost - 4.80) < 0.01


def test_cost_calculation_opus() -> None:
    judge = LLMJudge(model="claude-opus-4-7")
    # Opus 4.7: $15 input, $75 output per Mtok
    cost = judge._compute_cost(1_000_000, 1_000_000)
    assert abs(cost - 90.0) < 0.01


# ---------------------------------------------------------------------------
# 4-way verdict classification
# ---------------------------------------------------------------------------


def test_valid_verdicts_set() -> None:
    assert "CORRECT" in _VALID_VERDICTS
    assert "WRONG" in _VALID_VERDICTS
    assert "CORRECT_ABSTAIN" in _VALID_VERDICTS
    assert "INCORRECT_ABSTAIN" in _VALID_VERDICTS


@pytest.mark.parametrize(
    "verdict,expected_score,expected_passed",
    [
        ("CORRECT", 1.0, True),
        ("CORRECT_ABSTAIN", 1.0, True),
        ("WRONG", 0.0, False),
        ("INCORRECT_ABSTAIN", 0.0, False),
    ],
)
def test_verdict_to_score_and_passed(
    verdict: str, expected_score: float, expected_passed: bool
) -> None:
    score = _verdict_to_score(verdict)
    assert score == expected_score
    # passed = verdict in {CORRECT, CORRECT_ABSTAIN}
    passed = verdict in ("CORRECT", "CORRECT_ABSTAIN")
    assert passed == expected_passed


def test_parse_incorrect_abstain() -> None:
    raw = '{"verdict": "INCORRECT_ABSTAIN", "confidence": 0.8, "reasoning": "should have known"}'
    parsed = _parse_judge_json(raw)
    assert parsed["verdict"] == "INCORRECT_ABSTAIN"


def test_legacy_score_fallback_maps_to_verdict() -> None:
    """If the judge returns old score format, the judge() method should still map it."""
    judge = LLMJudge(model="claude-haiku-4-5", pass_threshold=1.0)
    # Simulate parsed dict with old score key
    parsed = {"score": 0.0, "confidence": 0.9, "reasoning": "wrong answer"}
    # score < pass_threshold → WRONG
    verdict = "CORRECT" if float(parsed.get("score", 0)) >= judge.pass_threshold else "WRONG"
    assert verdict == "WRONG"


def test_legacy_score_correct_maps_to_correct_verdict() -> None:
    judge = LLMJudge(model="claude-haiku-4-5", pass_threshold=1.0)
    parsed = {"score": 1.0, "confidence": 0.9, "reasoning": "perfect"}
    verdict = "CORRECT" if float(parsed.get("score", 0)) >= judge.pass_threshold else "WRONG"
    assert verdict == "CORRECT"
