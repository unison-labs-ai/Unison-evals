"""LLM-as-judge — uses Anthropic Claude to score answers vs. expected.

Pinned model + temperature=0 for reproducibility. Returns:
  - score: 0.0 / 0.5 / 1.0
  - passed: bool (score >= threshold)
  - confidence: 0..1 (model's self-reported certainty)
  - reasoning: short explanation
  - cost_usd: judge call cost

Cost notes: Opus judging ~$0.005/question; Haiku ~$0.0005/question.
For v0.0 published numbers use Opus; for CI smoke use Haiku.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic
from loguru import logger

from ...config import get_settings
from ...types import JudgeResult

# Pricing (per million tokens) for the judge models we support.
# Update when Anthropic publishes new prices.
JUDGE_PRICING: dict[str, tuple[float, float]] = {
    # claude-opus-4-5
    "claude-opus-4-5-20250101": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    # claude-opus-4-7 (latest as of 2026-05)
    "claude-opus-4-7-20260101": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    # claude-sonnet-4-5/4-6
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # claude-haiku-4-5
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-haiku-4-5": (0.80, 4.0),
}

JUDGE_PROMPT = """You are evaluating whether an AI agent's answer correctly answers a question, \
given the expected answer.

Classify the agent's answer as ONE of the following verdicts:
  CORRECT           — the answer captures the expected information; minor wording differences \
and extra correct context are fine
  WRONG             — the answer asserts something contradictory, fabricated, or clearly wrong \
(includes partial answers with material errors)
  CORRECT_ABSTAIN   — the agent correctly said "I don't know" / "I cannot find this" / \
"no information available", and the expected answer is genuinely not derivable from the \
provided context (i.e. the question is unanswerable given what the agent was given)
  INCORRECT_ABSTAIN — the agent said "I don't know" / refused to answer, but the answer \
was derivable from the context

Rules:
- A different correct phrasing of the same fact is CORRECT.
- A confidently stated wrong fact is WRONG (not INCORRECT_ABSTAIN).
- Only use CORRECT_ABSTAIN / INCORRECT_ABSTAIN when the agent explicitly abstained \
(said it doesn't know or can't find the answer). If the agent gave a substantive answer, \
use CORRECT or WRONG.
- Use CORRECT_ABSTAIN only when the expected answer truly is not in the context the agent \
was given. If context contained the answer and the agent abstained, use INCORRECT_ABSTAIN.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

AGENT'S ANSWER:
{actual}

Respond with ONLY a JSON object on a single line, no markdown:
{{"verdict": "CORRECT|WRONG|CORRECT_ABSTAIN|INCORRECT_ABSTAIN", "confidence": 0.0-1.0, \
"reasoning": "one short sentence"}}"""


class LLMJudge:
    """Anthropic-backed answer judge.

    Threshold for `passed`: 1.0 (strict) by default. Lower it (0.5) for
    partial-credit datasets — pass via `pass_threshold`.
    """

    def __init__(
        self,
        model: str | None = None,
        pass_threshold: float = 1.0,
    ) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.judge_model
        self.pass_threshold = pass_threshold
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set. Required for the LLM judge.")
            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    async def judge(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
    ) -> JudgeResult:
        if not actual_answer.strip():
            return JudgeResult(
                score=0.0,
                passed=False,
                verdict="WRONG",
                confidence=1.0,
                reasoning="Empty answer.",
                cost_usd=0.0,
            )

        client = self._get_client()
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected=expected_answer,
            actual=actual_answer,
        )

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.settings.judge_timeout,
            )
        except Exception as e:
            logger.warning("Judge call failed: {}", e)
            return JudgeResult(
                score=0.0,
                passed=False,
                verdict="WRONG",
                confidence=0.0,
                reasoning=f"Judge error: {e}",
                cost_usd=0.0,
            )

        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        parsed = _parse_judge_json(text)

        verdict = str(parsed.get("verdict", "")).upper()
        if verdict not in _VALID_VERDICTS:
            # Fallback: if the model returned the old score-based format, map it.
            score_raw = parsed.get("score")
            if score_raw is not None:
                score_val = float(score_raw)
                verdict = "CORRECT" if score_val >= self.pass_threshold else "WRONG"
            else:
                verdict = "WRONG"

        # Derive legacy score for back-compat.
        score = _verdict_to_score(verdict)
        # passed = True for CORRECT or CORRECT_ABSTAIN (both are "good" outcomes).
        passed = verdict in ("CORRECT", "CORRECT_ABSTAIN")

        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", text[:200]))
        cost = self._compute_cost(response.usage.input_tokens, response.usage.output_tokens)

        return JudgeResult(
            score=score,
            passed=passed,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            cost_usd=cost,
        )

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = JUDGE_PRICING.get(self.model, (15.0, 75.0))  # default to Opus pricing
        input_price, output_price = pricing
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


_VALID_VERDICTS = {"CORRECT", "WRONG", "CORRECT_ABSTAIN", "INCORRECT_ABSTAIN"}


def _verdict_to_score(verdict: str) -> float:
    """Map 4-way verdict to legacy 0/0.5/1.0 score for back-compat."""
    if verdict == "CORRECT":
        return 1.0
    if verdict == "CORRECT_ABSTAIN":
        return 1.0
    if verdict == "INCORRECT_ABSTAIN":
        return 0.0
    return 0.0  # WRONG


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Best-effort JSON parse. The judge is instructed to return a single JSON
    object; strip any surrounding chatter as a fallback."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json fences
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
