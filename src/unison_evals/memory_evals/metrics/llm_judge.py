"""LLM-as-judge — scores answers vs. expected using OpenAI (gpt-*) OR Anthropic
(claude-*), routed automatically from the judge model name.

Pinned model + temperature=0 for reproducibility. Returns:
  - score: 0.0 / 0.5 / 1.0
  - passed: bool (score >= threshold)
  - confidence: 0..1 (model's self-reported certainty)
  - reasoning: short explanation
  - cost_usd: judge call cost

The canonical judges are provider-specific: LongMemEval/MemoryAgentBench use
gpt-4o-2024-08-06 (OpenAI), so the judge MUST be able to call OpenAI — not just
Anthropic. Provider is inferred from the model prefix.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic
from loguru import logger
from openai import AsyncOpenAI

from ...config import get_settings
from ...types import JudgeResult

# Pricing (per million tokens) for the judge models we support. (input, output)
JUDGE_PRICING: dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-opus-4-5-20250101": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-7-20260101": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-haiku-4-5": (0.80, 4.0),
    # --- OpenAI ---
    "gpt-4o-2024-08-06": (2.50, 10.0),  # canonical LongMemEval / MemoryAgentBench judge
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.0),
}


def judge_provider(model: str) -> str:
    """Infer the API provider from a judge model name."""
    m = model.lower()
    if m.startswith(("gpt-", "gpt4", "gpt5", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        raise ValueError(
            f"Judge model {model!r} is a Google model — the memory-eval judge only wires "
            "OpenAI + Anthropic. Use a gpt-* or claude-* judge (e.g. gpt-4o-2024-08-06)."
        )
    raise ValueError(
        f"Cannot infer a provider for judge model {model!r} (expected gpt-* or claude-*)."
    )

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
    """Answer judge backed by OpenAI (gpt-*) or Anthropic (claude-*), selected
    from the model name. Threshold for `passed`: 1.0 (strict) by default. Lower
    it (0.5) for partial-credit datasets — pass via `pass_threshold`.
    """

    def __init__(
        self,
        model: str | None = None,
        pass_threshold: float = 1.0,
    ) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.judge_model
        self.provider = judge_provider(self.model)
        self.pass_threshold = pass_threshold
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    def _anthropic_client(self) -> AsyncAnthropic:
        if self._anthropic is None:
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set — required for a claude-* judge.")
            self._anthropic = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._anthropic

    def _openai_client(self) -> AsyncOpenAI:
        if self._openai is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set — required for a gpt-* judge.")
            self._openai = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._openai

    async def _call_model(self, prompt: str) -> tuple[str, int, int]:
        """Call the right provider for self.model. Returns (text, in_tok, out_tok)."""
        if self.provider == "anthropic":
            resp = await self._anthropic_client().messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.settings.judge_timeout,
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        # openai — the gpt-5*/o1*/o3*/o4* families have two API quirks vs gpt-4o:
        # they reject temperature != 1.0, and they take `max_completion_tokens`
        # instead of `max_tokens`. Coerce both so any OpenAI judge stays usable
        # (gpt-4o, the canonical memory-bench judge, takes the classic params).
        m = self.model.lower()
        new_family = m.startswith(("o1", "o3", "o4")) or "gpt-5" in m
        token_kwarg = {"max_completion_tokens": 300} if new_family else {"max_tokens": 300}
        resp = await self._openai_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0 if new_family else 0.0,
            timeout=self.settings.judge_timeout,
            **token_kwarg,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return text, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)

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

        prompt = JUDGE_PROMPT.format(
            question=question,
            expected=expected_answer,
            actual=actual_answer,
        )

        try:
            text, input_tokens, output_tokens = await self._call_model(prompt)
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
        cost = self._compute_cost(input_tokens, output_tokens)

        return JudgeResult(
            score=score,
            passed=passed,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            cost_usd=cost,
        )

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = JUDGE_PRICING.get(self.model)
        if pricing is None:
            # Sensible default per provider when a model isn't in the table.
            pricing = (2.50, 10.0) if self.provider == "openai" else (15.0, 75.0)
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
