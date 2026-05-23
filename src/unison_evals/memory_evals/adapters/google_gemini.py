"""Google Gemini adapter — direct Gemini call with no memory layer.

What this system is:
  A direct call to Google's Gemini 2.5 Pro via the google-genai SDK with no
  retrieval, no persistent memory, and no tool use. The model answers from its
  training weights plus any oracle_context prepended to the prompt.

Pricing (Gemini 2.5 Pro, as of 2026-05):
  Input:  $1.25 per million tokens (prompts ≤200k tokens)
  Output: $10.00 per million tokens
  Source: https://ai.google.dev/pricing

  NOTE: Google prices Gemini 2.5 Pro in two tiers based on prompt length.
  Prompts over 200k tokens are charged at $2.50/$15.00. The leaderboard
  uses the standard tier since eval questions + short oracle context are
  well under 200k tokens. Update this docstring and the constants below
  if pricing changes.

This is a NO-MEMORY baseline. On memory benchmarks (LongMemEval,
MemoryAgentBench) it will lose to systems with persistent memory — that is
the point. When Unison-with-brain beats this adapter, the delta is the
structural advantage of the brain layer, not a model-quality difference.

Track 2 (oracle): oracle_context is prepended inside <context> delimiters.
Track 3 (no-memory baseline): the question is sent alone; the model must rely
  solely on its training knowledge.

Setup:
  pip install google-genai>=1.0.0
  Set GOOGLE_API_KEY in .env.
  Optionally set GEMINI_MODEL (default: gemini-2.5-pro).
"""

from __future__ import annotations

import time

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from .base import AgentAdapter

# Gemini 2.5 Pro pricing per million tokens (standard tier, prompts ≤200k).
GEMINI_INPUT_USD_PER_MTOK = 1.25
GEMINI_OUTPUT_USD_PER_MTOK = 10.0


class GoogleGeminiAdapter(AgentAdapter):
    """Track 2/3 AgentAdapter: Gemini 2.5 Pro called directly with no memory or retrieval.

    One-shot Q&A. oracle_context is prepended to the prompt when provided
    (Track 2 — oracle). When oracle_context is None (Track 3 / no-memory
    baseline), the model answers from training knowledge only.
    """

    name = "google-gemini"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: object | None = None

    async def setup(self) -> None:
        if not self.settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set — required for the google-gemini adapter. "
                "Get a key at https://aistudio.google.com and set GOOGLE_API_KEY in .env."
            )
        from google import genai

        self._client = genai.Client(api_key=self.settings.google_api_key)
        logger.debug("google-gemini adapter ready, model={}", self.settings.gemini_model)

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        assert self._client is not None, "setup() must be called first"

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # No-memory baseline: inline seed_docs as context (no brain to seed).
        effective_context = oracle_context or _format_seed_docs(seed_docs)
        prompt = _build_prompt(question, effective_context)
        start = time.perf_counter()

        try:
            from google.genai import types as genai_types

            response = await self._client.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=1500,
                    temperature=0,
                ),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            answer_text = response.text or ""
            usage = response.usage_metadata

            cost = 0.0
            raw_usage: dict[str, int] = {}
            input_tok = 0
            output_tok = 0
            if usage is not None:
                input_tok = getattr(usage, "prompt_token_count", 0) or 0
                output_tok = getattr(usage, "candidates_token_count", 0) or 0
                raw_usage = {
                    "prompt_token_count": input_tok,
                    "candidates_token_count": output_tok,
                }
                cost = (
                    input_tok * GEMINI_INPUT_USD_PER_MTOK + output_tok * GEMINI_OUTPUT_USD_PER_MTOK
                ) / 1_000_000.0

            inlined_chars = len(oracle_context or "") + (
                sum(len(d.body) for d in seed_docs) if seed_docs else 0
            )
            return AdapterResult(
                answer=answer_text,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                input_tokens=input_tok,
                output_tokens=output_tok,
                tokens_unavailable=usage is None,
                raw={
                    "model": self.settings.gemini_model,
                    "usage": raw_usage,
                    "has_oracle_context": oracle_context is not None,
                    "inlined_chars": inlined_chars,
                    "inlined_tokens": input_tok,
                },
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("google-gemini answer failed: {}", e)
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                raw={"has_oracle_context": oracle_context is not None},
                error=str(e),
            )

    async def teardown(self) -> None:
        self._client = None


def _format_seed_docs(seed_docs: list[Document] | None) -> str | None:
    """Flatten seed_docs into inline context — raw-model adapters have no brain."""
    if not seed_docs:
        return None
    return "\n\n".join(f"[Document: {d.path}]\n{d.body}" for d in seed_docs)


def _build_prompt(question: str, oracle_context: str | None) -> str:
    """Build the final prompt string.

    Mirrors the pattern in ClaudeCodeAdapter._build_prompt so oracle-track
    prompts are consistent across adapters.
    """
    if oracle_context is None:
        return question
    return (
        "Use the following context to answer the question. Do not use any "
        "other knowledge or tools.\n\n"
        f"<context>\n{oracle_context}\n</context>\n\n"
        f"Question: {question}"
    )
