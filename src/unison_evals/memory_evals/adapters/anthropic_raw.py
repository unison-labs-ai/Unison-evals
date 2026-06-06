"""Anthropic raw adapter — direct Claude call with no memory layer.

What this system is:
  A direct call to Anthropic's Claude (claude-sonnet-4-5 by default) with no
  retrieval, no persistent memory, and no tool use. The model answers from its
  training weights plus any oracle_context prepended to the prompt.

Pricing (Sonnet 4.5, as of 2026-05):
  Input:  $3.00 per million tokens
  Output: $15.00 per million tokens
  Source: https://www.anthropic.com/pricing

This is a NO-MEMORY baseline. On memory benchmarks (LongMemEval,
MemoryAgentBench) it will lose to systems with persistent memory — that is
the point. When Unison-with-brain beats this adapter, the delta is the
structural advantage of the brain layer, not a model-quality difference
(since Unison also uses Claude under the hood).

Track 2 (oracle): oracle_context is prepended inside <context> delimiters.
Track 3 (no-memory baseline): the question is sent alone; the model must rely
  solely on its training knowledge.

Setup:
  Set ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from .base import AgentAdapter

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

# Anthropic claude-sonnet-4-5 pricing (per million tokens)
# Update when pricing changes — affects $/task numbers in the leaderboard.
SONNET_INPUT_USD_PER_MTOK = 3.0
SONNET_OUTPUT_USD_PER_MTOK = 15.0


class AnthropicRawAdapter(AgentAdapter):
    """Track 2/3 AgentAdapter: Claude called directly with no memory or retrieval.

    One-shot Q&A. oracle_context is prepended to the prompt when provided
    (Track 2 — oracle). When oracle_context is None (Track 3 / no-memory
    baseline), the model answers from training knowledge only.
    """

    name = "anthropic-raw"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncAnthropic | None = None

    async def setup(self) -> None:
        if not self.settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — required for the anthropic-raw adapter. "
                "Get a key at https://console.anthropic.com and set ANTHROPIC_API_KEY in .env."
            )
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        logger.debug("anthropic-raw adapter ready, model={}", self.settings.baseline_agent_model)

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

        # No-memory baseline: when seed_docs is provided we inline them as
        # context (no brain to seed). Same prompt path as oracle_context.
        effective_context = oracle_context or _format_seed_docs(seed_docs)
        prompt = _build_prompt(question, effective_context)
        start = time.perf_counter()

        try:
            response = await self._client.messages.create(
                model=self.settings.baseline_agent_model,
                max_tokens=1500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            answer_text = response.content[0].text if response.content else ""
            usage = response.usage
            cost = (
                usage.input_tokens * SONNET_INPUT_USD_PER_MTOK
                + usage.output_tokens * SONNET_OUTPUT_USD_PER_MTOK
            ) / 1_000_000.0

            inlined_chars = len(oracle_context or "") + (
                sum(len(d.body) for d in seed_docs) if seed_docs else 0
            )
            inlined_tokens = usage.input_tokens
            return AdapterResult(
                answer=answer_text,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                raw={
                    "model": response.model,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                    },
                    "has_oracle_context": oracle_context is not None,
                    "inlined_chars": inlined_chars,
                    "inlined_tokens": inlined_tokens,
                },
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("anthropic-raw answer failed: {}", e)
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                raw={"has_oracle_context": oracle_context is not None},
                error=str(e),
            )

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _format_seed_docs(seed_docs: list[Document] | None) -> str | None:
    """Flatten seed_docs into an inline context string for no-brain adapters.

    Raw-model baselines have no persistent store; treating seed_docs as inline
    context is the honest fallback (the model gets the haystack via the prompt).
    Returns None when seed_docs is None/empty so we don't wrap an empty context.
    """
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
