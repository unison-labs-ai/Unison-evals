"""OpenAI GPT-5 adapter — direct chat completion with no memory layer.

What this system is:
  A direct call to OpenAI's GPT-5 via the OpenAI Chat Completions API with no
  retrieval, no persistent memory, and no tool use. The model answers from its
  training weights plus any oracle_context prepended to the prompt.

Pricing (GPT-5, as of 2026-05):
  Input:  $5.00 per million tokens  (placeholder — verify at platform.openai.com/docs/pricing)
  Output: $15.00 per million tokens (placeholder — verify at platform.openai.com/docs/pricing)

  NOTE: GPT-5 pricing was not publicly confirmed at the time of writing (May 2026).
  These figures are documented placeholders based on analyst estimates. Update
  when OpenAI publishes official pricing. The cost column in the leaderboard
  will reflect whatever OpenAI reports in the usage object (prompt_tokens /
  completion_tokens), so token counts are always accurate even if the per-token
  rate needs adjustment.

This is a NO-MEMORY baseline. On memory benchmarks (LongMemEval,
MemoryAgentBench) it will lose to systems with persistent memory — that is
the point. When Unison-with-brain beats this adapter, the delta is the
structural advantage of the brain layer, not a model-quality difference.

Track 2 (oracle): oracle_context is prepended inside <context> delimiters.
Track 3 (no-memory baseline): the question is sent alone; the model must rely
  solely on its training knowledge.

Setup:
  Set OPENAI_API_KEY in .env.
  Optionally set OPENAI_CHAT_MODEL (default: gpt-5).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from .base import AgentAdapter

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# GPT-5 pricing per million tokens (placeholder as of 2026-05 — see docstring).
GPT5_INPUT_USD_PER_MTOK = 5.0
GPT5_OUTPUT_USD_PER_MTOK = 15.0


class OpenAIGpt5Adapter(AgentAdapter):
    """Track 2/3 AgentAdapter: GPT-5 called directly with no memory or retrieval.

    One-shot Q&A. oracle_context is prepended to the prompt when provided
    (Track 2 — oracle). When oracle_context is None (Track 3 / no-memory
    baseline), the model answers from training knowledge only.
    """

    name = "openai-gpt5"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncOpenAI | None = None

    async def setup(self) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for the openai-gpt5 adapter. "
                "Get a key at https://platform.openai.com and set OPENAI_API_KEY in .env."
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        logger.debug("openai-gpt5 adapter ready, model={}", self.settings.openai_chat_model)

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
            response = await self._client.chat.completions.create(
                model=self.settings.openai_chat_model,
                max_completion_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            choice = response.choices[0] if response.choices else None
            answer_text = choice.message.content or "" if choice else ""
            usage = response.usage

            cost = 0.0
            raw_usage: dict[str, int] = {}
            input_tok = 0
            output_tok = 0
            if usage is not None:
                input_tok = usage.prompt_tokens
                output_tok = usage.completion_tokens
                raw_usage = {
                    "prompt_tokens": input_tok,
                    "completion_tokens": output_tok,
                }
                cost = (
                    input_tok * GPT5_INPUT_USD_PER_MTOK + output_tok * GPT5_OUTPUT_USD_PER_MTOK
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
                    "model": response.model,
                    "usage": raw_usage,
                    "has_oracle_context": oracle_context is not None,
                    "inlined_chars": inlined_chars,
                    "inlined_tokens": input_tok,
                },
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("openai-gpt5 answer failed: {}", e)
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
