"""Mem0 adapter — BrainAdapter (Track 1) and AgentAdapter (Track 2/3).

What this system is:
  Mem0 (https://mem0.ai) is a memory layer for LLM agents. It ingests text or
  conversation turns, extracts facts, and stores them as "memories" scoped to a
  user namespace. Retrieval uses Mem0's proprietary hybrid search (dense + sparse
  fusion). Two API surfaces are relevant here: the managed cloud SaaS accessed via
  MemoryClient (used in this adapter) and a self-hostable open-source variant.
  The cloud client requires a MEM0_API_KEY; memories are isolated per user_id,
  which we exploit to get fast per-question reset semantics: each question gets a
  fresh UUID user_id so there is never cross-contamination without a slow delete_all.

Setup notes:
  Install: mem0ai>=0.1.0
  Set MEM0_API_KEY in .env.
  Cloud endpoint: https://api.mem0.ai (default in the SDK).
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, BrainSearchResult, Document, RetrievedChunk
from .base import AgentAdapter, BrainAdapter

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

# Anthropic claude-sonnet-4-5 pricing (per million tokens) — used for AgentAdapter cost.
SONNET_INPUT_USD_PER_MTOK = 3.0
SONNET_OUTPUT_USD_PER_MTOK = 15.0

# Default number of memories to retrieve when building agent context.
_AGENT_RETRIEVE_K = 10


class Mem0BrainAdapter(BrainAdapter):
    """Track 1 BrainAdapter backed by Mem0 cloud MemoryClient.

    Each eval question gets a fresh UUID user_id (set by reset()) so
    corpora never bleed across questions — no delete_all() needed.
    """

    name = "mem0"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Any = None
        self._current_user_id: str = f"unison-evals-{uuid.uuid4().hex}"

    async def setup(self) -> None:
        if not self.settings.mem0_api_key:
            raise RuntimeError(
                "MEM0_API_KEY not set — required for the Mem0 adapter. "
                "Get a key at https://mem0.ai and set MEM0_API_KEY in .env."
            )
        from mem0 import MemoryClient

        self._client = MemoryClient(api_key=self.settings.mem0_api_key)
        logger.debug("mem0 brain adapter ready, initial user_id={}", self._current_user_id)

    async def reset(self) -> None:
        """Rotate to a fresh user_id — effectively a clean-slate without delete_all."""
        self._current_user_id = f"unison-evals-{uuid.uuid4().hex}"
        logger.debug("mem0 user_id rotated to {}", self._current_user_id)

    async def ingest(self, docs: list[Document]) -> None:
        assert self._client is not None, "setup() must be called first"
        if not docs:
            return

        for doc in docs:
            try:
                self._client.add(
                    messages=[{"role": "user", "content": doc.body}],
                    user_id=self._current_user_id,
                    metadata=doc.metadata,
                )
            except Exception as e:
                logger.warning("mem0 ingest failed for doc {}: {}", doc.path, e)

        logger.debug("mem0 ingested {} docs for user_id={}", len(docs), self._current_user_id)

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        assert self._client is not None, "setup() must be called first"
        start = time.perf_counter()
        try:
            results = self._client.search(
                query=query,
                filters={"user_id": self._current_user_id},
                limit=k,
                version="v2",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # mem0 v2 returns either a bare list or a dict like {"results": [...]}.
            # Normalize before iterating so downstream code can assume dicts.
            items = _normalize_mem0_results(results)

            chunks: list[RetrievedChunk] = []
            for rank, item in enumerate(items, start=1):
                # Mem0 result fields: id, memory (text), score
                mem_id: str = str(item.get("id", ""))
                mem_text: str = str(item.get("memory", ""))
                score: float = float(item.get("score", 0.0))
                # Use the memory id as doc_path if present; fall back to a
                # stable hash of the memory text so the metric layer has a
                # consistent key even if id is absent.
                doc_path = mem_id if mem_id else _stable_id(mem_text)
                chunks.append(
                    RetrievedChunk(
                        doc_path=doc_path,
                        chunk_text=mem_text,
                        score=score,
                        rank=rank,
                        raw={"mem0_id": mem_id},
                    )
                )

            return BrainSearchResult(
                chunks=chunks,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("mem0 search failed: {}", e)
            return BrainSearchResult(
                chunks=[],
                latency_ms=elapsed_ms,
                error=str(e),
            )

    async def teardown(self) -> None:
        # MemoryClient has no close method; nothing to clean up.
        self._client = None


class Mem0AgentAdapter(AgentAdapter):
    """Track 2/3 AgentAdapter: Mem0 for retrieval + Anthropic Claude for answering.

    Pattern:
      1. If oracle_context is provided (Track 2), write it as a memory first.
      2. Retrieve top-k memories via Mem0 search.
      3. Pass retrieved memories as system context to Claude.
      4. Track token cost from the Anthropic response.

    Uses a fresh UUID user_id per answer() call so each question is isolated.
    """

    name = "mem0-agent"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._mem0: Any = None
        self._anthropic: AsyncAnthropic | None = None

    async def setup(self) -> None:
        if not self.settings.mem0_api_key:
            raise RuntimeError(
                "MEM0_API_KEY not set — required for the Mem0 agent adapter. "
                "Get a key at https://mem0.ai and set MEM0_API_KEY in .env."
            )
        if not self.settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — required for the Mem0 agent adapter "
                "to call Claude for answer generation."
            )
        from anthropic import AsyncAnthropic
        from mem0 import MemoryClient

        self._mem0 = MemoryClient(api_key=self.settings.mem0_api_key)
        self._anthropic = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        logger.debug("mem0-agent adapter ready")

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        assert self._mem0 is not None and self._anthropic is not None, (
            "setup() must be called first"
        )

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # Fresh user_id per question — no cross-contamination.
        user_id = f"unison-evals-{uuid.uuid4().hex}"
        start = time.perf_counter()

        try:
            # Track 2: write the oracle context as a memory so Mem0 can retrieve it.
            if oracle_context is not None:
                try:
                    self._mem0.add(
                        messages=[{"role": "user", "content": oracle_context}],
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning("mem0-agent oracle_context add failed: {}", e)

            # Track 3: ingest seed_docs into Mem0 before answering so the
            # retrieval step below can find them.
            if seed_docs is not None:
                for doc in seed_docs:
                    try:
                        self._mem0.add(
                            messages=[{"role": "user", "content": doc.body}],
                            user_id=user_id,
                            metadata={"path": doc.path, **doc.metadata},
                        )
                    except Exception as e:
                        logger.warning("mem0-agent seed_doc ingest failed for {}: {}", doc.path, e)

            # Retrieve relevant memories for the question.
            try:
                mem_results = self._mem0.search(
                    query=question,
                    filters={"user_id": user_id},
                    limit=_AGENT_RETRIEVE_K,
                    version="v2",
                )
                mem_items = _normalize_mem0_results(mem_results)
                memory_texts = [str(m.get("memory", "")) for m in mem_items if m.get("memory")]
            except Exception as e:
                logger.warning("mem0-agent search failed: {}", e)
                memory_texts = []

            # Build system prompt from retrieved memories.
            if memory_texts:
                system = (
                    "You are a helpful assistant with access to the following memories. "
                    "Use them to answer the question accurately.\n\n"
                    "<memories>\n" + "\n".join(f"- {t}" for t in memory_texts) + "\n</memories>"
                )
            else:
                system = "You are a helpful assistant. Answer the following question."

            response = await self._anthropic.messages.create(
                model=self.settings.baseline_agent_model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": question}],
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer_text = response.content[0].text if response.content else ""

            # Cost from real token usage.
            usage = response.usage
            cost = (
                usage.input_tokens * SONNET_INPUT_USD_PER_MTOK
                + usage.output_tokens * SONNET_OUTPUT_USD_PER_MTOK
            ) / 1_000_000.0

            # mem0_ops_count = number of add() calls (one per seed_doc / oracle chunk)
            # plus the search() call — this is the Mem0 API call footprint per question.
            n_adds = len(seed_docs) if seed_docs is not None else (1 if oracle_context else 0)
            mem0_ops_count = n_adds + 1  # +1 for the search call

            return AdapterResult(
                answer=answer_text,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                raw={
                    "user_id": user_id,
                    "memories_retrieved": len(memory_texts),
                    "mem0_ops_count": mem0_ops_count,
                    "model": response.model,
                    "usage": {"input": usage.input_tokens, "output": usage.output_tokens},
                },
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("mem0-agent answer failed: {}", e)
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                raw={"user_id": user_id},
                error=str(e),
            )

    async def teardown(self) -> None:
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None
        self._mem0 = None


def _stable_id(text: str) -> str:
    """Return a deterministic short id for a memory text string.

    Used as doc_path fallback when the Mem0 result has no id field.
    """
    import hashlib

    return "mem0-" + hashlib.sha1(text.encode()).hexdigest()[:16]


def _normalize_mem0_results(raw: Any) -> list[dict[str, Any]]:
    """Coerce mem0 search() output into a list of memory dicts.

    v1: returns a bare list of {id, memory, score, ...} dicts.
    v2: returns a dict like {"results": [...]} — the inner list has the
        same item shape. Some accounts also return the bare list under v2.

    Anything else (string, None, malformed) → empty list, so callers never
    see a TypeError on `item.get(...)`.
    """
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        inner = raw.get("results")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []
