"""letta — BrainAdapter backed by Letta archival memory.

What this system is:
  Letta (formerly MemGPT, https://letta.com) is an open-source stateful-agent
  framework built around a structured memory hierarchy: core memory (in-context
  persona/human blocks), archival memory (unlimited, vector-indexed long-term
  store), and recall memory (message history). The archival memory layer is the
  one we evaluate here — it ingests arbitrary text passages and answers semantic
  search queries using embedding-based similarity (via the agent's configured
  embedding model, e.g. openai/text-embedding-3-small). Letta exposes both a
  cloud offering (letta.ai) and a self-hosted server; we default to the cloud
  path with LETTA_API_KEY. Each benchmark question gets a fresh ephemeral agent
  (reset() = delete old + create new) so corpora don't bleed between questions.

SDK: letta-client >= 1.10.3 (AsyncLetta, client.agents.passages.create /
     .search). Install: pip install letta-client.

Auth / config:
  LETTA_API_KEY — required for cloud (https://app.letta.com → API keys)
  LETTA_BASE_URL — override to point at a self-hosted server (optional)

Agent model: LETTA_AGENT_MODEL (default "openai/gpt-4o-mini") selects the
  LLM attached to the ephemeral agent.  Only archival-memory operations are
  exercised in Brain mode so the LLM is never invoked — the model choice only
  affects the embedding config default.

Embedding: LETTA_AGENT_EMBEDDING (default "openai/text-embedding-3-small").
  Both defaults match Letta cloud defaults as of 2026.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...config import get_settings
from ...types import BrainSearchResult, Document, RetrievedChunk
from .base import BrainAdapter

if TYPE_CHECKING:
    from letta_client import AsyncLetta

# Letta cloud free tier rate-limits archival_memory operations.
# Empirically ~3-5 ops/sec is the ceiling before 429s. Throttle ingest at
# ~3 docs/sec (300ms between calls) and retry search up to 4x with backoff.
_LETTA_INGEST_DELAY_S = 0.3
_LETTA_MAX_RETRIES = 4
_LETTA_RETRY_BASE_S = 1.0


class LettaBrainAdapter(BrainAdapter):
    """Track 1 brain adapter: ingest docs into Letta archival memory, search
    with semantic similarity.  One ephemeral agent per question; reset()
    deletes the prior agent and creates a fresh one."""

    name = "letta"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncLetta | None = None
        self._agent_id: str | None = None

    async def setup(self) -> None:
        if not self.settings.letta_api_key:
            raise RuntimeError(
                "LETTA_API_KEY not set — required for the Letta brain adapter. "
                "Get one at https://app.letta.com → API keys."
            )

        from letta_client import AsyncLetta

        kwargs: dict[str, Any] = {"api_key": self.settings.letta_api_key}
        if self.settings.letta_base_url:
            kwargs["base_url"] = self.settings.letta_base_url

        self._client = AsyncLetta(**kwargs)

    async def reset(self) -> None:
        assert self._client is not None, "setup() must be called first"
        await self._delete_current_agent()
        await self._create_agent()

    async def ingest(self, docs: list[Document]) -> None:
        assert self._client is not None and self._agent_id is not None
        if not docs:
            return

        for doc in docs:
            text = f"[{doc.path}] {doc.body}" if doc.path else doc.body
            await self._ingest_one_with_retry(text, doc.path)
            await asyncio.sleep(_LETTA_INGEST_DELAY_S)

        logger.debug("letta ingested {} docs into agent {}", len(docs), self._agent_id)

    async def _ingest_one_with_retry(self, text: str, doc_path: str) -> None:
        """Insert one passage with exponential backoff on 429."""
        assert self._client is not None and self._agent_id is not None
        for attempt in range(_LETTA_MAX_RETRIES):
            try:
                await self._client.agents.passages.create(
                    agent_id=self._agent_id,
                    text=text,
                )
                return
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
                if is_rate_limit and attempt < _LETTA_MAX_RETRIES - 1:
                    backoff = _LETTA_RETRY_BASE_S * (2**attempt)
                    logger.debug(
                        "letta ingest 429 for {} (attempt {}/{}); sleeping {:.1f}s",
                        doc_path,
                        attempt + 1,
                        _LETTA_MAX_RETRIES,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.warning("letta: failed to ingest doc {}: {}", doc_path, e)
                return

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        assert self._client is not None and self._agent_id is not None
        start = time.perf_counter()
        try:
            resp = await self._search_with_retry(query=query, k=k)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            chunks: list[RetrievedChunk] = []
            for i, result in enumerate(resp.results[:k]):
                # result.content has our "[path] body" format; extract path.
                content = result.content
                doc_path, chunk_text = _split_tagged_content(content)
                chunks.append(
                    RetrievedChunk(
                        doc_path=doc_path,
                        chunk_text=chunk_text,
                        score=float(k - i) / k,
                        rank=i + 1,
                        raw={"letta_id": result.id, "timestamp": result.timestamp},
                    )
                )

            return BrainSearchResult(
                chunks=chunks,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("letta search failed: {}", e)
            return BrainSearchResult(
                chunks=[],
                latency_ms=elapsed_ms,
                error=str(e),
            )

    async def _search_with_retry(self, query: str, k: int) -> Any:
        """Run archival search with exponential backoff on 429."""
        assert self._client is not None and self._agent_id is not None
        last_exc: Exception | None = None
        for attempt in range(_LETTA_MAX_RETRIES):
            try:
                return await self._client.agents.passages.search(
                    agent_id=self._agent_id,
                    query=query,
                    top_k=k,
                )
            except Exception as e:
                last_exc = e
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
                if is_rate_limit and attempt < _LETTA_MAX_RETRIES - 1:
                    backoff = _LETTA_RETRY_BASE_S * (2**attempt)
                    logger.debug(
                        "letta search 429 (attempt {}/{}); sleeping {:.1f}s",
                        attempt + 1,
                        _LETTA_MAX_RETRIES,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    async def teardown(self) -> None:
        await self._delete_current_agent()
        if self._client is not None:
            await self._client._client.aclose()  # close httpx.AsyncClient
            self._client = None

    async def _create_agent(self) -> None:
        assert self._client is not None
        agent = await self._client.agents.create(
            name="unison-evals-brain",
            model=self.settings.letta_agent_model,
            embedding=self.settings.letta_agent_embedding,
            include_base_tools=False,
            include_default_source=False,
        )
        self._agent_id = agent.id
        logger.debug("letta created ephemeral agent {}", self._agent_id)

    async def _delete_current_agent(self) -> None:
        if self._client is not None and self._agent_id is not None:
            try:
                await self._client.agents.delete(self._agent_id)
                logger.debug("letta deleted agent {}", self._agent_id)
            except Exception:
                logger.warning("letta: could not delete agent {}", self._agent_id)
            finally:
                self._agent_id = None


def _split_tagged_content(content: str) -> tuple[str, str]:
    """Reverse the '[path] body' encoding written by ingest().

    Returns (doc_path, chunk_text). If the prefix is absent (e.g. doc had no
    path, or content was inserted by another process), doc_path is set to the
    full content string and chunk_text matches.
    """
    if content.startswith("[") and "] " in content:
        bracket_end = content.index("] ")
        doc_path = content[1:bracket_end]
        chunk_text = content[bracket_end + 2 :]
        return doc_path, chunk_text
    return content, content
