"""Zep adapter — BrainAdapter (Track 1).

What this system is:
  Zep (https://www.getzep.com) is a memory layer for LLM agents that combines
  dense vector retrieval with a temporal knowledge graph (Graphiti). It tracks
  facts over time with bitemporal semantics — recording both when a fact was
  true in the world and when Zep learned about it. Zep extracts entities and
  relationships from ingested text, builds a graph of facts (edges between
  entities), and retrieves the most relevant facts at query time via a
  configurable reranker. This makes it structurally the closest competitor to
  Unison's brain among the adapters in this benchmark suite.

  Cloud (api.getzep.com) and self-hosted (open-source Graphiti server) are
  both supported. This adapter defaults to the cloud path.

SDK:
  Package: zep-cloud>=2.0.0 (PyPI)  — verified by import in setup()
  Install: pip install zep-cloud
  Auth: ZEP_API_KEY (required)
  Self-hosted: ZEP_BASE_URL override (optional)

Per-question isolation:
  Each question gets a fresh UUID user_id (set by reset()). Zep scopes its
  knowledge graph per user, so rotating the user_id gives a clean slate
  without any explicit delete call. Fresh UUIDs are cheap — no teardown
  needed between questions.

IMPORTANT — async graph build pipeline:
  Zep's graph.add() call returns immediately, but the actual fact extraction
  and graph building happens in a background async pipeline. Searching too
  soon after ingest will miss facts that haven't been extracted yet.
  The adapter waits ZEP_INGEST_WAIT_SECONDS (default 10.0) after all ingest
  calls complete before returning from ingest(). This is best-effort: on
  heavily-loaded Zep cloud the pipeline may take longer. Tune the setting
  upward if recall looks anomalously low.

  The Zep SDK (as of v2.x) does not expose a synchronous "wait for graph
  build" primitive; polling task status is also not part of the public client
  API, so a fixed sleep is the documented best practice for eval harnesses.

Search scope:
  We use scope="episodes" (the raw ingested text, before fact extraction) as
  the primary retrieval mode because:
    1. Episodes preserve the original document text, which is what the
       Track 1 metrics compare against gold doc paths.
    2. Edge/fact retrieval returns synthesised fact strings that may not
       align with the original doc body — confusing for exact-match metrics.
  Each Episode result has a `content` (text) field and a `score` field.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...config import get_settings
from ...types import BrainSearchResult, Document, RetrievedChunk
from .base import BrainAdapter

if TYPE_CHECKING:
    from zep_cloud.client import Zep


class ZepBrainAdapter(BrainAdapter):
    """Track 1 BrainAdapter backed by Zep cloud (zep-cloud SDK).

    Each eval question gets a fresh UUID user_id (set by reset()) so
    corpora never bleed across questions — no user delete needed.
    """

    name = "zep"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Zep | None = None
        self._current_user_id: str = f"unison-evals-{uuid.uuid4().hex}"

    async def setup(self) -> None:
        if not self.settings.zep_api_key:
            raise RuntimeError(
                "ZEP_API_KEY not set — required for the Zep adapter. "
                "Get a key at https://www.getzep.com and set ZEP_API_KEY in .env."
            )

        from zep_cloud.client import Zep

        kwargs: dict[str, Any] = {"api_key": self.settings.zep_api_key}
        if self.settings.zep_base_url:
            kwargs["base_url"] = self.settings.zep_base_url

        self._client = Zep(**kwargs)
        logger.debug("zep brain adapter ready, initial user_id={}", self._current_user_id)

    async def reset(self) -> None:
        """Rotate to a fresh user_id — clean slate without explicit graph deletion."""
        self._current_user_id = f"unison-evals-{uuid.uuid4().hex}"
        logger.debug("zep user_id rotated to {}", self._current_user_id)

    async def ingest(self, docs: list[Document]) -> None:
        """Ingest documents into the user's Zep knowledge graph.

        Each document is added as raw text via graph.add(type="text"). Zep
        processes this in a background pipeline that extracts entities, builds
        graph edges, and indexes episodes. We sleep ZEP_INGEST_WAIT_SECONDS
        after all adds complete to give the pipeline time to run before the
        first search() call.

        WARNING: the wait is best-effort. If the Zep cloud pipeline is slow,
        increase ZEP_INGEST_WAIT_SECONDS in .env to avoid recall degradation.
        """
        assert self._client is not None, "setup() must be called first"
        if not docs:
            return

        try:
            self._client.user.add(user_id=self._current_user_id)
        except Exception as e:
            logger.warning(
                "zep user.add failed for {}: {} (continuing — may already exist)",
                self._current_user_id,
                e,
            )

        for doc in docs:
            try:
                self._client.graph.add(
                    user_id=self._current_user_id,
                    type="text",
                    data=doc.body,
                )
            except Exception as e:
                logger.warning("zep ingest failed for doc {}: {}", doc.path, e)

        wait_s = self.settings.zep_ingest_wait_seconds
        logger.debug(
            "zep ingested {} docs for user_id={}; waiting {:.1f}s for graph build",
            len(docs),
            self._current_user_id,
            wait_s,
        )
        await asyncio.sleep(wait_s)

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        """Search the user's Zep graph for the top-k relevant episodes.

        Uses scope="episodes" so results carry the original ingested text
        rather than synthesised fact strings. Each Episode result has:
          - content  (str)  — the ingested text
          - score    (float | None) — reranker score
          - uuid_    (str)  — stable episode identifier (aliased from "uuid")
        """
        assert self._client is not None, "setup() must be called first"
        start = time.perf_counter()
        try:
            results = self._client.graph.search(
                user_id=self._current_user_id,
                query=query,
                limit=k,
                scope="episodes",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # results is GraphSearchResults; episodes is the relevant list.
            episodes = results.episodes or []

            chunks: list[RetrievedChunk] = []
            for rank, ep in enumerate(episodes[:k], start=1):
                content: str = ep.content if ep.content else ""
                score: float = float(ep.score) if ep.score is not None else 0.0
                ep_id: str = ep.uuid_ if ep.uuid_ else _stable_id(content)
                chunks.append(
                    RetrievedChunk(
                        doc_path=ep_id,
                        chunk_text=content,
                        score=score,
                        rank=rank,
                        raw={"zep_uuid": ep.uuid_},
                    )
                )

            return BrainSearchResult(
                chunks=chunks,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("zep search failed: {}", e)
            return BrainSearchResult(
                chunks=[],
                latency_ms=elapsed_ms,
                error=str(e),
            )

    async def teardown(self) -> None:
        # Zep client has no explicit close method.
        self._client = None


def _stable_id(text: str) -> str:
    """Return a deterministic short id for an episode text string.

    Used as doc_path fallback when a Zep Episode has no uuid field.
    """
    return "zep-" + hashlib.sha1(text.encode()).hexdigest()[:16]
