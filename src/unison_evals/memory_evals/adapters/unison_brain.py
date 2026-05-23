"""unison_brain — BrainAdapter that hits Unison's `agents.cortex.search` tRPC.

This is the apples-to-apples Track 1 measurement of Unison's *brain* layer
(hybrid BM25 + dense + RRF + rerank + recency/importance boosts) without
the agent. Compares against pgvector_naive, mem0, letta, zep on the same
corpus to show whether Unison's retrieval pipeline beats commodity
vector search.

What this system *is* (for fairness context):
  Unison's brain is a Postgres-native hybrid retrieval system: pgvector
  for dense embeddings, tsvector for BM25 lexical, RRF fusion, optional
  cross-encoder reranking (Voyage / Cohere / local bge), kind boosts
  (wiki_page > note > raw), recency decay (configurable half-life),
  and importance multipliers from cortex_facts. See the Unison API docs
  for the full pipeline at packages/ai/src/cortex/postgres-fs.ts.

Important limitation in v0.1:
  This adapter does NOT implement bulk ingest. The Unison API does not
  yet expose an unauthenticated bulk-write endpoint suitable for an
  eval harness — every existing write path goes through the user-scoped
  agents.cortex.write tRPC, which is fine for one-off writes but slow
  for thousands. The runner should pre-seed the eval tenant via the
  brain-cli `import` command (or wait for the eval-side ingest endpoint
  documented in the Unison ingest API (not yet public)) before invoking this adapter.

  When `ingest()` is called with a non-empty document list the adapter
  raises a clear error so misuse is loud, not silent.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

from ...config import get_settings
from ...types import BrainSearchResult, Document, RetrievedChunk
from ._url_utils import is_localhost_url
from .base import BrainAdapter


class UnisonBrainAdapter(BrainAdapter):
    name = "unison-brain"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        is_localhost = is_localhost_url(self.settings.unison_api_url)
        has_jwt = bool(self.settings.unison_jwt)

        if not has_jwt and not is_localhost:
            raise RuntimeError(
                "UNISON_JWT not set. Required when UNISON_API_URL is not localhost.\n"
                "For local dev: set UNISON_API_URL=http://localhost:3001 and configure\n"
                "UNISON_LOCAL_EVAL_TENANT_ID on your local Unison server — JWT becomes unnecessary."
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if has_jwt:
            headers["Authorization"] = f"Bearer {self.settings.unison_jwt}"

        self._client = httpx.AsyncClient(
            base_url=self.settings.unison_api_url,
            headers=headers,
            timeout=self.settings.adapter_timeout,
        )
        # Probe /health to fail fast on a wrong URL or down API.
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Unison API health check failed at {self.settings.unison_api_url}: {e}"
            ) from e

    async def ingest(self, docs: list[Document]) -> None:
        # Loud failure: forces the eval operator to wire the right ingest
        # path rather than silently producing degraded numbers.
        if not docs:
            return
        raise NotImplementedError(
            "unison_brain.ingest is not implemented in v0.1. The Unison API "
            "does not yet expose a bulk-ingest endpoint. Pre-seed the eval "
            "tenant via `brain-cli import` before "
            "running brain retrieval evals against unison-brain. See the "
            "adapter docstring for the v0.2 endpoint plan."
        )

    async def reset(self) -> None:
        # No-op — corpus management lives outside this adapter (see ingest).
        return None

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        assert self._client is not None, "setup() must be called first"
        start = time.perf_counter()

        # tRPC v10 query encoding: GET /trpc/<procedure>?input=<urlencoded JSON>
        # The double-`json` wrapping is the v10 superjson-compatible shape.
        payload = {"json": {"query": query, "k": k}}
        encoded = quote(json.dumps(payload, separators=(",", ":")))
        url = f"/trpc/agents.cortex.search?input={encoded}"

        try:
            response = await self._client.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if response.status_code != 200:
                logger.warning("unison_brain non-200 response", status=response.status_code)
                return BrainSearchResult(
                    chunks=[],
                    latency_ms=elapsed_ms,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            hits = _extract_trpc_hits(data)
            chunks = _hits_to_chunks(hits)
            return BrainSearchResult(
                chunks=chunks,
                latency_ms=elapsed_ms,
                cost_usd=0.0,  # query cost is server-side; not surfaced here
                raw={"hits_count": len(hits)},
            )
        except httpx.HTTPError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return BrainSearchResult(
                chunks=[],
                latency_ms=elapsed_ms,
                error=f"HTTP error: {e}",
            )

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _extract_trpc_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """tRPC v10 response: {"result":{"data":{"json": <result>}}}.

    The brain.search() result is an array of RankedHit objects shaped like:
      {doc: {id, path, bodyMd, ...}, score?, sources?, ...}
    Defensive: accept either the wrapped or already-unwrapped shape.
    """
    if not isinstance(payload, dict):
        return []
    result = payload.get("result", payload)
    data = result.get("data", result) if isinstance(result, dict) else result
    if isinstance(data, dict) and "json" in data:
        data = data["json"]
    if isinstance(data, list):
        return [h for h in data if isinstance(h, dict)]
    return []


def _hits_to_chunks(hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Map RankedHit[] → RetrievedChunk[] with stable rank assignment."""
    chunks: list[RetrievedChunk] = []
    for i, hit in enumerate(hits):
        doc = hit.get("doc") or {}
        path = str(doc.get("path") or doc.get("id") or "")
        body = str(doc.get("bodyMd") or doc.get("body") or doc.get("text") or "")
        # `score` may be on the hit or absent — fall back to 1/(rank+1).
        score_raw = hit.get("score")
        score = float(score_raw) if isinstance(score_raw, int | float) else 1.0 / (i + 1)
        chunks.append(
            RetrievedChunk(
                doc_path=path,
                chunk_text=body,
                score=score,
                rank=i + 1,
                raw={"sources": hit.get("sources", [])},
            )
        )
    return chunks
