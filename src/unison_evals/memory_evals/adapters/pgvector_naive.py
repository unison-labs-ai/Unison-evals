"""pgvector_naive — honest baseline BrainAdapter.

Pure cosine similarity over OpenAI text-embedding-3-small (1536 dim) in a
single Postgres table. No reranking, no hybrid, no chunking — every doc is
its own chunk. This is what you get with "just a vector store" and nothing
else, included so the leaderboard has a credible floor that other systems
must beat.

What this system *is* (for fairness context):
  pgvector is the de facto Postgres vector extension. The "naive" config
  here is what a developer would write in an afternoon: HNSW index with
  default parameters, cosine similarity, top-k SELECT. No filters, no
  rerank, no chunking strategy. We use this rather than a managed vector
  DB because (a) it's the most-cited reference in 2025-2026 benchmarks
  and (b) it has zero API cost beyond OpenAI embeddings.

Setup notes:
  Requires a running Postgres instance with the pgvector extension.
  Quick local dev:
    docker run -d --name pgvec -p 5433:5432 \\
      -e POSTGRES_PASSWORD=evals pgvector/pgvector:pg17
  Set PGVECTOR_DSN in .env.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...config import get_settings
from ...types import BrainSearchResult, Document, RetrievedChunk
from .base import BrainAdapter

if TYPE_CHECKING:
    import asyncpg
    from openai import AsyncOpenAI

# Bigger batches saturate the OpenAI embeddings endpoint without hitting
# the per-request token cap. 96 docs * ~500 tokens = ~48k tokens, well under
# the 8192-input-batch limit on text-embedding-3-small.
EMBED_BATCH_SIZE = 96
INSERT_BATCH_SIZE = 256


class PgvectorNaiveBrainAdapter(BrainAdapter):
    name = "pgvector-naive"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pool: asyncpg.Pool | None = None
        self._openai: AsyncOpenAI | None = None

    async def setup(self) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for pgvector_naive embeddings.")

        import asyncpg
        from openai import AsyncOpenAI
        from pgvector.asyncpg import register_vector

        self._openai = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self._pool = await asyncpg.create_pool(
            self.settings.pgvector_dsn,
            min_size=1,
            max_size=4,
            init=register_vector,
        )

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS unison_evals_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    doc_path TEXT NOT NULL,
                    body TEXT NOT NULL,
                    embedding vector({self.settings.openai_embedding_dim}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            # HNSW index — pgvector default params (m=16, ef_construction=64).
            # We deliberately don't tune; this is the "naive" baseline.
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS unison_evals_chunks_embed_idx
                ON unison_evals_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )

    async def reset(self) -> None:
        assert self._pool is not None, "setup() must be called first"
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE unison_evals_chunks RESTART IDENTITY;")

    async def ingest(self, docs: list[Document]) -> None:
        assert self._pool is not None and self._openai is not None
        if not docs:
            return

        # Embed in batches to keep request size sane.
        embeddings: list[list[float]] = []
        for i in range(0, len(docs), EMBED_BATCH_SIZE):
            batch = docs[i : i + EMBED_BATCH_SIZE]
            inputs = [_truncate_for_embedding(d.body) for d in batch]
            resp = await self._openai.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=inputs,
            )
            embeddings.extend([e.embedding for e in resp.data])

        # Bulk INSERT in batches.
        async with self._pool.acquire() as conn:
            for i in range(0, len(docs), INSERT_BATCH_SIZE):
                batch_docs = docs[i : i + INSERT_BATCH_SIZE]
                batch_embeds = embeddings[i : i + INSERT_BATCH_SIZE]
                await conn.executemany(
                    """
                    INSERT INTO unison_evals_chunks (doc_path, body, embedding, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    [
                        (d.path, d.body, emb, _safe_json(d.metadata))
                        for d, emb in zip(batch_docs, batch_embeds, strict=True)
                    ],
                )
        logger.debug("pgvector_naive ingested {} docs", len(docs))

    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        assert self._pool is not None and self._openai is not None
        start = time.perf_counter()
        try:
            embed_resp = await self._openai.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=_truncate_for_embedding(query),
            )
            qvec = embed_resp.data[0].embedding

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        doc_path,
                        body,
                        1 - (embedding <=> $1) AS score
                    FROM unison_evals_chunks
                    ORDER BY embedding <=> $1
                    LIMIT $2;
                    """,
                    qvec,
                    k,
                )

            chunks = [
                RetrievedChunk(
                    doc_path=r["doc_path"],
                    chunk_text=r["body"],
                    score=float(r["score"]),
                    rank=i + 1,
                )
                for i, r in enumerate(rows)
            ]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Embedding cost: text-embedding-3-small is $0.02 per 1M tokens.
            # ~rough estimate: query token count.
            embed_tokens = embed_resp.usage.total_tokens if embed_resp.usage else 0
            cost = embed_tokens * 0.02 / 1_000_000.0
            return BrainSearchResult(
                chunks=chunks,
                latency_ms=elapsed_ms,
                cost_usd=cost,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("pgvector_naive search failed: {}", e)
            return BrainSearchResult(
                chunks=[],
                latency_ms=elapsed_ms,
                error=str(e),
            )

    async def teardown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _truncate_for_embedding(text: str, max_chars: int = 24_000) -> str:
    """text-embedding-3-small caps at ~8192 tokens; ~24k chars is a safe cap
    that keeps the encoder fast without truncating most realistic docs."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _safe_json(metadata: dict[str, Any]) -> str:
    """Serialize metadata to JSON for the JSONB column.

    asyncpg's executemany doesn't auto-encode dicts; we pass the JSON string
    and cast to ::jsonb in the SQL.
    """
    import json

    try:
        return json.dumps(metadata, default=str)
    except (TypeError, ValueError):
        return "{}"
