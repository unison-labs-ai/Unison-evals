"""Unison agent adapter — POSTs to /api/rest/agents/eval-turn.

This is the production agent code path. Track 3 calls without oracleContext
(full agent + brain). Track 2 passes oracleContext (agent reasons from the
provided context only — brain/FS/workspace tools are stripped server-side).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from ._url_utils import is_localhost_url
from .base import AgentAdapter


class UnisonAgentAdapter(AgentAdapter):
    name = "unison-agent"

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
        # Probe /health to fail fast on bad URL.
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Unison API health check failed at {self.settings.unison_api_url}: {e}"
            ) from e

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

        body: dict[str, Any] = {
            "question": question,
            "model": self.settings.default_agent_model,
        }
        if oracle_context is not None:
            body["oracleContext"] = oracle_context
        if seed_docs is not None:
            # kind="raw" skips Unison's extract pipeline (Gemini calls per doc),
            # which is critical for cost in per-question-haystack benchmarks.
            body["seedDocs"] = [
                {"path": doc.path, "body": doc.body, "kind": "raw"} for doc in seed_docs
            ]

        start = time.perf_counter()
        try:
            response = await self._client.post("/api/rest/agents/eval-turn", json=body)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if response.status_code != 200:
                logger.warning(
                    "unison-agent non-200 response",
                    status=response.status_code,
                    body=response.text[:500],
                )
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    raw={"status": response.status_code, "body": response.text},
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            raw: dict[str, Any] = dict(data)
            # Surface Track 3 ingest telemetry if the server returned it.
            if "seedDocsCount" in data:
                raw["seed_docs_count"] = int(data["seedDocsCount"])
            if "seedEmbedMs" in data:
                raw["seed_embed_ms"] = float(data["seedEmbedMs"])
            # Populate seed_docs_count from the request if server didn't echo it.
            if seed_docs is not None and "seed_docs_count" not in raw:
                raw["seed_docs_count"] = len(seed_docs)
            # Token counts from server response (when the eval-turn endpoint returns them).
            input_tok: int = int(data.get("inputTokens", data.get("input_tokens", 0)) or 0)
            output_tok: int = int(data.get("outputTokens", data.get("output_tokens", 0)) or 0)
            tokens_unavailable = input_tok == 0 and output_tok == 0
            if input_tok:
                raw["input_tokens"] = input_tok
            if output_tok:
                raw["output_tokens"] = output_tok
            return AdapterResult(
                answer=str(data.get("answer", "")),
                cost_usd=float(data.get("totalCostUsd", 0.0)),
                latency_ms=elapsed_ms,
                input_tokens=input_tok,
                output_tokens=output_tok,
                tokens_unavailable=tokens_unavailable,
                raw=raw,
            )
        except httpx.HTTPError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                raw={},
                error=f"HTTP error: {e}",
            )

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
