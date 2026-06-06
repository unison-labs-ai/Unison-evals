"""Unison agent adapter — POSTs to /api/rest/agents/eval-turn.

This is the production agent code path. Track 3 calls without oracleContext
(full agent + brain). Track 2 passes oracleContext (agent reasons from the
provided context only — brain/FS/workspace tools are stripped server-side).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import httpx
from loguru import logger


def _to_writable_seed_path(path: str, ns: str) -> str:
    """Map an eval-corpus document path into a path the Unison brain accepts.

    The brain's write-path contract only allows writes under /private, /teams,
    /tenant, /wiki, /skills; ingest-style /private/sources/* is accepted by the
    eval-turn seed path (brain.write bypasses the user-facing gate). Eval datasets
    use roots like /sessions/<id>.md, which the brain rejects. We flatten the
    original path into a single slugged filename under a per-question namespace so
    (a) writes succeed and (b) one question's docs don't overwrite another's.
    """
    stem = path.strip("/")
    if stem.endswith(".md"):
        stem = stem[:-3]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem.replace("/", "--")).strip("-") or "doc"
    return f"/private/sources/eval/{ns}/{slug[:120]}.md"

from ...config import get_settings
from ...types import AdapterResult, Document
from ._url_utils import is_localhost_url
from .base import AgentAdapter


class UnisonAgentAdapter(AgentAdapter):
    name = "unison-agent"
    # "raw" = seed docs as-is (skips the brain's extract→promote→compact
    # pipeline). "pipeline" = the honest memory-pipeline test (server seeds as
    # notes, drives extraction in a fresh per-question sub-tenant). Subclassed
    # below as `unison-agent-pipeline` so a run can compare both side by side.
    ingest_mode: str = "raw"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: httpx.AsyncClient | None = None
        self._isolate_per_question: bool = False

    async def setup(self) -> None:
        is_localhost = is_localhost_url(self.settings.unison_api_url)
        has_jwt = bool(self.settings.unison_jwt)
        has_secret = bool(self.settings.unison_eval_secret)

        if not has_jwt and not has_secret and not is_localhost:
            raise RuntimeError(
                "Neither UNISON_EVAL_SECRET nor UNISON_JWT set. One is required when\n"
                "UNISON_API_URL is not localhost. Preferred: set UNISON_EVAL_SECRET to the\n"
                "server's value — the adapter then provisions a fresh ephemeral tenant per\n"
                "question (ADR-0008), no Supabase JWT needed. For local dev: point at\n"
                "http://localhost:3001 with UNISON_LOCAL_EVAL_TENANT_ID on the server."
            )

        # ADR-0008: when the eval secret is configured, every question runs in a
        # freshly-provisioned ephemeral tenant that is torn down afterward — true
        # per-question isolation, no cross-question retrieval contamination.
        self._isolate_per_question = has_secret

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if has_jwt:
            headers["Authorization"] = f"Bearer {self.settings.unison_jwt}"
        if has_secret:
            headers["X-Unison-Eval"] = self.settings.unison_eval_secret

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

        # Submit the task WITHOUT a model so the server runs its production model
        # path (auto + escalation) exactly like a live user turn. Only pin a
        # model for an explicit ablation (unison_agent_model set). The eval must
        # not choose the model — that's the server's job.
        body: dict[str, Any] = {"question": question}
        if self.settings.unison_agent_model:
            body["model"] = self.settings.unison_agent_model
        if oracle_context is not None:
            body["oracleContext"] = oracle_context
        if seed_docs is not None:
            # Per-question namespace so one question's seeded docs don't collide
            # with another's (belt-and-suspenders; the ephemeral tenant already
            # isolates when self._isolate_per_question is on).
            ns = hashlib.sha256(question.encode()).hexdigest()[:10]
            # kind="raw" skips Unison's extract pipeline (Gemini calls per doc),
            # which is critical for cost in per-question-haystack benchmarks.
            body["seedDocs"] = [
                {"path": _to_writable_seed_path(doc.path, ns), "body": doc.body, "kind": "raw"}
                for doc in seed_docs
            ]
            # "pipeline" tells the server to seed as notes, run a fresh
            # per-question sub-tenant through extract→promote→compact, then answer.
            body["ingestMode"] = self.ingest_mode

        # ADR-0008: provision a throwaway tenant for this single question, run the
        # turn against it with memoryMode="fresh", and tear it down afterward —
        # zero cross-question contamination. Skipped on the localhost-bypass path.
        tenant_id: str | None = None
        if self._isolate_per_question:
            tenant_id = await self._provision_tenant()
            if tenant_id is None:
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=0.0,
                    raw={},
                    error="failed to provision ephemeral eval tenant",
                )
            body["tenantId"] = tenant_id
            body["memoryMode"] = "fresh"

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
        finally:
            if tenant_id is not None:
                await self._teardown_tenant(tenant_id)

    async def _provision_tenant(self) -> str | None:
        """Provision a fresh ephemeral eval tenant (ADR-0008). Returns its
        tenantId, or None on failure."""
        assert self._client is not None
        try:
            resp = await self._client.post(
                "/api/rest/agents/eval/provision", json={"label": "longmemeval"}
            )
            if resp.status_code != 200:
                logger.warning("eval/provision failed", status=resp.status_code, body=resp.text[:300])
                return None
            return str(resp.json().get("tenantId")) or None
        except httpx.HTTPError as e:
            logger.warning("eval/provision error", error=str(e))
            return None

    async def _teardown_tenant(self, tenant_id: str) -> None:
        """Hard-delete an ephemeral eval tenant. Best-effort; logs on failure so a
        stuck teardown never fails the question (a reaper sweeps stragglers)."""
        assert self._client is not None
        try:
            resp = await self._client.post(
                "/api/rest/agents/eval/teardown", json={"tenantId": tenant_id}
            )
            if resp.status_code != 200:
                logger.warning("eval/teardown failed", status=resp.status_code, body=resp.text[:300])
        except httpx.HTTPError as e:
            logger.warning("eval/teardown error", tenant=tenant_id, error=str(e))

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class UnisonAgentPipelineAdapter(UnisonAgentAdapter):
    """Same agent, but exercises the real memory pipeline (extract → promote →
    compact) in a fresh per-question sub-tenant instead of dumping raw docs.
    Register as `unison-agent-pipeline` to compare against `unison-agent`."""

    name = "unison-agent-pipeline"
    ingest_mode = "pipeline"
