"""Unison agent adapter — POSTs to /api/rest/agents/eval-turn.

This is the production agent code path. Track 3 calls without oracleContext
(full agent + brain). Track 2 passes oracleContext (agent reasons from the
provided context only — brain/FS/workspace tools are stripped server-side).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from typing import Any

import httpx
from loguru import logger


def _to_writable_seed_path(path: str, ns: str) -> str:
    """Map an eval-corpus document path into a path the Unison brain accepts.

    The brain's write-path contract only allows writes under /private, /teams,
    /workspace, /wiki, /skills; ingest-style /private/sources/* is accepted by the
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


from ...config import get_settings  # noqa: E402
from ...types import AdapterResult, Document  # noqa: E402
from ..preingest import load_manifest, save_manifest, workspace_for  # noqa: E402
from ._url_utils import is_localhost_url  # noqa: E402
from .base import AgentAdapter  # noqa: E402


class UnisonAgentAdapter(AgentAdapter):
    name = "unison-agent"
    # "raw" = seed docs as-is (skips the brain's extract→promote→compact
    # pipeline). "pipeline" = the honest memory-pipeline test (server seeds as
    # notes, drives extraction in a fresh per-question sub-workspace). Subclassed
    # below as `unison-agent-pipeline` so a run can compare both side by side.
    ingest_mode: str = "raw"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: httpx.AsyncClient | None = None
        self._isolate_per_question: bool = False
        # When UNISON_PREINGEST_MANIFEST is set, the manifest is a reuse cache:
        # a question already in it reuses its persistent workspace (no re-seed, no
        # teardown); a question NOT in it is seeded once, then recorded + kept so
        # the NEXT run reuses it. First run seeds, later runs are query-only.
        self._manifest: dict[str, Any] | None = None
        self._manifest_path: str | None = None
        # Per-corpus_key locks: when many questions share one corpus (LOCOMO), the
        # FIRST to arrive seeds the conversation while the rest WAIT on this lock,
        # then reuse the freshly-cached workspace instead of each re-seeding it.
        self._seed_locks: dict[str, asyncio.Lock] = {}

    def _seed_lock(self, key: str) -> asyncio.Lock:
        lock = self._seed_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._seed_locks[key] = lock
        return lock

    async def setup(self) -> None:
        is_localhost = is_localhost_url(self.settings.unison_api_url)
        has_jwt = bool(self.settings.unison_jwt)
        has_secret = bool(self.settings.unison_eval_secret)

        if not has_jwt and not has_secret and not is_localhost:
            raise RuntimeError(
                "Neither UNISON_EVAL_SECRET nor UNISON_JWT set. One is required when\n"
                "UNISON_API_URL is not localhost. Preferred: set UNISON_EVAL_SECRET to the\n"
                "server's eval secret — the adapter then provisions a fresh ephemeral workspace\n"
                "per question with no JWT needed. For local dev: point UNISON_API_URL at\n"
                "http://localhost:3001 and set UNISON_LOCAL_EVAL_WORKSPACE_ID on the server."
            )

        # When the eval secret is configured, every question runs in a
        # freshly-provisioned ephemeral workspace that is torn down afterward — true
        # per-question isolation, no cross-question retrieval contamination.
        self._isolate_per_question = has_secret

        manifest_path = os.environ.get("UNISON_PREINGEST_MANIFEST")
        if manifest_path:
            self._manifest_path = manifest_path
            self._manifest = load_manifest(manifest_path)
            logger.info(
                "preingest reuse-cache loaded",
                path=manifest_path,
                cached=len(self._manifest.get("questions", {})),
            )

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
        question_id: str | None = None,
        corpus_key: str | None = None,
    ) -> AdapterResult:
        assert self._client is not None, "setup() must be called first"

        # Cache/seed key: the CORPUS, not the question. Many questions can share a
        # corpus (LOCOMO) — they seed it once and reuse. Defaults to question_id.
        cache_key = corpus_key or question_id

        # Pre-ingested reuse: when this corpus was already ingested into a
        # persistent workspace, query it read-only (no re-seed, no teardown).
        preingested_workspace: str | None = None
        if self._manifest is not None and cache_key is not None:
            preingested_workspace = workspace_for(self._manifest, cache_key)

        # If not yet seeded, serialize on the corpus so concurrent same-corpus
        # questions don't each seed it: the first holds the lock through its seed,
        # the rest wait then fall through to reuse the freshly-cached workspace.
        held_lock: asyncio.Lock | None = None
        if (
            preingested_workspace is None
            and self._manifest is not None
            and cache_key is not None
            and seed_docs is not None
        ):
            held_lock = self._seed_lock(cache_key)
            await held_lock.acquire()
            preingested_workspace = workspace_for(self._manifest, cache_key)  # re-check under lock
            if preingested_workspace is not None:
                held_lock.release()
                held_lock = None

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
        if seed_docs is not None and preingested_workspace is None:
            # Per-question namespace so one question's seeded docs don't collide
            # with another's (belt-and-suspenders; the ephemeral workspace already
            # isolates when self._isolate_per_question is on).
            ns = hashlib.sha256(question.encode()).hexdigest()[:10]
            # kind="raw" skips Unison's extract pipeline (Gemini calls per doc),
            # which is critical for cost in per-question-haystack benchmarks.
            body["seedDocs"] = [
                {"path": _to_writable_seed_path(doc.path, ns), "body": doc.body, "kind": "raw"}
                for doc in seed_docs
            ]
            # "pipeline" tells the server to seed as notes, run a fresh
            # per-question sub-workspace through extract→promote→compact, then answer.
            body["ingestMode"] = self.ingest_mode

        # Workspace selection:
        #  - pre-ingested → reuse the persistent workspace read-only (no teardown).
        #  - else isolate_per_question → provision a throwaway workspace + teardown.
        #  - else → localhost-bypass path (caller's own workspace).
        provisioned_workspace: str | None = None
        if preingested_workspace is not None:
            body["workspaceId"] = preingested_workspace
            body["memoryMode"] = "fresh"
        elif self._isolate_per_question:
            provisioned_workspace = await self._provision_workspace()
            if provisioned_workspace is None:
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=0.0,
                    raw={},
                    error="failed to provision ephemeral eval workspace",
                )
            body["workspaceId"] = provisioned_workspace
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
            # Self-cache: a freshly-seeded workspace (manifest mode, not already
            # cached) is recorded + kept so the NEXT run reuses it read-only
            # instead of re-seeding from scratch.
            if (
                self._manifest is not None
                and self._manifest_path is not None
                and provisioned_workspace is not None
                and cache_key is not None
            ):
                self._manifest.setdefault("questions", {})[cache_key] = provisioned_workspace
                save_manifest(self._manifest_path, self._manifest)

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
            # Release the per-corpus seed lock so waiting same-corpus questions
            # can now reuse the workspace this call just cached.
            if held_lock is not None:
                held_lock.release()
            # Tear down only in pure-ephemeral mode. In manifest mode we KEEP
            # every provisioned workspace — it's just been cached for reuse by the
            # next run (and its brain + trajectory persist for analysis).
            if provisioned_workspace is not None and self._manifest is None:
                await self._teardown_workspace(provisioned_workspace)

    async def retrieve(self, question: str, corpus_key: str, k: int = 25) -> list[str]:
        """Track-1 recall@k diagnostic: run ONE semantic search against the
        (pre-seeded) workspace for this corpus and return the ranked retrieved doc
        paths — no agent turn. Requires the corpus to already be in the manifest
        (run `preingest`, or a prior `run`, first); returns [] otherwise.
        """
        assert self._client is not None, "setup() must be called first"
        workspace = workspace_for(self._manifest, corpus_key) if self._manifest is not None else None
        body: dict[str, Any] = {"question": question, "retrieveOnly": True, "retrieveK": k}
        if workspace is not None:
            body["workspaceId"] = workspace
            body["memoryMode"] = "fresh"
        try:
            resp = await self._client.post("/api/rest/agents/eval-turn", json=body)
            if resp.status_code != 200:
                logger.warning("retrieve non-200", status=resp.status_code, body=resp.text[:200])
                return []
            return list(resp.json().get("retrievedPaths") or [])
        except httpx.HTTPError as e:
            logger.warning("retrieve error", error=str(e))
            return []

    async def preingest_question(
        self, question: str, seed_docs: list[Document], question_id: str
    ) -> str | None:
        """Seed one question's haystack into a NEW persistent workspace and build the
        full memory graph. Seeds as kind="note" so the server enqueues the real
        extract jobs; with AGENT_WAIT_GRAPH=1 it drives extract → signals →
        promote → cortex_facts before returning. Returns the workspace_id and does
        NOT tear it down — `run` reuses it read-only via the manifest. Used by
        `unison-evals preingest`."""
        assert self._client is not None, "setup() must be called first"
        workspace_id = await self._provision_workspace()
        if workspace_id is None:
            return None
        ns = hashlib.sha256(question.encode()).hexdigest()[:10]
        body: dict[str, Any] = {
            "question": question,
            "workspaceId": workspace_id,
            "memoryMode": "fresh",
            # Build the brain only — skip the (throwaway) agent answer. ~30%
            # cheaper + faster during bulk pre-ingestion.
            "seedOnly": True,
            # kind="note" (not "raw") so the seed enqueues extraction — the
            # whole point of pre-ingest is to build the fact/observation graph.
            "seedDocs": [
                {"path": _to_writable_seed_path(doc.path, ns), "body": doc.body, "kind": "note"}
                for doc in seed_docs
            ],
        }
        try:
            resp = await self._client.post("/api/rest/agents/eval-turn", json=body)
            if resp.status_code != 200:
                logger.warning(
                    "preingest seed failed",
                    question_id=question_id,
                    status=resp.status_code,
                    body=resp.text[:300],
                )
                await self._teardown_workspace(workspace_id)
                return None
            return workspace_id
        except httpx.HTTPError as e:
            logger.warning("preingest seed error", question_id=question_id, error=str(e))
            await self._teardown_workspace(workspace_id)
            return None

    async def _provision_workspace(self) -> str | None:
        """Provision a fresh ephemeral eval workspace. Returns its workspaceId, or None on failure."""
        assert self._client is not None
        try:
            resp = await self._client.post(
                "/api/rest/agents/eval/provision", json={"label": "longmemeval"}
            )
            if resp.status_code != 200:
                logger.warning(
                    "eval/provision failed", status=resp.status_code, body=resp.text[:300]
                )
                return None
            return str(resp.json().get("workspaceId")) or None
        except httpx.HTTPError as e:
            logger.warning("eval/provision error", error=str(e))
            return None

    async def _teardown_workspace(self, workspace_id: str) -> None:
        """Hard-delete an ephemeral eval workspace. Best-effort; logs on failure so a
        stuck teardown never fails the question (a reaper sweeps stragglers)."""
        assert self._client is not None
        try:
            resp = await self._client.post(
                "/api/rest/agents/eval/teardown", json={"workspaceId": workspace_id}
            )
            if resp.status_code != 200:
                logger.warning(
                    "eval/teardown failed", status=resp.status_code, body=resp.text[:300]
                )
        except httpx.HTTPError as e:
            logger.warning("eval/teardown error", workspace=workspace_id, error=str(e))

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class UnisonAgentPipelineAdapter(UnisonAgentAdapter):
    """Same agent, but exercises the real memory pipeline (extract → promote →
    compact) in a fresh per-question sub-workspace instead of dumping raw docs.
    Register as `unison-agent-pipeline` to compare against `unison-agent`."""

    name = "unison-agent-pipeline"
    ingest_mode = "pipeline"
