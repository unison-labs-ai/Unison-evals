"""Unison brain-context adapter — the new SDK-customer contract.

Flow per question:
  1. POST /v1/eval/provision   → {tenantId, userId}
  2. POST /v1/eval/seed        → bulk-write + synchronously embed haystack docs
  3. GET  /v1/brain/context?q= → {contextMd, hits, entities, weakEvidence, …}
  4. Reader LLM over contextMd  → answer
  5. POST /v1/eval/teardown    → cleanup

Auth:
  - Steps 1/2/5 use X-Unison-Eval (UNISON_EVAL_SECRET).
  - Step 3 uses Authorization: Bearer <JWT>. The JWT is:
      a. UNISON_JWT if set (works against prod or any server with that user's session).
      b. UNISON_EVAL_JWT if set (dedicated eval read key).
      c. Auto-minted via stdlib HS256 when SUPABASE_JWT_SECRET is set (local dev only).
         The token's `sub` is the provisioned userId, binding the read to the
         just-seeded tenant via the server's in-code tenant filter.

Name: "unison-brain-context"

This adapter implements the new brain-only contract that replaces the old
/api/rest/agents/eval-turn path.  The brain only retrieves — the answer is
generated here by the harness-owned reader LLM.  That keeps the eval honest:
retrieval quality and generation quality are separate, and the server never
generates answers for benchmarked questions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any

import httpx
from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from ..preingest import load_manifest, save_manifest
from ._url_utils import is_localhost_url
from .base import AgentAdapter

# ---------------------------------------------------------------------------
# Reader LLM prompt
# ---------------------------------------------------------------------------

READER_PROMPT = """\
You have access to a retrieved context block from a knowledge base. Answer the
question using ONLY the information in the context. Be concise and direct.
If the context does not contain enough information to answer, say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

Answer directly, no preamble:"""

# ---------------------------------------------------------------------------
# Minimal HS256 JWT minter (no external deps — stdlib only)
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _mint_hs256_jwt(user_id: str, tenant_id: str, secret: str, ttl: int = 300) -> str:
    """Mint a minimal Supabase-accepted HS256 JWT for local-dev eval use only."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "sub": user_id,
                "role": "authenticated",
                "aud": "authenticated",
                "iat": now,
                "exp": now + ttl,
                "app_metadata": {"tenant_id": tenant_id},
            }
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def _to_writable_seed_path(path: str, ns: str) -> str:
    """Mirror the path rewriter from unison_agent.py so seed paths are valid."""
    stem = path.strip("/")
    if stem.endswith(".md"):
        stem = stem[:-3]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem.replace("/", "--")).strip("-") or "doc"
    return f"/private/sources/eval/{ns}/{slug[:120]}.md"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class UnisonBrainContextAdapter(AgentAdapter):
    """Eval adapter for the new brain-only contract:
    provision → seed → GET /v1/brain/context → reader LLM → answer → teardown.

    Auth for the /v1/brain/context read:
      1. UNISON_EVAL_JWT (dedicated eval read JWT, most explicit)
      2. UNISON_JWT (legacy shared JWT already in most .env files)
      3. Auto-minted local JWT (requires SUPABASE_JWT_SECRET — local dev only)
    """

    name = "unison-brain-context"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._eval_client: httpx.AsyncClient | None = None
        self._manifest: dict[str, Any] | None = None
        self._manifest_path: str | None = None

    async def setup(self) -> None:
        is_localhost = is_localhost_url(self.settings.unison_api_url)
        has_eval_secret = bool(self.settings.unison_eval_secret)
        has_jwt = bool(self.settings.unison_jwt)
        has_eval_jwt = bool(self.settings.unison_eval_jwt)
        has_supabase_secret = bool(self.settings.supabase_jwt_secret)
        has_machine_key = bool(self.settings.unison_brain_machine_key)

        # Eval secret required for provision/seed/teardown.
        if not has_eval_secret and not is_localhost:
            raise RuntimeError(
                "UNISON_EVAL_SECRET is required for unison-brain-context. "
                "Set it to the server's eval secret so the adapter can provision/seed/teardown."
            )

        # Brain read auth: one of four paths must be available.
        # Priority: machine key > eval JWT > shared JWT > SUPABASE_JWT_SECRET (HS256 mint).
        if not has_machine_key and not has_eval_jwt and not has_jwt and not has_supabase_secret:
            raise RuntimeError(
                "unison-brain-context needs auth for GET /v1/brain/context. Set one of:\n"
                "  UNISON_BRAIN_MACHINE_KEY — usk_live_... machine key (local dev, ES256 Supabase)\n"
                "  UNISON_EVAL_JWT   — dedicated eval read JWT\n"
                "  UNISON_JWT        — existing shared JWT\n"
                "  SUPABASE_JWT_SECRET — auto-mint local JWTs (Supabase CLI < 1.200 only)"
            )

        manifest_path = os.environ.get("UNISON_PREINGEST_MANIFEST")
        if manifest_path:
            self._manifest_path = manifest_path
            self._manifest = load_manifest(manifest_path)
            logger.info(
                "brain-context preingest reuse-cache loaded",
                path=manifest_path,
                cached=len(self._manifest.get("questions", {})),
            )

        eval_headers: dict[str, str] = {"Content-Type": "application/json"}
        if has_eval_secret:
            eval_headers["X-Unison-Eval"] = self.settings.unison_eval_secret

        self._eval_client = httpx.AsyncClient(
            base_url=self.settings.unison_api_url,
            headers=eval_headers,
            timeout=self.settings.adapter_timeout,
        )
        try:
            r = await self._eval_client.get("/health")
            r.raise_for_status()
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
        assert self._eval_client is not None, "setup() must be called first"

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # oracle track: skip provision/seed, call context with no tenant override
        if oracle_context is not None:
            return await self._answer_oracle(question, oracle_context)

        cache_key = corpus_key or question_id

        # Preingest manifest reuse
        preingested_tenant: str | None = None
        preingested_user: str | None = None
        if self._manifest is not None and cache_key is not None:
            entry = self._manifest.get("questions", {}).get(cache_key)
            if isinstance(entry, dict):
                preingested_tenant = entry.get("tenantId")
                preingested_user = entry.get("userId")
            elif isinstance(entry, str):
                # Legacy: manifest stored only tenantId as string
                preingested_tenant = entry

        # Machine-key + shared-tenant fast path: UNISON_BRAIN_MACHINE_KEY is set
        # together with UNISON_EVAL_TENANT_ID / UNISON_EVAL_USER_ID. Skip per-question
        # provision and teardown; seed into the shared tenant under a question-namespaced
        # path so queries from one question don't bleed into another.
        if (
            self.settings.unison_brain_machine_key
            and self.settings.unison_eval_tenant_id
            and self.settings.unison_eval_user_id
            and not preingested_tenant
        ):
            preingested_tenant = self.settings.unison_eval_tenant_id
            preingested_user = self.settings.unison_eval_user_id

        start = time.perf_counter()

        if preingested_tenant and preingested_user:
            if not seed_docs:
                # Pre-seeded (manifest or repeated oracle call) — query only.
                return await self._query_context(
                    question=question,
                    tenant_id=preingested_tenant,
                    user_id=preingested_user,
                    start=start,
                    seed_docs_count=0,
                    embed_ms=0.0,
                )
            # Shared-tenant mode: seed into the shared tenant, then query.
            docs_to_seed = seed_docs or []
            ns = hashlib.sha256(question.encode()).hexdigest()[:10]
            seed_result = await self._seed(
                tenant_id=preingested_tenant,
                user_id=preingested_user,
                docs=[
                    {
                        "path": _to_writable_seed_path(doc.path, ns),
                        "body": doc.body,
                        "kind": "note",
                    }
                    for doc in docs_to_seed
                ],
            )
            if seed_result is None:
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    raw={},
                    error="seed failed (shared-tenant mode)",
                )
            return await self._query_context(
                question=question,
                tenant_id=preingested_tenant,
                user_id=preingested_user,
                start=start,
                seed_docs_count=seed_result.get("docsWritten", len(docs_to_seed)),
                embed_ms=float(seed_result.get("embedDurationMs", 0.0)),
            )

        # Slow path: provision → seed → query → (maybe) teardown
        provision = await self._provision_tenant()
        if provision is None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                raw={},
                error="failed to provision ephemeral eval tenant",
            )
        tenant_id, user_id = provision

        docs_to_seed = seed_docs or []
        ns = hashlib.sha256(question.encode()).hexdigest()[:10]
        seed_result = await self._seed(
            tenant_id=tenant_id,
            user_id=user_id,
            docs=[
                {
                    "path": _to_writable_seed_path(doc.path, ns),
                    "body": doc.body,
                    "kind": "note",
                }
                for doc in docs_to_seed
            ],
        )
        if seed_result is None:
            await self._teardown_tenant(tenant_id)
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                raw={},
                error="seed failed",
            )

        result = await self._query_context(
            question=question,
            tenant_id=tenant_id,
            user_id=user_id,
            start=start,
            seed_docs_count=seed_result.get("docsWritten", len(docs_to_seed)),
            embed_ms=float(seed_result.get("embedDurationMs", 0.0)),
        )

        # Save to manifest if pre-ingest mode
        if self._manifest is not None and self._manifest_path is not None and cache_key is not None:
            self._manifest.setdefault("questions", {})[cache_key] = {
                "tenantId": tenant_id,
                "userId": user_id,
            }
            save_manifest(self._manifest_path, self._manifest)
            # Keep the tenant — it's cached for future reuse.
        else:
            await self._teardown_tenant(tenant_id)

        return result

    async def _answer_oracle(self, question: str, oracle_context: str) -> AdapterResult:
        """Track 2: skip brain, run reader LLM over oracle_context directly."""
        start = time.perf_counter()
        answer = await self._reader_llm(question, oracle_context)
        latency = (time.perf_counter() - start) * 1000.0
        return AdapterResult(
            answer=answer,
            cost_usd=0.0,
            latency_ms=latency,
            raw={"mode": "oracle"},
            tokens_unavailable=True,
        )

    async def _query_context(
        self,
        question: str,
        tenant_id: str,
        user_id: str,
        start: float,
        seed_docs_count: int,
        embed_ms: float,
    ) -> AdapterResult:
        """GET /v1/brain/context?q=... and run the reader LLM over contextMd."""
        jwt = self._resolve_brain_jwt(user_id, tenant_id)
        brain_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt}",
        }
        brain_client = httpx.AsyncClient(
            base_url=self.settings.unison_api_url,
            headers=brain_headers,
            timeout=self.settings.adapter_timeout,
        )
        try:
            resp = await brain_client.get(
                "/v1/brain/context",
                params={"q": question},
            )
            context_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code != 200:
                logger.warning(
                    "brain/context non-200",
                    status=resp.status_code,
                    body=resp.text[:300],
                )
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=context_ms,
                    raw={"status": resp.status_code, "body": resp.text},
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()
            context_md: str = data.get("contextMd") or ""
            hits: list = data.get("hits") or []
            weak_evidence: bool = bool(data.get("weakEvidence", False))

            # Optional: fetch full document bodies for top-k hits and append to
            # contextMd. /v1/brain/context returns ~160-char snippets tuned for
            # interactive agents; single-shot reader LLMs need the full body.
            # Note: /system/facts/ paths ARE fetched — they contain extracted fact
            # nodes that frequently hold the direct answer (e.g. "FACT [date]: X").
            full_docs_fetched = 0
            if self.settings.context_fetch_full_docs and hits:
                k = self.settings.context_full_docs_k
                doc_parts: list[str] = ["\n\n---\n# Full document bodies (top hits)\n"]
                for hit in hits[:k]:
                    path = (hit.get("doc") or hit).get("path", "")
                    if not path:
                        continue
                    try:
                        doc_resp = await brain_client.get(
                            "/v1/brain/doc", params={"path": path}
                        )
                        if doc_resp.status_code == 200:
                            doc_data = doc_resp.json()
                            body = doc_data.get("bodyMd") or ""
                            if body:
                                doc_parts.append(f"\n## {path}\n\n{body}\n")
                                full_docs_fetched += 1
                    except httpx.HTTPError:
                        pass
                if full_docs_fetched > 0:
                    context_md = context_md + "".join(doc_parts)

            reader_start = time.perf_counter()
            answer = await self._reader_llm(question, context_md)
            reader_ms = (time.perf_counter() - reader_start) * 1000.0
            total_ms = (time.perf_counter() - start) * 1000.0

            return AdapterResult(
                answer=answer,
                cost_usd=0.0,
                latency_ms=total_ms,
                raw={
                    "mode": "brain-context",
                    "seed_docs_count": seed_docs_count,
                    "seed_embed_ms": embed_ms,
                    "context_fetch_ms": context_ms,
                    "reader_ms": reader_ms,
                    "hits": len(hits),
                    "full_docs_fetched": full_docs_fetched,
                    "weak_evidence": weak_evidence,
                },
                tokens_unavailable=True,
            )
        except httpx.HTTPError as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed,
                raw={},
                error=f"HTTP error on /v1/brain/context: {e}",
            )
        finally:
            await brain_client.aclose()

    def _resolve_brain_jwt(self, user_id: str, tenant_id: str) -> str:
        """Return the best available bearer token for GET /v1/brain/context reads.

        Priority:
          1. UNISON_BRAIN_MACHINE_KEY — usk_live_... machine key; bypasses Supabase
             JWT verification entirely. Required when the local Supabase stack uses
             ES256 JWKS (CLI >= ~1.200) which rejects HS256 auto-minted tokens.
          2. UNISON_EVAL_JWT — dedicated eval read JWT.
          3. UNISON_JWT — legacy shared JWT.
          4. SUPABASE_JWT_SECRET — auto-mint HS256 JWT (Supabase CLI < 1.200 only).
        """
        # 1. Machine API key (usk_) — no Supabase JWT verification involved.
        if self.settings.unison_brain_machine_key:
            return self.settings.unison_brain_machine_key
        # 2. Explicit eval JWT (most specific).
        if self.settings.unison_eval_jwt:
            return self.settings.unison_eval_jwt
        # 3. Legacy shared JWT from settings.
        if self.settings.unison_jwt:
            return self.settings.unison_jwt
        # 4. Auto-mint local JWT (local dev only, requires SUPABASE_JWT_SECRET).
        if self.settings.supabase_jwt_secret:
            return _mint_hs256_jwt(user_id, tenant_id, self.settings.supabase_jwt_secret)
        raise RuntimeError(
            "No bearer token available for /v1/brain/context. "
            "Set UNISON_BRAIN_MACHINE_KEY, UNISON_EVAL_JWT, UNISON_JWT, or SUPABASE_JWT_SECRET."
        )

    async def _reader_llm(self, question: str, context_md: str) -> str:
        """Run the reader LLM to produce an answer from contextMd."""
        prompt = READER_PROMPT.format(context=context_md, question=question)
        model = self.settings.context_reader_model
        if model.startswith("gpt") or model.startswith("o"):
            return await self._openai_reader(prompt, model)
        if model.startswith("claude"):
            return await self._anthropic_reader(prompt, model)
        if model.startswith("gemini"):
            return await self._google_reader(prompt, model)
        # Default to OpenAI.
        return await self._openai_reader(prompt, model)

    async def _openai_reader(self, prompt: str, model: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        new_family = model.lower().startswith(("o1", "o3", "o4")) or "gpt-5" in model.lower()
        token_kwarg = {"max_completion_tokens": 300} if new_family else {"max_tokens": 300}
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0 if new_family else 0.0,
            **token_kwarg,
        )
        return (resp.choices[0].message.content or "").strip()

    async def _anthropic_reader(self, prompt: str, model: str) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        resp = await client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    async def _google_reader(self, prompt: str, model: str) -> str:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=self.settings.google_api_key)
        resp = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(max_output_tokens=300, temperature=0),
        )
        return (resp.text or "").strip()

    async def _provision_tenant(self) -> tuple[str, str] | None:
        assert self._eval_client is not None
        try:
            resp = await self._eval_client.post(
                "/v1/eval/provision", json={"label": "longmemeval-context"}
            )
            if resp.status_code != 200:
                logger.warning(
                    "eval/provision failed", status=resp.status_code, body=resp.text[:300]
                )
                return None
            data = resp.json()
            tenant_id = str(data.get("tenantId") or "")
            user_id = str(data.get("userId") or "")
            if not tenant_id or not user_id:
                logger.warning("eval/provision returned no tenantId/userId", data=data)
                return None
            return tenant_id, user_id
        except httpx.HTTPError as e:
            logger.warning("eval/provision error", error=str(e))
            return None

    async def _seed(
        self,
        tenant_id: str,
        user_id: str,
        docs: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        assert self._eval_client is not None
        if not docs:
            return {"docsWritten": 0, "embedDurationMs": 0.0}
        try:
            resp = await self._eval_client.post(
                "/v1/eval/seed",
                json={"tenantId": tenant_id, "userId": user_id, "docs": docs},
            )
            if resp.status_code != 200:
                logger.warning("eval/seed failed", status=resp.status_code, body=resp.text[:300])
                return None
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning("eval/seed error", error=str(e))
            return None

    async def _teardown_tenant(self, tenant_id: str) -> None:
        assert self._eval_client is not None
        try:
            resp = await self._eval_client.post("/v1/eval/teardown", json={"tenantId": tenant_id})
            if resp.status_code != 200:
                logger.warning(
                    "eval/teardown failed", status=resp.status_code, body=resp.text[:300]
                )
        except httpx.HTTPError as e:
            logger.warning("eval/teardown error", tenant=tenant_id, error=str(e))

    async def teardown(self) -> None:
        if self._eval_client is not None:
            await self._eval_client.aclose()
            self._eval_client = None
