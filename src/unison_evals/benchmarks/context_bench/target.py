"""Thin HTTP client for Unison's `/api/rest/agents/eval-turn`.

Context-Bench uses a SINGLE fixed corpus (11 files) shared by all 100
questions, so isolation is **per-run** (one ephemeral workspace per run), not per-question:

  setup()  → provision one ephemeral `is_eval` workspace, seed the corpus
             into /private/sources/eval/context-bench/ once.
  ask(q)   → run one question against that workspace with memoryMode="fresh"
             (no extraction residue between questions; corpus stays put).
  close()  → hard-delete the ephemeral workspace.

No dedicated workspace, no cross-run residue — the secret
(`UNISON_EVAL_SECRET`, sent as X-Unison-Eval) is the only auth. On a
localhost server with UNISON_EVAL_LOCAL_BYPASS the secret is optional.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from . import seed

# Throwaway model for the one-time seed turn (the agent answer is discarded;
# we only need the server-side seedBrainSync to run).
_SEED_MODEL = "claude-haiku-4-5"


@dataclass
class TargetAnswer:
    answer: str
    session_id: str
    total_steps: int
    total_cost_usd: float
    elapsed_s: float
    seeded_pages: int


class UnisonContextBenchTarget:
    """Per-run-isolated Q&A target: provision+seed once, ask N times, teardown."""

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.api_url = (
            api_url or os.environ.get("UNISON_API_URL") or "http://localhost:3001"
        ).rstrip("/")
        self.model = model
        self.eval_secret = os.environ.get("UNISON_EVAL_SECRET", "")
        headers = {"Content-Type": "application/json"}
        if self.eval_secret:
            headers["X-Unison-Eval"] = self.eval_secret
        self._client = httpx.Client(base_url=self.api_url, headers=headers, timeout=timeout)
        self.workspace_id: str | None = None
        self.seeded_pages = 0

    def setup(self) -> None:
        """Provision a fresh ephemeral workspace and seed the fixed corpus once."""
        prov = self._client.post("/api/rest/agents/eval/provision", json={"label": "context-bench"})
        prov.raise_for_status()
        self.workspace_id = str(prov.json()["workspaceId"])

        docs = seed.corpus_seed_docs()
        # One seed-bearing turn writes + embeds the corpus server-side. The agent
        # answer ("READY") is discarded; the docs persist in the workspace for every
        # subsequent ask(). Cheap throwaway model keeps the seed turn near-free.
        seed_resp = self._client.post(
            "/api/rest/agents/eval-turn",
            json={
                "workspaceId": self.workspace_id,
                "question": "Reply with the single word READY.",
                "model": _SEED_MODEL,
                "memoryMode": "fresh",
                "seedDocs": docs,
            },
        )
        seed_resp.raise_for_status()
        self.seeded_pages = int(seed_resp.json().get("seedDocsCount") or len(docs))

    def ask(self, question: str) -> TargetAnswer:
        if self.workspace_id is None:
            raise RuntimeError("setup() must be called before ask()")

        framing = (
            "You are an analyst with read-only access to ten data files under /private/sources/eval/context-bench/. "
            "Start by reading /private/sources/eval/context-bench/SCHEMA.md to learn the layout, then use bash "
            "(cat, grep, awk, sort, etc.) to answer the question below. Cross-reference "
            "files via shared id fields (e.g. person_id) as needed. When you have the "
            "answer, reply with it directly — no preamble.\n\n"
            f"QUESTION:\n{question}"
        )

        t0 = time.monotonic()
        body: dict = {
            "workspaceId": self.workspace_id,
            "question": framing,
            # Each row must be iid — skip Memory-v2 extract.turn.
            "memoryMode": "fresh",
        }
        # No model = the server runs its production path (auto + escalation),
        # like a live user turn. Pin one only for an explicit ablation.
        if self.model:
            body["model"] = self.model
        resp = self._client.post("/api/rest/agents/eval-turn", json=body)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0

        return TargetAnswer(
            answer=data.get("answer") or "",
            session_id=data.get("sessionId") or "",
            total_steps=int(data.get("totalSteps") or 0),
            total_cost_usd=float(data.get("totalCostUsd") or 0.0),
            elapsed_s=elapsed,
            seeded_pages=self.seeded_pages,
        )

    def close(self) -> None:
        """Hard-delete the ephemeral workspace, then close the client. Best-effort
        teardown so a failed delete never masks the run's results."""
        try:
            if self.workspace_id is not None:
                self._client.post(
                    "/api/rest/agents/eval/teardown", json={"workspaceId": self.workspace_id}
                )
        except httpx.HTTPError:
            pass
        finally:
            self._client.close()
            self.workspace_id = None
