"""Thin HTTP client for Unison's `/api/rest/agents/eval-turn`.

One question → one answer. Wipes + reseeds the tenant brain per
question so every row is iid (the published Context-Bench cell does
the same via Letta's per-task setup_script).

The default tenant + user IDs match the Unison local-dev eval tenant
created earlier in this project. Override with constructor args for a
hosted Unison.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from . import seed

DEFAULT_TENANT_ID = "981825b2-33a0-4a0c-8e9b-1d2d671c014f"
DEFAULT_USER_ID = "4eb0ea0e-7d49-492e-a814-a2434be6f5fb"


@dataclass
class TargetAnswer:
    answer: str
    session_id: str
    total_steps: int
    total_cost_usd: float
    elapsed_s: float
    wiped_docs: int
    seeded_pages: int


class UnisonContextBenchTarget:
    """One-shot Q&A target: seed → ask → return assistant's final text."""

    def __init__(
        self,
        api_url: str | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
        user_id: str = DEFAULT_USER_ID,
        model: str = "claude-sonnet-4-5",
        timeout: float = 600.0,
    ) -> None:
        self.api_url = (
            api_url or os.environ.get("UNISON_API_URL") or "http://localhost:3001"
        ).rstrip("/")
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.model = model
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    def ask(self, question: str) -> TargetAnswer:
        wiped, seeded = seed.fresh_tenant(self.tenant_id, self.user_id)

        framing = (
            "You are an analyst with read-only access to ten data files under /wiki/. "
            "Start by reading /wiki/SCHEMA.md to learn the layout, then use bash "
            "(cat, grep, awk, sort, etc.) to answer the question below. Cross-reference "
            "files via shared id fields (e.g. person_id) as needed. When you have the "
            "answer, reply with it directly — no preamble.\n\n"
            f"QUESTION:\n{question}"
        )

        t0 = time.monotonic()
        resp = self._client.post(
            "/api/rest/agents/eval-turn",
            json={
                "question": framing,
                "model": self.model,
                # Skip Memory-v2 extract.turn — each row must be iid.
                "memoryMode": "fresh",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0

        return TargetAnswer(
            answer=data.get("answer") or "",
            session_id=data.get("sessionId") or "",
            total_steps=int(data.get("totalSteps") or 0),
            total_cost_usd=float(data.get("totalCostUsd") or 0.0),
            elapsed_s=elapsed,
            wiped_docs=wiped,
            seeded_pages=seeded,
        )

    def close(self) -> None:
        self._client.close()
