"""Shared types used across adapters, datasets, runners, and metrics."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Track(StrEnum):
    """Eval tracks."""

    AGENT_ORACLE = "agent-oracle"  # agent given perfect context (no retrieval)
    AGENT_E2E = "agent-e2e"  # full pipeline


class Question(BaseModel):
    """One eval question, dataset-agnostic."""

    id: str
    question: str
    expected_answer: str
    # For Track 2 (oracle): the gold context the agent should use to answer.
    # None for Track 3 — the agent must retrieve.
    oracle_context: str | None = None
    # Dataset-specific extras (haystack_session_ids, question_type, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterResult(BaseModel):
    """One adapter's response to one question."""

    answer: str
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    # First-class token / cache / tool metrics.
    # All default to 0 so existing run JSONs without these fields still load.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    tool_calls: int = 0
    memory_ops: int = 0  # Mem0/Letta-specific add+search count
    context_pct_used: float = 0.0  # 0.0-1.0 fraction of context window consumed
    # Set True by adapters that genuinely cannot return usage stats (e.g.
    # unison-agent until Unison surfaces them in /eval-turn). Distinct from
    # 0 = "ran but used zero tokens" which is impossible.
    tokens_unavailable: bool = False
    # Adapter-specific raw response (for debugging / replay)
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JudgeResult(BaseModel):
    """Judge's verdict on an answer.

    verdict is the 4-way classification:
      CORRECT           — answer captures the expected information
      WRONG             — answer asserts something contradictory or fabricated
      CORRECT_ABSTAIN   — agent correctly said "I don't know" (expected was not derivable)
      INCORRECT_ABSTAIN — agent said "I don't know" but the answer was derivable

    passed is kept for back-compat: True when verdict is CORRECT or CORRECT_ABSTAIN.
    """

    score: float  # 0.0, 0.5, or 1.0  (legacy; still computed for back-compat)
    passed: bool  # score >= threshold OR verdict in {CORRECT, CORRECT_ABSTAIN}
    verdict: str = "CORRECT"  # Literal["CORRECT","WRONG","CORRECT_ABSTAIN","INCORRECT_ABSTAIN"]
    confidence: float  # 0.0 - 1.0
    reasoning: str
    cost_usd: float = 0.0


class QuestionResult(BaseModel):
    """One (system, question) result row."""

    question_id: str
    system: str
    adapter: AdapterResult
    judge: JudgeResult | None = None  # None if judge errored


class Document(BaseModel):
    """A document to ingest into a brain.

    `path` is the brain's filesystem-style identifier (e.g. /sessions/2026-01-05.md).
    `body` is the markdown/text body. `metadata` may include date, source,
    tags, or any per-dataset extras the adapter wants to preserve.
    """

    path: str
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """One result returned by a brain search.

    `doc_path` is the canonical id of the source document — the same string
    a Track 1 metric uses to compare against gold doc paths.
    `chunk_text` is the actual text the adapter returned (full doc or
    chunk slice). `score` is the adapter's ranking score (higher = more
    relevant). `rank` is the 1-indexed position in the result list.
    `raw` carries adapter-specific debug data without polluting metrics.
    """

    doc_path: str
    chunk_text: str
    score: float = 0.0
    rank: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class BrainSearchResult(BaseModel):
    """One brain search call's full result, including timing/cost."""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SystemSummary(BaseModel):
    """Per-system aggregates for a run."""

    system: str
    n_questions: int
    n_passed: int
    # Abstention breakdown (all default 0 so old JSONs still load).
    n_correct: int = 0
    n_wrong: int = 0
    n_correct_abstain: int = 0
    n_incorrect_abstain: int = 0
    pass_rate: float
    # Bootstrap 95% CI for pass_rate over the per-question outcomes.
    # None for runs that pre-date the stats integration.
    pass_rate_ci_low: float | None = None
    pass_rate_ci_high: float | None = None
    # pass^k metrics (populated when repeat > 1).
    pass_at_k: float | None = None  # None when k=1 (trivially equal to pass_rate)
    repeat: int = 1  # how many runs per question were used
    # Derived quality metrics.
    hallucination_rate: float = 0.0  # n_wrong / n_questions
    abstention_precision: float | None = (
        None  # n_correct_abstain / (n_correct_abstain + n_incorrect_abstain)
    )
    # Cost metrics.
    total_cost_usd: float
    cost_per_question_usd: float
    cost_per_solved_usd: float | None  # None if 0 solved
    # Latency.
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    # Token efficiency (all default 0 so old JSONs load cleanly).
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_cache_read_tokens: float = 0.0
    mean_cache_creation_tokens: float = 0.0
    cache_hit_rate: float = 0.0  # cache_read / (cache_read + cache_creation + input)
    mean_tool_calls: float = 0.0
    mean_memory_ops: float = 0.0
    mean_input_tokens_per_q: float = (
        0.0  # alias for mean_input_tokens; kept for cross-runner compat
    )
    efficiency_ratio: float | None = None
    tokens_unavailable: bool = False


class RunSummary(BaseModel):
    """Top-level summary for one eval run."""

    run_id: str
    dataset: str
    track: Track
    systems: list[str]
    judge_model: str
    n_questions: int
    # Number of times each (system, question) pair was run (for pass^k).
    repeat: int = 1
    started_at: datetime
    finished_at: datetime | None = None
    total_cost_usd: float = 0.0
    summaries: list[SystemSummary] = Field(default_factory=list)
    efficiency_narrative: str | None = None


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Track 3 (agent-e2e) per-question corpus types
# ---------------------------------------------------------------------------


class BrainQuestion(BaseModel):
    """One Track 3 eval question — carries its own corpus and gold doc paths.

    `corpus` is the set of documents the agent ingests before answering `query`.
    `gold_doc_paths` is the set of document paths relevant to `query`
    (used for optional retrieval scoring; the LLM judge scores the final answer).
    """

    id: str
    query: str
    corpus: list[Document]
    gold_doc_paths: set[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Stable key identifying this question's CORPUS. When multiple questions share
    # the same corpus (e.g. LOCOMO: ~200 qa over one conversation), they share a
    # corpus_key so the brain is seeded ONCE per corpus and reused — not re-seeded
    # per question. Defaults to `id` (every question its own corpus, e.g.
    # LongMemEval). Read it via `effective_corpus_key`.
    corpus_key: str | None = None

    @property
    def effective_corpus_key(self) -> str:
        return self.corpus_key or self.id
