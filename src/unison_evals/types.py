"""Shared types used across adapters, datasets, runners, and metrics."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Track(StrEnum):
    """Eval tracks."""

    BRAIN_ONLY = "brain-only"  # retrieval ranking quality (no LLM)
    AGENT_ORACLE = "agent-oracle"  # agent given perfect context (no retrieval)
    AGENT_E2E = "agent-e2e"  # full pipeline
    SCALE = "scale"  # query a pre-loaded large corpus, no per-Q ingest


class BrainMode(StrEnum):
    """Sub-mode selector for Track 1 (brain-only) runs.

    COLD        — per-question reset → ingest → search. Original Track 1 behavior.
    WARM        — corpus pre-loaded; runner skips reset/ingest, iterates search only.
                  Equivalent to what Track 4 (scale) does for smaller corpora.
    BITEMPORAL  — per-question reset → ingest → search at as_of timestamp;
                  scored with temporal_correct_at_1 in addition to standard metrics.
                  Only questions with metadata["as_of"] are probed for temporal
                  correctness; the rest fall back to standard hit@1 scoring.
    COMPACTION  — ingest raw transcripts, poll for compacted wiki page, LLM-judge
                  completeness + accuracy + compression. Only applicable to
                  unison-brain (the only adapter with a compactor). Other adapters
                  return compaction_quality_score=N/A and are skipped with [SKIP].
    """

    COLD = "cold"
    WARM = "warm"
    BITEMPORAL = "bitemporal"
    COMPACTION = "compaction"


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
    # Brain-efficiency (Track 3). efficiency_ratio = baseline_tokens / system_tokens
    # (where baseline = anthropic-raw, the most token-hungry honest approach).
    # >1 = more efficient than baseline. None when tokens unavailable or this
    # IS the baseline.
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
    # brain-efficiency narrative string (Track 3 only) — e.g.
    # "unison-agent: 83% pass-rate at 1 420 mean input tokens (8.2x more efficient than anthropic-raw)"
    efficiency_narrative: str | None = None


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Track 1 (brain-only) types
# ---------------------------------------------------------------------------


class BrainQuestion(BaseModel):
    """One Track 1 eval question — carries its own corpus and gold doc paths.

    `corpus` is the set of documents to ingest before running `search`.
    `gold_doc_paths` is the set of document paths that are relevant to `query`
    and are used to compute recall/nDCG/MRR.
    """

    id: str
    query: str
    corpus: list[Document]
    gold_doc_paths: set[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrainQuestionResult(BaseModel):
    """One (system, question) result row for Track 1."""

    question_id: str
    system: str
    search_result: BrainSearchResult
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class BrainSystemSummary(BaseModel):
    """Per-system aggregates for a Track 1 run."""

    system: str
    n_questions: int
    mean_recall_at_10: float
    mean_ndcg_at_10: float
    mean_mrr: float
    mean_hit_at_1: float
    # Bootstrap 95% CIs for the two headline metrics. None for legacy runs.
    recall_at_10_ci_low: float | None = None
    recall_at_10_ci_high: float | None = None
    hit_at_1_ci_low: float | None = None
    hit_at_1_ci_high: float | None = None
    total_cost_usd: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    # Only populated in BITEMPORAL mode; None otherwise.
    mean_temporal_correct_at_1: float | None = None
    # Only populated in COMPACTION mode; None otherwise.
    mean_compaction_quality: float | None = None


class BrainRunSummary(BaseModel):
    """Top-level summary for one Track 1 eval run."""

    run_id: str
    dataset: str
    track: Track
    mode: BrainMode = BrainMode.COLD
    systems: list[str]
    n_questions: int
    started_at: datetime
    finished_at: datetime | None = None
    total_cost_usd: float = 0.0
    summaries: list[BrainSystemSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Track 4 (scale) types
# ---------------------------------------------------------------------------


class ScaleQuestion(BaseModel):
    """One Track 4 (scale) eval question.

    Assumes the corpus is already loaded into the adapter's store.
    Carries only the query text and the gold doc paths used to score
    Recall@k / nDCG@10 / MRR.
    """

    id: str
    query: str
    gold_doc_paths: set[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScaleQuestionResult(BaseModel):
    """One (system, question) result row for Track 4."""

    question_id: str
    system: str
    search_result: BrainSearchResult
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class ScaleSystemSummary(BaseModel):
    """Per-system aggregates for a Track 4 run."""

    system: str
    n_questions: int
    mean_recall_at_10: float
    mean_ndcg_at_10: float
    mean_mrr: float
    mean_hit_at_1: float
    total_cost_usd: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class ScaleRunSummary(BaseModel):
    """Top-level summary for one Track 4 (scale) eval run."""

    run_id: str
    dataset: str
    track: Track  # always Track.SCALE
    systems: list[str]
    n_questions: int
    corpus_label: str  # human-friendly: "msmarco-passages-v1" etc.
    started_at: datetime
    finished_at: datetime | None = None
    total_cost_usd: float = 0.0
    summaries: list[ScaleSystemSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# --track all (multi-track) types
# ---------------------------------------------------------------------------


class MultiTrackRunSummary(BaseModel):
    """Combined summary produced by --track all.

    Runs Track 1 (brain-only), Track 2 (agent-oracle), and Track 3 (agent-e2e)
    for the same dataset x systems combo back-to-back. Each sub-summary is keyed
    by track name. Tracks that could not run (e.g. dataset raises NotImplementedError
    for load_brain_questions, or no brain adapters in the systems list) are absent.
    """

    run_id: str
    dataset: str
    systems: list[str]
    started_at: datetime
    finished_at: datetime | None = None
    tracks: dict[str, BrainRunSummary | RunSummary] = Field(default_factory=dict)
    skips: list[str] = Field(default_factory=list)  # human-readable skip reasons
