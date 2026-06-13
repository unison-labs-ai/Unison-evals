"""Adapter contracts — every system being evaluated implements one of these.

Two contracts:
  * `AgentAdapter` for Track 2 (oracle) and Track 3 (E2E) — answers questions.
  * `BrainAdapter` for brain retrieval — ingests docs and returns ranked chunks.

The contracts are intentionally small (one required method each) so
adding a new adapter is ~80 LOC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...types import AdapterResult, BrainSearchResult, Document


class AgentAdapter(ABC):
    """Abstract base for any agent system being evaluated.

    Subclasses must define `name` (registry key) and implement `answer()`.
    """

    name: str

    async def setup(self) -> None:
        """Optional one-time setup (auth, container start, etc.).

        Called once before the first `answer()` call. Default is no-op.
        """
        return None

    @abstractmethod
    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
        question_id: str | None = None,
        corpus_key: str | None = None,
    ) -> AdapterResult:
        """Answer one question.

        question_id (optional): the dataset id, used to reuse a pre-ingested
        persistent workspace from the manifest when one exists.
        corpus_key (optional): stable key for this question's CORPUS. Questions
        sharing a corpus (e.g. LOCOMO) share a key so the brain is seeded once
        per corpus and reused. Defaults to question_id.

        Args:
            question: The user-facing question text.
            oracle_context: Track 2 — gold context. When provided, the adapter
                must use ONLY this context to answer — no retrieval, no memory
                lookup. Mutually exclusive with seed_docs.
            seed_docs: Track 3 — per-question corpus to ingest before answering.
                When provided, the adapter writes these into its brain (or its
                best equivalent) and then answers using its full agent loop.
                Mutually exclusive with oracle_context.

        When BOTH are None: Track 2 with no oracle context (no-memory baseline)
        or Track 3 against a pre-loaded corpus.
        When BOTH are non-None: return AdapterResult with
        error="seed_docs and oracle_context are mutually exclusive".

        Returns:
            AdapterResult with answer text, cost, latency, and raw response.
            On error, set `error` and leave `answer` empty.
        """

    async def teardown(self) -> None:
        """Optional cleanup (close HTTP client, kill containers). Default no-op."""
        return None


class BrainAdapter(ABC):
    """Abstract base for brain / memory / vector-store systems being evaluated
    on Track 1 (retrieval ranking quality).

    A brain adapter must support two operations:
      * `ingest(docs)` — bulk-load documents into the system.
      * `search(query, k)` — return the top-k most relevant chunks.

    Optionally:
      * `reset()` — wipe all ingested data so the next ingest starts clean.
        Used between eval questions when each question has its own corpus
        (LongMemEval, MemoryAgentBench). Defaults to a no-op; adapters that
        support reset override it here.

    The contract is dataset-agnostic. Datasets convert their per-question
    haystack into `list[Document]`; the runner calls `reset()` then
    `ingest()` then iterates `search()` per query.
    """

    name: str

    async def setup(self) -> None:
        """Optional one-time setup (DB connection, auth, schema migration).
        Called once before the first ingest/search."""
        return None

    @abstractmethod
    async def ingest(self, docs: list[Document]) -> None:
        """Bulk-load documents. Implementations should batch internally for
        throughput. Idempotency is NOT required — the runner calls `reset()`
        first when a clean slate is needed.

        Raises if ingestion fails fatally. Partial failures should be logged
        but not raised — let the search call surface the resulting recall hit.
        """

    @abstractmethod
    async def search(self, query: str, k: int = 10) -> BrainSearchResult:
        """Retrieve the top-k chunks for `query`.

        Returns:
            BrainSearchResult with up to `k` ranked chunks (1-indexed `rank`),
            wall-clock `latency_ms`, optional `cost_usd`, raw debug payload,
            and `error` set on failure (with empty chunks list).

        Determinism: implementations should produce stable results for
        identical (query, ingested-corpus) pairs. Tied scores get
        deterministic tiebreak (e.g. doc path).
        """

    async def reset(self) -> None:
        """Optional: wipe all ingested data. Default no-op for adapters
        backed by ephemeral storage (a fresh local Postgres instance, or
        Mem0 with per-call user_id) where reset is implicit."""
        return None

    async def teardown(self) -> None:
        """Optional cleanup (close DB pool, log out). Default no-op."""
        return None
