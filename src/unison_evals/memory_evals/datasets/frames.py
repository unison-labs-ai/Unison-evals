"""FRAMES — Factuality, Retrieval, And Reasoning MEasurement Set.

Paper: arxiv 2409.12941 (NAACL 2025).
HuggingFace: google/frames-benchmark
License: Apache 2.0

FRAMES is a DeepMind/Harvard/Meta benchmark of 824 multi-hop questions that
test end-to-end RAG factuality + retrieval + reasoning together. Every
question requires synthesising information across multiple Wikipedia articles.
SOTA without retrieval is ~40%; with multi-step retrieval pipelines, ~66%.
Each row ships gold Wikipedia URLs (`wiki_links`) that the Track 1 retrieval
scorer uses to measure recall of relevant documents.

oracle_context strategy (Track 2):
  We set oracle_context=None for all FRAMES questions. FRAMES was designed to
  test retrieval pipelines, and the ~40% SOTA measured on raw LLM knowledge
  *without* retrieval is the natural Track 2 baseline. Supplying oracle context
  from the wiki_links URLs would require fetching and parsing live Wikipedia
  pages, which creates non-determinism and network dependencies; that is Track
  3 territory (v0.2). For Track 2 runs the adapter receives the question and
  must answer from parametric knowledge — the same regime the FRAMES authors
  use for their no-retrieval baseline. Runners that want full Track 3 should
  fetch each wiki_links URL, ingest the article body as a Document, and then
  the brain's retrieved text becomes the context.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from loguru import logger

from ...config import get_settings
from ...types import Question, Track
from .base import Dataset

HF_DATASET = "google/frames-benchmark"
DEFAULT_SPLIT = "test"


class FramesDataset(Dataset):
    name = "frames"
    description = (
        "824 multi-hop questions requiring synthesis across multiple Wikipedia "
        "articles. Tests factuality, retrieval, and reasoning together. NAACL 2025."
    )
    total_questions = 824
    # FRAMES does not yet have a BrainQuestion converter — Track 2 only.
    supported_tracks = frozenset({Track.AGENT_ORACLE})

    def __init__(self) -> None:
        self.settings = get_settings()

    def load(self, limit: int | None = None) -> Iterable[Question]:
        rows = self._load_raw_rows()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_question(row)

    def load_brain_questions(self, limit: int | None = None) -> None:  # type: ignore[override]
        """Track 1 converter deferred to v0.2 — needs Wikipedia corpus loader.

        FRAMES questions reference gold Wikipedia URLs (``wiki_links``) but
        ship no inline corpus. Implementing Track 1 requires fetching and
        parsing each Wikipedia article at question-load time, which creates
        network dependencies and non-determinism incompatible with offline CI
        runs.

        v0.2 plan: add a ``WikipediaCorpusLoader`` that fetches, caches, and
        converts Wikipedia articles to Documents, then wire it here.

        Raises:
            NotImplementedError: always. Use ``load()`` for Track 2 (agent-oracle
                with parametric knowledge, no corpus injection).
        """
        raise NotImplementedError(
            "FRAMES Track 1 is deferred to v0.2 — needs a Wikipedia corpus loader. "
            "Each FRAMES question references gold Wikipedia URLs (wiki_links) but "
            "ships no inline corpus. Use load() for Track 2 evaluation instead."
        )

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        """Load from HuggingFace, falling back to embedded smoke rows when the
        network is unavailable. The fallback keeps offline/CI runs working."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("`datasets` library not installed. Run `uv sync`.") from e

        try:
            ds = load_dataset(
                HF_DATASET,
                split=DEFAULT_SPLIT,
                cache_dir=str(self.settings.cache_dir / "hf"),
            )
            return list(ds)
        except Exception as e:
            logger.warning(
                "Failed to load FRAMES from HuggingFace, falling back to "
                "embedded smoke set. error={}",
                e,
            )
            return _EMBEDDED_SMOKE_ROWS

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> Question:
        """Map a FRAMES row to the common Question shape.

        FRAMES columns (typical production schema):
          - Prompt: str — the multi-hop question
          - Answer: str — gold answer
          - wiki_links: list[str] — gold Wikipedia URLs (gold doc paths for Track 1)
          - reasoning_types: str — e.g. "Multi-hop", "Inference", "Comparison"
          - topic: str — broad topic category

        We read all fields defensively to survive minor schema changes.
        """
        qid = str(row.get("question_id") or row.get("id") or row.get("qid") or _stable_id(row))
        question = str(row.get("Prompt") or row.get("prompt") or row.get("question") or "")
        expected = str(row.get("Answer") or row.get("answer") or row.get("gold_answer") or "")

        wiki_links = row.get("wiki_links") or []
        if isinstance(wiki_links, str):
            wiki_links = [w.strip() for w in wiki_links.split(",") if w.strip()]

        return Question(
            id=qid,
            question=question,
            expected_answer=expected,
            oracle_context=None,
            metadata={
                "wiki_links": wiki_links,
                "reasoning_types": row.get("reasoning_types"),
                "topic": row.get("topic"),
                "original_id": row.get("question_id") or row.get("id"),
            },
        )


def _stable_id(row: dict[str, Any]) -> str:
    """Deterministic id when no upstream id exists — hash the question text."""
    import hashlib

    seed = json.dumps(row, sort_keys=True, default=str)
    return "frames-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


# Embedded smoke rows for offline/CI use. Three real-shape rows that exercise
# the multi-hop reasoning the benchmark is designed to test.
_EMBEDDED_SMOKE_ROWS: list[dict[str, Any]] = [
    {
        "question_id": "frames-smoke-001",
        "Prompt": (
            "What was the first feature film directed by the person "
            "who won the Academy Award for Best Director in 1994?"
        ),
        "Answer": "Grand Theft Auto",
        "wiki_links": [
            "https://en.wikipedia.org/wiki/66th_Academy_Awards",
            "https://en.wikipedia.org/wiki/Ron_Howard",
            "https://en.wikipedia.org/wiki/Grand_Theft_Auto_(film)",
        ],
        "reasoning_types": "Multi-hop",
        "topic": "Film",
    },
    {
        "question_id": "frames-smoke-002",
        "Prompt": ("Which novel by the author of A Tale of Two Cities was published first?"),
        "Answer": "The Pickwick Papers",
        "wiki_links": [
            "https://en.wikipedia.org/wiki/A_Tale_of_Two_Cities",
            "https://en.wikipedia.org/wiki/Charles_Dickens",
            "https://en.wikipedia.org/wiki/The_Pickwick_Papers",
        ],
        "reasoning_types": "Multi-hop",
        "topic": "Literature",
    },
    {
        "question_id": "frames-smoke-003",
        "Prompt": (
            "What is the capital of the country where the headquarters "
            "of the company that manufactures the iPhone is located?"
        ),
        "Answer": "Washington, D.C.",
        "wiki_links": [
            "https://en.wikipedia.org/wiki/IPhone",
            "https://en.wikipedia.org/wiki/Apple_Inc.",
            "https://en.wikipedia.org/wiki/United_States",
        ],
        "reasoning_types": "Multi-hop",
        "topic": "Technology",
    },
    {
        "question_id": "frames-smoke-004",
        "Prompt": (
            "How many years after the founding of the city that hosted "
            "the 1992 Summer Olympics was the Olympic Games first held there?"
        ),
        "Answer": "2000",
        "wiki_links": [
            "https://en.wikipedia.org/wiki/1992_Summer_Olympics",
            "https://en.wikipedia.org/wiki/Barcelona",
        ],
        "reasoning_types": "Multi-hop, Arithmetic",
        "topic": "Sports",
    },
]
