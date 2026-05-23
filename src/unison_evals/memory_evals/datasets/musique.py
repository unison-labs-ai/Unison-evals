"""MuSiQue — Multihop Questions via Single-hop Question Composition.

Paper: arxiv 2108.00573 (EMNLP 2022).
HuggingFace: dgslibisey/MuSiQue
License: CC BY 4.0

MuSiQue is a 2-4 hop multi-hop QA benchmark with ~25 k questions, designed to
be substantially harder than HotpotQA by requiring genuine multi-hop reasoning
chains that cannot be short-circuited via single-hop shortcuts. Each question
arrives with exactly 20 supporting paragraphs (its own private corpus), of
which 2-5 are gold (is_supporting=True). The benchmark evaluates both the
final answer string and which paragraphs the system identified as evidence.

We concatenate all 20 paragraphs — in original order, with index + title
headers — into oracle_context for Track 2 (agent-oracle) evaluation: given
perfect retrieval, can the agent reason across the chain? Track 3 evaluation
(with retrieval) is a natural extension: treat each paragraph as a separate
document and score the brain's ability to surface the gold ones.

Metadata carries the full paragraphs list, a list of gold paragraph indexes
(filtered by is_supporting), the question_decomposition sub-steps (useful for
diagnostic analysis), and the answerable flag (some MuSiQue questions are
deliberately unanswerable).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from loguru import logger

from ...config import get_settings
from ...types import BrainQuestion, Document, Question, Track
from .base import Dataset

HF_DATASET = "dgslibisey/MuSiQue"
DEFAULT_SPLIT = "train"


class MuSiQueDataset(Dataset):
    name = "musique"
    description = "Multi-hop QA, 2-4 hops, ~25k questions, gold paragraphs per question. CC BY 4.0."
    total_questions = 25_000
    supported_tracks = frozenset({Track.AGENT_ORACLE, Track.AGENT_E2E, Track.BRAIN_ONLY})

    def __init__(self, split: str = DEFAULT_SPLIT) -> None:
        self.split = split
        self.settings = get_settings()

    def load(self, limit: int | None = None) -> Iterable[Question]:
        rows = self._load_raw_rows()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_question(row)

    def load_brain_questions(self, limit: int | None = None) -> Iterable[BrainQuestion]:
        """Track 1 variant — each question's paragraphs become Documents.

        Path scheme: ``/paragraphs/{idx}-{slug(title)}.md``. Gold doc paths
        are the paths of paragraphs where ``is_supporting=True``.
        """
        rows = self._load_raw_rows()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_brain_question(row)

    @staticmethod
    def _row_to_brain_question(row: dict[str, Any]) -> BrainQuestion:
        """Convert one MuSiQue row to a BrainQuestion.

        Each paragraph in ``row["paragraphs"]`` becomes one Document with
        path ``/paragraphs/{idx}-{slug(title)}.md``. Gold paths are the
        paths of paragraphs where ``is_supporting=True``.
        """
        qid = str(row.get("id") or row.get("qid") or _stable_id(row))
        query = str(row.get("question") or row.get("query") or "")
        paragraphs: list[dict[str, Any]] = list(row.get("paragraphs") or [])

        corpus: list[Document] = []
        gold_paths: set[str] = set()

        for pos, para in enumerate(paragraphs):
            idx = para.get("idx", pos)
            title = str(para.get("title") or "").strip() or f"paragraph-{idx}"
            text = str(para.get("paragraph_text") or "").strip()
            path = f"/paragraphs/{idx}-{_slug(title)}.md"
            body = f"# {title}\n\n{text}" if text else f"# {title}"
            corpus.append(
                Document(
                    path=path,
                    body=body,
                    metadata={
                        "idx": idx,
                        "title": title,
                        "is_supporting": para.get("is_supporting", False),
                    },
                )
            )
            if para.get("is_supporting", False):
                gold_paths.add(path)

        return BrainQuestion(
            id=qid,
            query=query,
            corpus=corpus,
            gold_doc_paths=gold_paths,
            metadata={
                "answerable": row.get("answerable"),
                "question_decomposition": row.get("question_decomposition"),
                "expected_answer": str(row.get("answer") or row.get("expected_answer") or ""),
            },
        )

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        """Load from HuggingFace, falling back to a tiny embedded sample if
        the network is unavailable. The fallback ensures `unison-evals run`
        works in offline / CI environments without surprise."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("`datasets` library not installed. Run `uv sync`.") from e

        try:
            ds = load_dataset(
                HF_DATASET,
                split=self.split,
                cache_dir=str(self.settings.cache_dir / "hf"),
            )
            return list(ds)
        except Exception as e:
            logger.warning(
                "Failed to load MuSiQue from HuggingFace, falling back to "
                "embedded smoke set. error={}",
                e,
            )
            return _EMBEDDED_SMOKE_ROWS

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> Question:
        """Map a MuSiQue row to our common Question shape.

        MuSiQue columns (typical schema):
          - id: str
          - question: str
          - answer: str
          - paragraphs: list[{idx, title, paragraph_text, is_supporting}]
          - question_decomposition: list[{id, question, answer, paragraph_support_idx}]
          - answerable: bool
        """
        qid = str(row.get("id") or row.get("qid") or _stable_id(row))
        question = str(row.get("question") or row.get("query") or "")
        expected = str(row.get("answer") or row.get("expected_answer") or "")
        paragraphs: list[dict[str, Any]] = list(row.get("paragraphs") or [])
        oracle_context = _format_paragraphs(paragraphs)
        gold_indexes = [
            p.get("idx", i) for i, p in enumerate(paragraphs) if p.get("is_supporting", False)
        ]
        return Question(
            id=qid,
            question=question,
            expected_answer=expected,
            oracle_context=oracle_context,
            metadata={
                "paragraphs": paragraphs,
                "gold_paragraph_indexes": gold_indexes,
                "question_decomposition": row.get("question_decomposition"),
                "answerable": row.get("answerable"),
            },
        )


def _slug(text: str) -> str:
    """Convert a title to a filesystem-safe slug for use in doc paths.

    Lowercases, replaces non-alphanumeric characters with hyphens, and
    collapses consecutive hyphens. Truncates to 48 characters so paths
    stay readable even for verbose paragraph titles.
    """
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "untitled"


def _stable_id(row: dict[str, Any]) -> str:
    """Deterministic id when no upstream id exists — hash the question."""
    import hashlib

    seed = json.dumps(row, sort_keys=True, default=str)
    return "msq-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


def _format_paragraphs(paragraphs: list[dict[str, Any]]) -> str:
    """Render paragraphs as a numbered, titled context block.

    Each paragraph gets a ``### [N] Title`` header so the model can reason
    about which one contributed to the answer if it cites paragraph indexes
    in its chain-of-thought. Missing title or paragraph_text fields are
    handled gracefully — the block is emitted with whatever is present.
    """
    if not paragraphs:
        return ""

    parts: list[str] = []
    for i, para in enumerate(paragraphs):
        idx = para.get("idx", i)
        title = str(para.get("title") or "").strip() or f"Paragraph {idx}"
        text = str(para.get("paragraph_text") or "").strip()
        parts.append(f"### [{idx}] {title}")
        if text:
            parts.append(text)
        parts.append("")  # blank line between paragraphs

    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Embedded smoke set — 3 real-shape rows for offline / CI use.
# Questions are genuinely solvable from the supplied paragraphs.
# ---------------------------------------------------------------------------

_EMBEDDED_SMOKE_ROWS: list[dict[str, Any]] = [
    {
        "id": "smoke-musique-001",
        "question": "What is the capital of the country where the painter of the Mona Lisa was born?",
        "answer": "Rome",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Mona Lisa",
                "paragraph_text": "The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Leonardo da Vinci.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Leonardo da Vinci",
                "paragraph_text": "Leonardo da Vinci was an Italian polymath born in 1452 in Vinci, a town in the Republic of Florence, which is now part of Italy.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "Italy",
                "paragraph_text": "Italy is a country in southern Europe. Its capital city is Rome, which also serves as the seat of the national government.",
                "is_supporting": True,
            },
            {
                "idx": 3,
                "title": "France",
                "paragraph_text": "France is a country in Western Europe. Its capital is Paris.",
                "is_supporting": False,
            },
            {
                "idx": 4,
                "title": "Spain",
                "paragraph_text": "Spain is a country on the Iberian Peninsula. Its capital is Madrid.",
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Who painted the Mona Lisa?",
                "answer": "Leonardo da Vinci",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "Where was Leonardo da Vinci born?",
                "answer": "Italy",
                "paragraph_support_idx": 1,
            },
            {
                "id": 3,
                "question": "What is the capital of Italy?",
                "answer": "Rome",
                "paragraph_support_idx": 2,
            },
        ],
        "answerable": True,
    },
    {
        "id": "smoke-musique-002",
        "question": "In what year was the author of 'Pride and Prejudice' born, and what is the birth year of the composer of the Symphony No. 5 in C minor?",
        "answer": "1775 and 1770",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Pride and Prejudice",
                "paragraph_text": "Pride and Prejudice is a novel written by Jane Austen, first published in 1813.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Jane Austen",
                "paragraph_text": "Jane Austen was an English novelist born on 16 December 1775 in Steventon, Hampshire.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "Symphony No. 5 (Beethoven)",
                "paragraph_text": "Symphony No. 5 in C minor is a symphony composed by Ludwig van Beethoven between 1804 and 1808.",
                "is_supporting": True,
            },
            {
                "idx": 3,
                "title": "Ludwig van Beethoven",
                "paragraph_text": "Ludwig van Beethoven was a German composer born on 17 December 1770 in Bonn.",
                "is_supporting": True,
            },
            {
                "idx": 4,
                "title": "Charles Dickens",
                "paragraph_text": "Charles Dickens was an English writer born on 7 February 1812 in Portsmouth.",
                "is_supporting": False,
            },
            {
                "idx": 5,
                "title": "Wolfgang Amadeus Mozart",
                "paragraph_text": "Wolfgang Amadeus Mozart was a prolific composer born on 27 January 1756 in Salzburg.",
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Who wrote 'Pride and Prejudice'?",
                "answer": "Jane Austen",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "When was Jane Austen born?",
                "answer": "1775",
                "paragraph_support_idx": 1,
            },
            {
                "id": 3,
                "question": "Who composed Symphony No. 5 in C minor?",
                "answer": "Ludwig van Beethoven",
                "paragraph_support_idx": 2,
            },
            {
                "id": 4,
                "question": "When was Ludwig van Beethoven born?",
                "answer": "1770",
                "paragraph_support_idx": 3,
            },
        ],
        "answerable": True,
    },
    {
        "id": "smoke-musique-003",
        "question": "What ocean borders the country that is home to the city where the Eiffel Tower is located?",
        "answer": "Atlantic Ocean",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Eiffel Tower",
                "paragraph_text": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "France",
                "paragraph_text": "France is a country in Western Europe bordered to the west by the Atlantic Ocean.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "Atlantic Ocean",
                "paragraph_text": "The Atlantic Ocean is the second-largest ocean on Earth, separating the Americas from Europe and Africa.",
                "is_supporting": True,
            },
            {
                "idx": 3,
                "title": "Pacific Ocean",
                "paragraph_text": "The Pacific Ocean is the largest ocean on Earth, bordered by Asia and Australia to the west.",
                "is_supporting": False,
            },
            {
                "idx": 4,
                "title": "Germany",
                "paragraph_text": "Germany is a country in Central Europe bordered by the North Sea and Baltic Sea.",
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Where is the Eiffel Tower located?",
                "answer": "Paris, France",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "What ocean borders France?",
                "answer": "Atlantic Ocean",
                "paragraph_support_idx": 1,
            },
        ],
        "answerable": True,
    },
]
