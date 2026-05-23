"""BitempoQA — bitemporal question-answering benchmark.

A hand-curated dataset of 100 questions over a synthetic SaaS/tech-industry
corpus of 110 atomic facts, each carrying valid_from / valid_to / supersedes
metadata. The dataset probes four question types:

  current_truth   — what is X right now
  historical_truth — what was X on date D  (uses as_of)
  predecessor     — who came before Y as X's attribute
  transition      — when did X change

License: CC BY 4.0
Schema: see data/bitempoqa/README.md
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ...types import BrainQuestion, Document, Question, Track
from .base import Dataset

_DATA_DIR = Path(__file__).parents[4] / "data" / "bitempoqa"


class BitempoQADataset(Dataset):
    name = "bitempoqa"
    description = (
        "100 bitemporal QA questions over 110 synthetic SaaS/tech facts "
        "(valid_from/valid_to/supersedes). Four question types: current_truth, "
        "historical_truth, predecessor, transition. License CC BY 4.0."
    )
    total_questions = 100
    supported_tracks = frozenset({Track.AGENT_ORACLE, Track.AGENT_E2E, Track.BRAIN_ONLY})

    def __init__(self) -> None:
        self.data_dir = _DATA_DIR

    def load(self, limit: int | None = None) -> Iterable[Question]:
        corpus = self._load_corpus()
        for i, row in enumerate(self._load_questions()):
            if limit is not None and i >= limit:
                return
            yield self._row_to_question(row, corpus)

    def load_brain_questions(self, limit: int | None = None) -> Iterable[BrainQuestion]:
        """Track 1 variant — each question carries the full BitempoQA corpus.

        Every BrainQuestion gets the same corpus (all 110+ facts) because
        BitempoQA is a closed-world dataset where all facts are always
        relevant to the eval. Gold doc paths are derived from the question's
        ``fact_ids`` — each fact maps to ``/facts/{fact_id}.md``.
        """
        corpus_raw = self._load_corpus()
        corpus_docs = [
            Document(
                path=f"/facts/{fact_id}.md",
                body=self._fact_to_markdown(fact),
                metadata={
                    "fact_id": fact_id,
                    "valid_from": fact.get("valid_from"),
                    "valid_to": fact.get("valid_to"),
                    "supersedes": fact.get("supersedes"),
                },
            )
            for fact_id, fact in corpus_raw.items()
        ]
        for i, row in enumerate(self._load_questions()):
            if limit is not None and i >= limit:
                return
            fact_ids: list[str] = row.get("fact_ids", [])
            gold_paths = {f"/facts/{fid}.md" for fid in fact_ids if fid in corpus_raw}
            # expected_versions maps doc_path → fact_id for temporal scoring.
            # A question's gold facts are the temporally-correct documents for
            # the as_of timestamp. The fact_id encodes the version ("f001",
            # "f002", …) so the temporal metric can verify the top-1 doc
            # path contains the right fact_id substring.
            expected_versions: dict[str, str] = {
                f"/facts/{fid}.md": fid for fid in fact_ids if fid in corpus_raw
            }
            yield BrainQuestion(
                id=row["id"],
                query=row["question"],
                corpus=corpus_docs,
                gold_doc_paths=gold_paths,
                metadata={
                    "as_of": row.get("as_of"),
                    "fact_ids": fact_ids,
                    "question_type": row.get("question_type"),
                    "difficulty": row.get("difficulty"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_versions": expected_versions,
                },
            )

    def _load_corpus(self) -> dict[str, dict]:
        path = self.data_dir / "corpus.jsonl"
        corpus: dict[str, dict] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fact = json.loads(line)
                corpus[fact["fact_id"]] = fact
        return corpus

    def _load_questions(self) -> Iterable[dict]:
        path = self.data_dir / "questions.jsonl"
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _fact_to_markdown(fact: dict) -> str:
        """Render a BitempoQA corpus fact as a short markdown document."""
        valid_to = fact.get("valid_to") or "present"
        return (
            f"# {fact.get('subject')} — {fact.get('predicate')}\n\n"
            f"**Object:** {fact.get('object')}\n\n"
            f"**Valid:** {fact.get('valid_from')} → {valid_to}\n"
        )

    @staticmethod
    def _row_to_question(row: dict, corpus: dict[str, dict]) -> Question:
        oracle_parts: list[str] = []
        for fid in row.get("fact_ids", []):
            fact = corpus.get(fid)
            if fact is None:
                continue
            valid_to = fact["valid_to"] or "present"
            oracle_parts.append(
                f"{fact['subject']} {fact['predicate']} {fact['object']}"
                f" (valid {fact['valid_from']} → {valid_to})"
            )

        as_of = row.get("as_of")
        if as_of:
            oracle_parts.append(f"[Question asks about the state as of {as_of}]")

        oracle_context = "\n".join(oracle_parts) if oracle_parts else None

        return Question(
            id=row["id"],
            question=row["question"],
            expected_answer=row["expected_answer"],
            oracle_context=oracle_context,
            metadata={
                "as_of": as_of,
                "fact_ids": row.get("fact_ids", []),
                "question_type": row.get("question_type"),
                "difficulty": row.get("difficulty"),
            },
        )
