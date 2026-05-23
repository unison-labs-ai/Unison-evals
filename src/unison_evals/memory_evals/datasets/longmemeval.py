"""LongMemEval — long-term conversational memory benchmark.

Paper: arxiv 2410.10813 (ICLR 2025).
HuggingFace: xiaowu0162/longmemeval
License: MIT

Each question contains a "haystack" of conversation sessions and a question
about facts mentioned in those sessions. We flatten the haystack into a
single oracle context for Track 2 evaluation. (Track 3 will ingest each
session as a separate document — added in v0.1.)

Three official subsets exist; we use `longmemeval_s` (smallest) for fast
iteration and `longmemeval_oracle` for cleaner Track 2 runs where
distractor sessions are pre-filtered.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger

from ...config import get_settings
from ...types import BrainQuestion, Document, Question, Track
from .base import Dataset

# HuggingFace dataset id. Pinned per release to keep numbers comparable.
# As of 2026-05, the original `xiaowu0162/longmemeval` was deprecated and
# its data files removed. The canonical replacement is `-cleaned` which
# strips noisy history sessions that interfered with answer correctness.
HF_DATASET = "xiaowu0162/longmemeval-cleaned"
# Use the "oracle" split which strips the noisy distractor sessions, so
# Track 2 measures pure reasoning quality on the relevant context.
DEFAULT_SPLIT = "longmemeval_oracle"


class LongMemEvalDataset(Dataset):
    name = "longmemeval"
    description = (
        "500 questions across 5 memory abilities (single-session, multi-session, "
        "temporal reasoning, knowledge update, abstention). 53-session haystacks. "
        "ICLR 2025."
    )
    total_questions = 500
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
        """Track 1 variant — each question's haystack sessions become Documents.

        Path scheme: ``/sessions/{session_id}.md`` when ``haystack_session_ids``
        are present; ``/sessions/{idx}.md`` otherwise. Gold doc paths are the
        session paths corresponding to ``answer_session_ids``. When
        ``answer_session_ids`` is missing or empty, gold_doc_paths is an empty
        set (the question is treated as unanswerable for retrieval scoring).
        """
        rows = self._load_raw_rows()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_brain_question(row)

    @staticmethod
    def _row_to_brain_question(row: dict) -> BrainQuestion:
        """Convert one LongMemEval row to a BrainQuestion.

        Each haystack session becomes one Document. The session's path is
        ``/sessions/{session_id}.md`` when the row supplies
        ``haystack_session_ids``; otherwise ``/sessions/{idx}.md``.
        Gold paths are built by mapping ``answer_session_ids`` through the
        same scheme — positional when the IDs are positional integers/indices,
        or direct string matching when they are explicit session ID strings.
        """
        qid = str(row.get("question_id") or row.get("id") or row.get("qid") or _stable_id(row))
        query = str(row.get("question") or row.get("query") or "")
        haystack: list = row.get("haystack_sessions") or row.get("sessions") or []
        dates: list = row.get("haystack_dates") or [None] * len(haystack)
        session_ids: list | None = row.get("haystack_session_ids")
        answer_session_ids: list = row.get("answer_session_ids") or []

        # Build one Document per session.
        corpus: list[Document] = []
        path_by_session_id: dict[str, str] = {}
        for idx, session in enumerate(haystack):
            sid = str(session_ids[idx]) if session_ids and idx < len(session_ids) else str(idx)
            path = f"/sessions/{sid}.md"
            date = dates[idx] if idx < len(dates) else None
            body = _format_session(session, date, idx)
            corpus.append(
                Document(path=path, body=body, metadata={"session_id": sid, "date": date})
            )
            path_by_session_id[sid] = path

        # Compute gold_doc_paths from answer_session_ids.
        gold_paths: set[str] = set()
        for asid in answer_session_ids:
            key = str(asid)
            if key in path_by_session_id:
                gold_paths.add(path_by_session_id[key])
            # If not found by direct key, skip — no dangling gold path.

        return BrainQuestion(
            id=qid,
            query=query,
            corpus=corpus,
            gold_doc_paths=gold_paths,
            metadata={
                "question_type": row.get("question_type"),
                "answer_session_ids": answer_session_ids,
                "expected_answer": str(row.get("answer") or row.get("expected_answer") or ""),
            },
        )

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        """Load from HuggingFace, falling back to a tiny embedded sample if
        the network is unavailable.

        We use streaming=True to skip generating the broken `longmemeval_m_cleaned`
        split (which has type-inconsistent `answer` columns and breaks PyArrow
        conversion). With streaming, only the requested split is iterated.
        """
        try:
            ds = load_dataset(
                HF_DATASET,
                split=self.split,
                streaming=True,  # avoid generating other (broken) splits
            )
            # Materialize only the requested split.
            return list(ds)
        except Exception as e:
            logger.warning(
                "Failed to load LongMemEval from HuggingFace, falling back to "
                "embedded smoke set. error={}",
                e,
            )
            return _EMBEDDED_SMOKE_ROWS

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> Question:
        """Map a LongMemEval row to our common Question shape.

        LongMemEval columns vary by version; we read defensively. The
        typical schema:
          - question_id: str
          - question: str
          - answer: str
          - question_type: str (single-session-user, multi-session, etc.)
          - haystack_sessions: list[list[{role, content}]]
          - haystack_session_ids: list[str]
          - haystack_dates: list[str]
          - answer_session_ids: list[str]
        """
        qid = str(row.get("question_id") or row.get("id") or row.get("qid") or _stable_id(row))
        question = str(row.get("question") or row.get("query") or "")
        expected = str(row.get("answer") or row.get("expected_answer") or "")
        haystack = row.get("haystack_sessions") or row.get("sessions") or []
        dates = row.get("haystack_dates") or [None] * len(haystack)
        oracle_context = _format_haystack(haystack, dates)
        return Question(
            id=qid,
            question=question,
            expected_answer=expected,
            oracle_context=oracle_context,
            metadata={
                "question_type": row.get("question_type"),
                "haystack_session_ids": row.get("haystack_session_ids"),
                "answer_session_ids": row.get("answer_session_ids"),
            },
        )


def _format_session(
    session: list[dict[str, Any]] | Any,
    date: str | None,
    idx: int,
) -> str:
    """Render one haystack session as a markdown document body.

    Mirrors ``_format_haystack`` but produces a self-contained document for
    a single session (no cross-session concatenation).
    """
    header = f"## Session {idx + 1}"
    if date:
        header += f" — {date}"

    parts: list[str] = [header]
    if not isinstance(session, list):
        parts.append(str(session))
        return "\n".join(parts)

    for turn in session:
        if isinstance(turn, dict):
            role = str(turn.get("role", "?")).upper()
            content = str(turn.get("content", ""))
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(turn))
    return "\n".join(parts)


def _stable_id(row: dict[str, Any]) -> str:
    """Deterministic id when no upstream id exists — hash the question."""
    import hashlib

    seed = json.dumps(row, sort_keys=True, default=str)
    return "lme-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


def _format_haystack(
    sessions: list[list[dict[str, Any]]] | list[Any],
    dates: list[str | None],
) -> str:
    """Flatten haystack sessions into a single readable context string.

    Each session is rendered as a dated conversation. The model sees
    everything it would have seen across sessions in one pass.
    """
    if not sessions:
        return ""

    parts: list[str] = []
    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else None
        header = f"## Session {i + 1}"
        if date:
            header += f" — {date}"
        parts.append(header)

        if not isinstance(session, list):
            parts.append(str(session))
            continue

        for turn in session:
            if isinstance(turn, dict):
                role = str(turn.get("role", "?")).upper()
                content = str(turn.get("content", ""))
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(turn))
        parts.append("")  # blank line between sessions

    return "\n".join(parts)


# Tiny embedded sample so smoke tests / offline dev work without a HF download.
# Three real-shape rows to exercise the pipeline.
_EMBEDDED_SMOKE_ROWS: list[dict[str, Any]] = [
    {
        "question_id": "smoke-001",
        "question_type": "single-session-user",
        "question": "What time did I say my flight to Berlin lands?",
        "answer": "9:45 PM",
        "haystack_sessions": [
            [
                {"role": "user", "content": "Booked my flight to Berlin — lands 9:45 PM Friday."},
                {"role": "assistant", "content": "Got it, noted."},
            ],
        ],
        "haystack_dates": ["2026-04-08"],
        "haystack_session_ids": ["s1"],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "smoke-002",
        "question_type": "knowledge-update",
        "question": "What is the current address of my dentist?",
        "answer": "200 Hauptstrasse, Berlin",
        "haystack_sessions": [
            [
                {"role": "user", "content": "My dentist Dr Schmidt is at 14 Friedrichstrasse."},
                {"role": "assistant", "content": "Noted."},
            ],
            [
                {"role": "user", "content": "Dr Schmidt moved her practice to 200 Hauptstrasse."},
                {"role": "assistant", "content": "Updated."},
            ],
        ],
        "haystack_dates": ["2026-01-05", "2026-04-12"],
        "haystack_session_ids": ["s1", "s2"],
        "answer_session_ids": ["s2"],
    },
    {
        "question_id": "smoke-003",
        "question_type": "multi-session",
        "question": "Who was the speaker at the conference my colleague mentioned?",
        "answer": "Yann LeCun",
        "haystack_sessions": [
            [
                {"role": "user", "content": "Anna told me about an AI conference next month."},
                {"role": "assistant", "content": "Sounds interesting!"},
            ],
            [
                {
                    "role": "user",
                    "content": "Anna said the keynote at that conference is by Yann LeCun.",
                },
                {"role": "assistant", "content": "Got it."},
            ],
        ],
        "haystack_dates": ["2026-03-10", "2026-03-15"],
        "haystack_session_ids": ["s1", "s2"],
        "answer_session_ids": ["s1", "s2"],
    },
]
