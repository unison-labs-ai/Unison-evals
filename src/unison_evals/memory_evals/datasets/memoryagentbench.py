"""MemoryAgentBench — multi-ability LLM memory benchmark (ICLR 2026).

Paper: arxiv 2507.05257 (ICLR 2026).
HuggingFace: ai-hyz/MemoryAgentBench
License: MIT

MemoryAgentBench is the most discriminating memory benchmark of 2025-2026.
It tests four memory abilities across multi-turn agent trajectories:

  1. Accurate Retrieval    — single-session lookup of specific facts.
  2. Test-Time Learning    — incorporating new skills/facts during interaction.
  3. Long-Range Understanding — synthesising across many turns / a long context.
  4. Conflict Resolution   — handling contradictory information over time
                             (the ability where bitemporal models like Unison's
                             structurally win; SOTA on this ability scores ~6%).

Dataset layout: each row contains a long `context` (the agent trajectory),
a list of `questions`, a parallel list of `answers`, and a `metadata` dict
that includes `qa_pair_ids`. The benchmark uses an "inject once, query
multiple times" design — one context yields 60-100 Q/A pairs. We explode
every row into one Question per (question, answer) pair.

Splits: one per memory ability.
  Accurate_Retrieval       22 rows
  Test_Time_Learning        6 rows
  Long_Range_Understanding 110 rows
  Conflict_Resolution       8 rows

HF id uncertainty: The official authors (HUST-AI-HYZ) host the canonical
evaluation code at github.com/HUST-AI-HYZ/MemoryAgentBench. The HF dataset
`ai-hyz/MemoryAgentBench` is the public upload linked from the paper and was
used here (MIT licence, verified 2026-05-10). If the authors publish a
different canonical HF id, update HF_DATASET and repin.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from loguru import logger

from ...config import get_settings
from ...types import BrainQuestion, Document, Question, Track
from .base import Dataset

# Imported at module scope so tests can monkeypatch it cleanly.
# Wrapped in try/except so the import chain doesn't fail if `datasets`
# is not installed (e.g. a minimal install without HF deps).
try:
    from datasets import load_dataset  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

# HuggingFace dataset id — verified 2026-05-10 (MIT licence).
HF_DATASET = "ai-hyz/MemoryAgentBench"

# All four ability splits.  Load them together for a unified dataset;
# callers can pass a specific split name to restrict to one ability.
_ALL_SPLITS = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
]

# Default to all splits so the dataset exercises all four abilities.
DEFAULT_SPLIT = "all"


class MemoryAgentBenchDataset(Dataset):
    """MemoryAgentBench: four memory abilities, ICLR 2026.

    Covers accurate retrieval, test-time learning, long-range understanding,
    and conflict resolution — the last being Unison's structural advantage.

    Pass `split="Conflict_Resolution"` (or any of the four HF split names)
    to restrict to a single ability. Default loads all four.
    """

    name = "memoryagentbench"
    description = (
        "Four memory abilities (accurate retrieval, test-time learning, "
        "long-range understanding, conflict resolution). 146 trajectory rows → "
        "~10K Q/A pairs. Bitemporal models win on conflict resolution. "
        "ICLR 2026 (arxiv 2507.05257)."
    )
    total_questions = 10_000
    supported_tracks = frozenset({Track.AGENT_ORACLE, Track.AGENT_E2E, Track.BRAIN_ONLY})

    def __init__(self, split: str = DEFAULT_SPLIT) -> None:
        self.split = split
        self.settings = get_settings()

    def load(self, limit: int | None = None) -> Iterable[Question]:
        rows = self._load_raw_rows()
        count = 0
        for row in rows:
            if limit is not None and count >= limit:
                return
            q = self._row_to_question(row)
            yield q
            count += 1

    def load_brain_questions(self, limit: int | None = None) -> Iterable[BrainQuestion]:
        """Track 1 variant — each Q/A pair's trajectory turns become Documents.

        The trajectory stored in ``oracle_context`` (or the raw ``context``
        field before explosion) is split into individual turns. Each turn
        becomes one Document with path ``/turns/{turn_idx:04d}.md``.

        Gold doc paths use a best-effort heuristic: turns whose text contains
        the expected answer (case-insensitive substring match). This mirrors
        the MemoryAgentBench evaluation approximation used by the paper's own
        evaluation code — there are no explicit gold paragraph annotations, so
        answer-in-turn is the standard proxy. When no turn contains the answer,
        gold_doc_paths is empty (the question is treated as unanswerable for
        retrieval scoring).

        Note: ``_EMBEDDED_SMOKE_ROWS`` are already-exploded rows with
        ``oracle_context`` as a pre-formatted string; when falling back to
        them we parse the trajectory from that field.
        """
        rows = self._load_raw_rows()
        count = 0
        for row in rows:
            if limit is not None and count >= limit:
                return
            yield self._row_to_brain_question(row)
            count += 1

    @staticmethod
    def _row_to_brain_question(row: dict[str, Any]) -> BrainQuestion:
        """Convert an (already-exploded) MemoryAgentBench row to a BrainQuestion.

        Trajectory turns are extracted from ``oracle_context`` (the pre-formatted
        form produced by ``_explode_row``). Each non-blank line becomes a Document.
        Gold paths are those of turns whose body contains the expected answer
        (case-insensitive substring). This is the standard MemoryAgentBench
        best-effort gold approximation — the dataset carries no explicit
        paragraph-level relevance labels.
        """
        qid = str(row.get("question_id") or row.get("id") or row.get("qid") or _stable_id(row))
        query = str(row.get("question") or row.get("query") or "")
        expected = str(
            row.get("answer") or row.get("expected_answer") or row.get("gold_answer") or ""
        )
        oracle_context: str = row.get("oracle_context") or ""

        # Strip the "## Trajectory\n\n" header if present.
        trajectory_text = oracle_context
        if trajectory_text.startswith("## Trajectory"):
            trajectory_text = trajectory_text[len("## Trajectory") :].lstrip("\n")

        turns = _split_trajectory_turns(trajectory_text)

        corpus: list[Document] = []
        gold_paths: set[str] = set()
        expected_lower = expected.lower()

        for idx, turn_text in enumerate(turns):
            path = f"/turns/{idx:04d}.md"
            corpus.append(Document(path=path, body=turn_text, metadata={"turn_idx": idx}))
            if expected_lower and expected_lower in turn_text.lower():
                gold_paths.add(path)

        return BrainQuestion(
            id=qid,
            query=query,
            corpus=corpus,
            gold_doc_paths=gold_paths,
            metadata={
                "memory_ability": row.get("memory_ability"),
                "turn_count": row.get("turn_count"),
                "source_row_id": row.get("source_row_id"),
                "expected_answer": expected,
            },
        )

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        """Load from HuggingFace, falling back to embedded smoke set if the
        network is unavailable.  The fallback ensures `unison-evals run`
        works in offline / CI environments without surprise.

        Uses the module-level `load_dataset` name so tests can monkeypatch
        it with `monkeypatch.setattr(mod, "load_dataset", ...)`.
        """
        import unison_evals.memory_evals.datasets.memoryagentbench as _mod

        _load_fn = _mod.load_dataset
        if _load_fn is None:
            raise RuntimeError("`datasets` library not installed. Run `uv sync`.")

        splits_to_load = _ALL_SPLITS if self.split == DEFAULT_SPLIT else [self.split]
        rows: list[dict[str, Any]] = []
        for split_name in splits_to_load:
            try:
                ds = _load_fn(
                    HF_DATASET,
                    split=split_name,
                    cache_dir=str(self.settings.cache_dir / "hf"),
                )
                for raw_row in ds:
                    # Explode: each (context, question[i], answer[i]) → one row.
                    rows.extend(_explode_row(dict(raw_row), split_name))
            except Exception as e:
                logger.warning(
                    "Failed to load MemoryAgentBench split '{}' from "
                    "HuggingFace, falling back to embedded smoke set. error={}",
                    split_name,
                    e,
                )
                return _EMBEDDED_SMOKE_ROWS

        if not rows:
            logger.warning(
                "MemoryAgentBench: no rows loaded from HuggingFace, "
                "falling back to embedded smoke set."
            )
            return _EMBEDDED_SMOKE_ROWS

        return rows

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> Question:
        """Map an already-exploded MemoryAgentBench row to a Question.

        After _explode_row, each row has exactly one question + answer pair.
        Defensive fallbacks handle schema drift between HF uploads.

        Expected keys after explosion:
          - question_id / id / qid   → Question.id
          - question / query         → Question.question
          - answer / expected_answer / gold_answer → Question.expected_answer
          - oracle_context           → Question.oracle_context (pre-formatted)
          - memory_ability           → metadata["memory_ability"]
          - turn_count               → metadata["turn_count"]
        """
        qid = str(row.get("question_id") or row.get("id") or row.get("qid") or _stable_id(row))
        question = str(row.get("question") or row.get("query") or "")
        expected = str(
            row.get("answer") or row.get("expected_answer") or row.get("gold_answer") or ""
        )
        oracle_context = row.get("oracle_context") or ""
        return Question(
            id=qid,
            question=question,
            expected_answer=expected,
            oracle_context=oracle_context if oracle_context else None,
            metadata={
                "memory_ability": row.get("memory_ability"),
                "turn_count": row.get("turn_count"),
                "source_row_id": row.get("source_row_id"),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _explode_row(
    raw: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    """Expand one HF row into N flattened rows — one per Q/A pair.

    HF schema:
      context:   str   — the full agent trajectory / document text
      questions: list[str]
      answers:   list[list[str]] — each element is a list of acceptable answers
      metadata:  dict  — qa_pair_ids, demo, haystack_sessions, keypoints, ...
    """
    context: str = str(raw.get("context") or "")
    questions: list[Any] = raw.get("questions") or []
    answers_outer: list[Any] = raw.get("answers") or []
    metadata: dict[str, Any] = raw.get("metadata") or {}
    qa_pair_ids: list[Any] = metadata.get("qa_pair_ids") or []
    turn_count = len(context.split("\n")) if context else 0
    memory_ability = _split_to_ability(split_name)

    oracle_context = _format_context(context)

    result: list[dict[str, Any]] = []
    for i, q_text in enumerate(questions):
        # answers[i] may be a list of acceptable strings; take first.
        answer_candidates: list[str] = []
        if i < len(answers_outer):
            cell = answers_outer[i]
            if isinstance(cell, list):
                answer_candidates = [str(a) for a in cell if a]
            elif cell is not None:
                answer_candidates = [str(cell)]
        gold = answer_candidates[0] if answer_candidates else ""

        qa_id = str(qa_pair_ids[i]) if i < len(qa_pair_ids) and qa_pair_ids[i] is not None else None

        result.append(
            {
                "question_id": qa_id,
                "question": str(q_text),
                "answer": gold,
                "oracle_context": oracle_context,
                "memory_ability": memory_ability,
                "turn_count": turn_count,
                "source_row_id": raw.get("id"),
            }
        )
    return result


def _split_trajectory_turns(trajectory: str) -> list[str]:
    """Split a trajectory string into individual turn strings.

    Each non-blank line is treated as one turn. This matches the MemoryAgentBench
    format where each line is ``Turn N ROLE: text``. Blank lines are dropped.
    Returns at least one element (the whole trajectory) when the trajectory
    is a single unstructured block.
    """
    if not trajectory:
        return []
    lines = [line.strip() for line in trajectory.splitlines() if line.strip()]
    return lines if lines else [trajectory]


def _split_to_ability(split_name: str) -> str:
    """Normalise HF split name to a snake_case memory ability tag."""
    return split_name.lower()


def _format_context(context: str) -> str:
    """Wrap the raw trajectory string in a minimal header for readability."""
    if not context:
        return ""
    return f"## Trajectory\n\n{context}"


def _stable_id(row: dict[str, Any]) -> str:
    """Deterministic id when no upstream id exists — hash the row content."""
    seed = json.dumps(row, sort_keys=True, default=str)
    return "mab-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Embedded smoke rows (offline dev / CI fallback)
# Three real-shape rows, one per memory ability exercised in tests.
# ---------------------------------------------------------------------------

_EMBEDDED_SMOKE_ROWS: list[dict[str, Any]] = [
    # --- 1. Accurate Retrieval ---
    {
        "question_id": "smoke-mab-001",
        "memory_ability": "accurate_retrieval",
        "question": "What flight time did the user mention for their trip to Berlin?",
        "answer": "9:45 PM",
        "oracle_context": (
            "## Trajectory\n\n"
            "Turn 1 USER: I just booked flights to Berlin. My flight lands at 9:45 PM on Friday.\n"
            "Turn 2 ASSISTANT: Got it, noted.\n"
            "Turn 3 USER: Can you remind me of my Berlin arrival time before I leave?\n"
            "Turn 4 ASSISTANT: Sure, your Berlin flight lands at 9:45 PM."
        ),
        "turn_count": 4,
        "source_row_id": None,
    },
    # --- 2. Long-Range Understanding ---
    {
        "question_id": "smoke-mab-002",
        "memory_ability": "long_range_understanding",
        "question": "Across all the project updates, what was the recurring blocker?",
        "answer": "API rate limits from the third-party provider",
        "oracle_context": (
            "## Trajectory\n\n"
            "Turn 1 USER: Sprint 1 done. We hit API rate limits again from the provider.\n"
            "Turn 2 ASSISTANT: Understood, I'll note the rate limit issue.\n"
            "Turn 3 USER: Sprint 2 report: good progress but rate limits slowed us again.\n"
            "Turn 4 ASSISTANT: That's the second sprint in a row with that blocker.\n"
            "Turn 5 USER: Sprint 3 wrap-up: rate limits from the third-party provider are "
            "still the main bottleneck.\n"
            "Turn 6 ASSISTANT: Noted. It's been a recurring theme across all three sprints."
        ),
        "turn_count": 6,
        "source_row_id": None,
    },
    # --- 3. Conflict Resolution ---
    # Two contradictory facts at different turns; gold answer = most recent.
    {
        "question_id": "smoke-mab-003",
        "memory_ability": "conflict_resolution",
        "question": "What is Project Phoenix's launch date?",
        "answer": "2026-09-15",
        "oracle_context": (
            "## Trajectory\n\n"
            "Turn 1 USER: Project Phoenix is scheduled to launch on 2026-06-01.\n"
            "Turn 2 ASSISTANT: Got it, launch date 2026-06-01 noted.\n"
            "Turn 3 USER: Update: Project Phoenix launch slipped to 2026-09-15 "
            "due to integration issues.\n"
            "Turn 4 ASSISTANT: Updated. New launch date is 2026-09-15."
        ),
        "turn_count": 4,
        "source_row_id": None,
    },
]
