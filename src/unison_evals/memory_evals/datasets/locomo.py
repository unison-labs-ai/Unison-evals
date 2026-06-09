"""LOCOMO — Long Conversational Memory benchmark.

Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
(Maharana et al., ACL 2024). Source: github.com/snap-research/locomo
(`data/locomo10.json`). License: see upstream repo (research use).

The benchmark Mem0, Zep, MemMachine, and Memori all report on — so it's the
head-to-head surface for competitor comparison. 10 very-long multi-session
dialogues between two speakers; each carries `qa` pairs over the conversation,
categorized: single-hop, multi-hop, temporal, open-domain, adversarial.

Mapped onto the same Question / BrainQuestion shape as every other dataset, so
it runs through the existing CLI, runners, adapter, and judge unchanged. Each
qa pair becomes one question whose corpus is the conversation's sessions (one
Document per session); gold doc paths are the sessions holding the evidence
dialog-ids.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from ...types import BrainQuestion, Document, Question, Track
from ._shared import (
    format_oracle,
    format_session,
    maybe_stratify,
    with_question_date,
)
from .base import Dataset

LOCOMO_URL = os.environ.get(
    "LOCOMO_URL",
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
)

# LOCOMO numeric category → readable question_type (the key the shared sampler
# stratifies / splits on, matching how LongMemEval uses question_type).
# Verified against locomo10.json counts (1:282, 2:321, 3:96, 4:841, 5:446) which
# match the published distribution single-hop/multi-hop/temporal/world-knowledge/
# adversarial. Category 5 (adversarial) = unanswerable (444/446 null answers).
CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}
# Adversarial (cat 5) has no gradeable ground truth; the field convention
# (Mem0/Zep/papers) is to EXCLUDE it from scoring. Excluded by default; set
# LOCOMO_INCLUDE_ADVERSARIAL=1 to include (e.g. to measure abstention).
ADVERSARIAL_CATEGORY = 5


def _cache_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "unison-evals"
    base.mkdir(parents=True, exist_ok=True)
    return base / "locomo10.json"


def _iter_sessions(conversation: dict[str, Any]) -> list[tuple[str, list, str | None]]:
    """Yield (session_id, turns, date) in session order. LOCOMO stores sessions
    as ``session_1``, ``session_2``, … with a sibling ``session_N_date_time``."""
    out: list[tuple[str, list, str | None]] = []
    idx = 1
    while f"session_{idx}" in conversation:
        turns = conversation.get(f"session_{idx}") or []
        date = conversation.get(f"session_{idx}_date_time")
        out.append((f"session_{idx}", turns, date))
        idx += 1
    return out


class LocomoDataset(Dataset):
    name = "locomo"
    description = (
        "LOCOMO — ~2000 QA over 10 very-long multi-session dialogues "
        "(single-hop, multi-hop, temporal, open-domain, adversarial). The "
        "Mem0/Zep/MemMachine head-to-head surface. ACL 2024."
    )
    total_questions = None  # ~2000, varies by category filtering
    supported_tracks = frozenset({Track.AGENT_ORACLE, Track.AGENT_E2E})

    def load(self, limit: int | None = None) -> Iterable[Question]:
        rows = maybe_stratify(self._load_raw_rows(), limit)
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_question(row)

    def load_brain_questions(self, limit: int | None = None) -> Iterable[BrainQuestion]:
        rows = maybe_stratify(self._load_raw_rows(), limit)
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield self._row_to_brain_question(row)

    # --- shaping -------------------------------------------------------------

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> Question:
        return Question(
            id=row["question_id"],
            question=with_question_date(row.get("date"), row["question"]),
            expected_answer=row["answer"],
            oracle_context=format_oracle(row["sessions"], row["dates"]),
            metadata={"question_type": row["question_type"], "evidence": row["evidence"]},
        )

    @staticmethod
    def _row_to_brain_question(row: dict[str, Any]) -> BrainQuestion:
        corpus: list[Document] = []
        path_by_session: dict[str, str] = {}
        for idx, (sid, turns, date) in enumerate(
            zip(row["session_ids"], row["sessions"], row["dates"], strict=False)
        ):
            path = f"/sessions/{sid}.md"
            corpus.append(
                Document(
                    path=path,
                    body=format_session(turns, date, idx),
                    metadata={"session_id": sid, "date": date},
                )
            )
            path_by_session[sid] = path

        # Gold = the sessions that contain an evidence dialog-id.
        gold_paths: set[str] = set()
        for sid in row.get("evidence_session_ids", []):
            if sid in path_by_session:
                gold_paths.add(path_by_session[sid])

        return BrainQuestion(
            id=row["question_id"],
            query=with_question_date(row.get("date"), row["question"]),
            corpus=corpus,
            gold_doc_paths=gold_paths,
            # All qa over the same conversation share its corpus → one seed per
            # conversation, reused across its ~200 questions (not re-seeded each).
            corpus_key=row["corpus_key"],
            metadata={
                "question_type": row["question_type"],
                "expected_answer": row["answer"],
                "evidence": row["evidence"],
            },
        )

    # --- loading -------------------------------------------------------------

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        """Download + cache locomo10.json, then flatten to one normalized row
        per qa pair. Falls back to an embedded smoke sample when offline."""
        try:
            samples = self._fetch()
        except Exception as e:
            logger.warning("Failed to load LOCOMO ({}); using embedded smoke set.", e)
            samples = _EMBEDDED_SMOKE
        return list(self._flatten(samples))

    def _fetch(self) -> list[dict[str, Any]]:
        cache = _cache_path()
        if not cache.exists():
            logger.info("Downloading LOCOMO from {} → {}", LOCOMO_URL, cache)
            with urllib.request.urlopen(LOCOMO_URL, timeout=60) as resp:
                cache.write_bytes(resp.read())
        data = json.loads(cache.read_text())
        return data if isinstance(data, list) else data.get("data", [])

    @staticmethod
    def _flatten(samples: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for s_idx, sample in enumerate(samples):
            conv = sample.get("conversation") or {}
            sessions_meta = _iter_sessions(conv)
            session_ids = [sid for sid, _, _ in sessions_meta]
            sessions = [turns for _, turns, _ in sessions_meta]
            dates = [date for _, _, date in sessions_meta]
            # dialog-id → session id, so evidence ("D3:5") maps to its session.
            dia_to_session: dict[str, str] = {}
            for sid, turns, _ in sessions_meta:
                for turn in turns or []:
                    did = turn.get("dia_id") if isinstance(turn, dict) else None
                    if did:
                        dia_to_session[str(did)] = sid
            sample_key = str(sample.get("sample_id") or s_idx)
            include_adv = bool(os.environ.get("LOCOMO_INCLUDE_ADVERSARIAL"))
            for q_idx, qa in enumerate(sample.get("qa") or []):
                cat = qa.get("category")
                # Exclude adversarial (cat 5) by default — ungradeable, and the
                # field convention excludes it. q_idx stays stable (gaps are fine).
                if cat == ADVERSARIAL_CATEGORY and not include_adv:
                    continue
                answer = qa.get("answer")
                if answer is None:
                    answer = qa.get("adversarial_answer")  # category 5
                evidence = qa.get("evidence") or []
                ev_session_ids = {
                    dia_to_session[str(e)] for e in evidence if str(e) in dia_to_session
                }
                yield {
                    "question_id": f"locomo-{sample_key}-{q_idx}",
                    "corpus_key": f"locomo-conv-{sample_key}",
                    "question_type": CATEGORY_NAMES.get(cat, str(cat)),
                    "question": str(qa.get("question") or ""),
                    "answer": str(answer if answer is not None else ""),
                    "evidence": evidence,
                    "evidence_session_ids": sorted(ev_session_ids),
                    "session_ids": session_ids,
                    "sessions": sessions,
                    "dates": dates,
                    "date": dates[-1] if dates else None,  # latest session = "now"
                }


# Embedded smoke sample (offline dev / CI without network) — real LOCOMO shape.
_EMBEDDED_SMOKE: list[dict[str, Any]] = [
    {
        "sample_id": "smoke",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I adopted a golden retriever named Toby.",
                },
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "That's wonderful!"},
            ],
            "session_2_date_time": "10:10 am on 20 June, 2023",
            "session_2": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D2:1",
                    "text": "Toby just turned 1 year old today.",
                },
                {"speaker": "Melanie", "dia_id": "D2:2", "text": "Happy birthday, Toby!"},
            ],
        },
        "qa": [
            {
                "question": "What is the name of Caroline's dog?",
                "answer": "Toby",
                "evidence": ["D1:1"],
                "category": 4,
            },
        ],
    },
]
