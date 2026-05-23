"""MS MARCO passage ranking dataset — Track 4 (scale) only.

Reference:
  Bajaj et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"
  arXiv:1611.09268 (NIPS 2016 Workshop)

HuggingFace: microsoft/ms_marco, config "v1.1", split "validation"

Corpus size: 8.8M passages.
Dev queries with judged qrels: 6,980.

Why Track 4 only: MS MARCO is too large to reset + ingest per question.
The correct lifecycle is:
  1. (One-time) `scripts/load_corpus_msmarco.sh` — bulk-ingest passages into
     the chosen brain adapter, writing each passage at path
     `/msmarco/passages/<passage_id>.md`.
  2. (Per run) `load_scale_questions()` → `ScaleRetrievalRunner.run()`

`load()` raises NotImplementedError to make it clear that Track 1/2 are not
supported for this dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loguru import logger

from ...config import get_settings
from ...types import Question, ScaleQuestion, Track
from .base import Dataset

# HuggingFace dataset id + config.
HF_DATASET = "microsoft/ms_marco"
HF_CONFIG = "v1.1"
HF_SPLIT = "validation"


class MsMarcoDataset(Dataset):
    name = "msmarco"
    description = (
        "MS MARCO passage ranking — 6,980 dev queries with judged qrels over 8.8M passages. "
        "Track 4 (scale) only. Corpus must be pre-loaded via scripts/load_corpus_msmarco.sh."
    )
    total_questions = 6_980
    supported_tracks = frozenset({Track.SCALE})

    def __init__(self) -> None:
        self.settings = get_settings()

    def load(self, limit: int | None = None) -> Iterable[Question]:
        raise NotImplementedError(
            "MS MARCO is a scale benchmark; use load_scale_questions() with Track 4. "
            "Run `scripts/load_corpus_msmarco.sh` first to populate the brain, "
            "then: dataset.load_scale_questions(limit=N)"
        )

    def load_scale_questions(self, limit: int | None = None) -> Iterable[ScaleQuestion]:
        """Load MS MARCO dev queries as ScaleQuestion objects.

        Gold doc paths follow the corpus loader scheme:
            /msmarco/passages/<passage_id>.md

        This matches the paths written by `scripts/load_corpus_msmarco.sh`
        and `scripts/_load_corpus.py`.

        Falls back to a tiny embedded smoke set (5 synthetic queries) if the
        HuggingFace download is unavailable — so offline tests always pass.
        """
        rows = self._load_raw_rows()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                return
            sq = self._row_to_scale_question(row)
            if sq is not None:
                yield sq

    def _load_raw_rows(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("`datasets` library not installed. Run `uv sync`.") from e

        try:
            ds = load_dataset(
                HF_DATASET,
                HF_CONFIG,
                split=HF_SPLIT,
                cache_dir=str(self.settings.cache_dir / "hf"),
            )
            return list(ds)
        except Exception as e:
            logger.warning(
                "Failed to load MS MARCO from HuggingFace, falling back to "
                "embedded smoke set. error={}",
                e,
            )
            return _EMBEDDED_SMOKE_ROWS

    @staticmethod
    def _row_to_scale_question(row: dict[str, Any]) -> ScaleQuestion | None:
        """Map a MS MARCO v1.1 row to a ScaleQuestion.

        MS MARCO v1.1 validation schema:
          - query_id: int
          - query: str
          - query_type: str
          - passages: list[{passage_text, is_selected, url, passage_id}]
          - answers: list[str]

        Gold passages are those where is_selected == 1.
        Path scheme: /msmarco/passages/<passage_id>.md — must match the
        corpus loader script exactly.
        """
        qid = str(row.get("query_id", ""))
        query = str(row.get("query", ""))
        if not query:
            return None

        passages = row.get("passages", {})
        # HF stores passages as a dict of lists (columnar format).
        if isinstance(passages, dict):
            passage_ids = passages.get("passage_id", [])
            is_selected = passages.get("is_selected", [])
        elif isinstance(passages, list):
            passage_ids = [p.get("passage_id", "") for p in passages]
            is_selected = [p.get("is_selected", 0) for p in passages]
        else:
            passage_ids = []
            is_selected = []

        gold_doc_paths: set[str] = set()
        for pid, sel in zip(passage_ids, is_selected, strict=False):
            if sel:
                gold_doc_paths.add(f"/msmarco/passages/{pid}.md")

        if not gold_doc_paths:
            return None

        return ScaleQuestion(
            id=f"msmarco-{qid}",
            query=query,
            gold_doc_paths=gold_doc_paths,
            metadata={
                "query_id": qid,
                "query_type": row.get("query_type", ""),
            },
        )


# ---------------------------------------------------------------------------
# Embedded smoke set — 5 synthetic queries so offline tests always pass.
#
# Path scheme mirrors what load_corpus_msmarco.sh writes:
#   /msmarco/passages/<passage_id>.md
# Passage IDs are fictional (smoke-p-XXXX) — they only need to match
# consistently within this file.
# ---------------------------------------------------------------------------

_EMBEDDED_SMOKE_ROWS: list[dict[str, Any]] = [
    {
        "query_id": 1001,
        "query": "what is the capital of France",
        "query_type": "description",
        "passages": {
            "passage_id": ["smoke-p-1001a", "smoke-p-1001b", "smoke-p-1001c"],
            "is_selected": [1, 0, 0],
            "passage_text": [
                "Paris is the capital and most populous city of France.",
                "Lyon is a major city in eastern France.",
                "The French Republic was founded in 1792.",
            ],
            "url": ["", "", ""],
        },
        "answers": ["Paris"],
    },
    {
        "query_id": 1002,
        "query": "how does photosynthesis work",
        "query_type": "description",
        "passages": {
            "passage_id": ["smoke-p-1002a", "smoke-p-1002b"],
            "is_selected": [1, 0],
            "passage_text": [
                "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide "
                "to produce oxygen and energy in the form of sugar.",
                "Chlorophyll is the green pigment in plants responsible for absorbing light energy.",
            ],
            "url": ["", ""],
        },
        "answers": ["Plants use sunlight, water and CO2 to produce oxygen and sugar."],
    },
    {
        "query_id": 1003,
        "query": "when was the Eiffel Tower built",
        "query_type": "numeric",
        "passages": {
            "passage_id": ["smoke-p-1003a", "smoke-p-1003b"],
            "is_selected": [1, 0],
            "passage_text": [
                "The Eiffel Tower was built in 1889 as the entrance arch to the 1889 World's Fair.",
                "Gustave Eiffel designed the tower that bears his name.",
            ],
            "url": ["", ""],
        },
        "answers": ["1889"],
    },
    {
        "query_id": 1004,
        "query": "what causes thunder",
        "query_type": "description",
        "passages": {
            "passage_id": ["smoke-p-1004a", "smoke-p-1004b", "smoke-p-1004c"],
            "is_selected": [0, 1, 0],
            "passage_text": [
                "Lightning occurs during thunderstorms when electrical charges build up in clouds.",
                "Thunder is the sound caused by the rapid expansion of air heated by a lightning bolt.",
                "The speed of sound is approximately 343 metres per second.",
            ],
            "url": ["", "", ""],
        },
        "answers": ["Rapid expansion of air heated by lightning."],
    },
    {
        "query_id": 1005,
        "query": "how many bones are in the human body",
        "query_type": "numeric",
        "passages": {
            "passage_id": ["smoke-p-1005a", "smoke-p-1005b"],
            "is_selected": [1, 0],
            "passage_text": [
                "The human body has 206 bones in adulthood. Babies are born with approximately 270 bones.",
                "The femur is the longest and strongest bone in the human body.",
            ],
            "url": ["", ""],
        },
        "answers": ["206"],
    },
]
