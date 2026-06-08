"""Dataset contract — every benchmark loader implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ...types import BrainQuestion, Question, Track


class Dataset(ABC):
    """Abstract base for benchmark datasets.

    Subclasses must define `name` and implement `load()`. `load()` is
    expected to download from upstream (HuggingFace, GitHub, etc.) and
    cache locally. Re-runs use the cache.

    Subclasses should set:
        name           — short identifier used in the registry
        description    — one-line human description shown in the UI
        total_questions— total Q count in the canonical split; None if unbounded
        supported_tracks— which tracks this dataset supports (AGENT_ORACLE and/or
                         AGENT_E2E). Determines which radio options the UI enables.
    """

    name: str
    description: str = ""
    total_questions: int | None = None
    supported_tracks: frozenset[Track] = frozenset(
        {Track.AGENT_ORACLE}
    )  # every dataset can serve Track 2 via load(); Track 3 opts in via load_brain_questions().

    @abstractmethod
    def load(self, limit: int | None = None) -> Iterable[Question]:
        """Return an iterable of questions. If `limit` is set, return at
        most that many (deterministic — same questions every call)."""

    def load_brain_questions(self, limit: int | None = None) -> Iterable[BrainQuestion]:
        """Track 3 variant — yields BrainQuestion (with per-question corpus + gold doc paths).

        Default implementation raises NotImplementedError so the runner can
        report a clear 'this dataset doesn't support Track 3' message.
        Override per-dataset.
        """
        raise NotImplementedError(
            f"{self.name} does not yet support Track 3 (needs a BrainQuestion converter)."
        )
