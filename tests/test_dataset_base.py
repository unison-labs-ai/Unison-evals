"""Tests for the Dataset ABC default load_brain_questions behaviour."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from unison_evals.memory_evals.datasets.base import Dataset
from unison_evals.types import Question


class _FakeDataset(Dataset):
    """Minimal concrete subclass that does NOT override load_brain_questions."""

    name = "fake"

    def load(self, limit: int | None = None) -> Iterable[Question]:
        return iter([])


def test_default_load_brain_questions_raises_not_implemented() -> None:
    ds = _FakeDataset()
    with pytest.raises(NotImplementedError) as exc_info:
        ds.load_brain_questions()
    assert "fake" in str(exc_info.value)
    assert "Track 1" in str(exc_info.value) or "BrainQuestion" in str(exc_info.value)


def test_default_load_brain_questions_message_is_helpful() -> None:
    ds = _FakeDataset()
    with pytest.raises(NotImplementedError) as exc_info:
        ds.load_brain_questions(limit=1)
    msg = str(exc_info.value)
    assert "fake" in msg, "Error message should include the dataset name"
