"""FRAMES dataset loader — unit tests.

All tests run offline (no HuggingFace network calls). Network paths are
exercised via monkeypatch that forces HF to raise, confirming the embedded
smoke fallback activates correctly.
"""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.datasets.frames import (
    _EMBEDDED_SMOKE_ROWS,
    FramesDataset,
    _stable_id,
)
from unison_evals.types import Question


class TestEmbeddedSmokeRows:
    def test_all_rows_parse_to_valid_questions(self) -> None:
        assert len(_EMBEDDED_SMOKE_ROWS) >= 3, "need at least 3 smoke rows"
        for raw in _EMBEDDED_SMOKE_ROWS:
            q = FramesDataset._row_to_question(raw)
            assert isinstance(q, Question)
            assert q.id, "id must be non-empty"
            assert q.question, "question text must be non-empty"
            assert q.expected_answer, "expected_answer must be non-empty"

    def test_oracle_context_is_none(self) -> None:
        """FRAMES Track 2 uses parametric knowledge — no context injected."""
        for raw in _EMBEDDED_SMOKE_ROWS:
            q = FramesDataset._row_to_question(raw)
            assert q.oracle_context is None

    def test_wiki_links_survive_roundtrip(self) -> None:
        raw = _EMBEDDED_SMOKE_ROWS[0]
        q = FramesDataset._row_to_question(raw)
        assert isinstance(q.metadata["wiki_links"], list)
        assert len(q.metadata["wiki_links"]) >= 1
        for link in q.metadata["wiki_links"]:
            assert link.startswith("https://en.wikipedia.org/")

    def test_metadata_fields_preserved(self) -> None:
        raw = _EMBEDDED_SMOKE_ROWS[1]
        q = FramesDataset._row_to_question(raw)
        assert q.metadata["reasoning_types"] == "Multi-hop"
        assert q.metadata["topic"] == "Literature"
        assert q.metadata["original_id"] == "frames-smoke-002"


class TestLoadLimit:
    def test_load_limit_returns_exact_count(self, monkeypatch) -> None:
        """Even if the loader would return many rows, limit is respected."""
        import unison_evals.memory_evals.datasets.frames as mod

        monkeypatch.setattr(
            mod.FramesDataset,
            "_load_raw_rows",
            lambda self: _EMBEDDED_SMOKE_ROWS,
        )

        ds = FramesDataset()
        questions = list(ds.load(limit=2))
        assert len(questions) == 2

    def test_load_no_limit_returns_all_smoke_rows(self, monkeypatch) -> None:
        import unison_evals.memory_evals.datasets.frames as mod

        monkeypatch.setattr(
            mod.FramesDataset,
            "_load_raw_rows",
            lambda self: _EMBEDDED_SMOKE_ROWS,
        )

        ds = FramesDataset()
        questions = list(ds.load())
        assert len(questions) == len(_EMBEDDED_SMOKE_ROWS)


class TestNetworkFallback:
    def test_hf_failure_falls_back_to_embedded(self, monkeypatch) -> None:
        """Simulate network failure — loader must silently fall back."""
        import sys

        # Patch the datasets module so the import inside _load_raw_rows raises
        fake_datasets = type(sys)("datasets")

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated network failure")

        fake_datasets.load_dataset = _boom
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        ds = FramesDataset()
        questions = list(ds.load(limit=2))
        assert len(questions) == 2
        assert questions[0].id == "frames-smoke-001"
        assert questions[1].id == "frames-smoke-002"

    def test_hf_failure_questions_are_valid(self, monkeypatch) -> None:
        """All embedded smoke rows are valid Questions after fallback."""
        import sys

        fake_datasets = type(sys)("datasets")

        def _boom(*_a, **_kw):
            raise OSError("offline")

        fake_datasets.load_dataset = _boom
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        ds = FramesDataset()
        for q in ds.load():
            assert isinstance(q, Question)
            assert q.question
            assert q.expected_answer


class TestStableId:
    def test_stable_id_is_deterministic(self) -> None:
        row = {"Prompt": "Who wrote Hamlet?", "Answer": "Shakespeare"}
        id1 = _stable_id(row)
        id2 = _stable_id(row)
        assert id1 == id2

    def test_stable_id_prefix(self) -> None:
        row = {"Prompt": "Test question", "Answer": "Test answer"}
        assert _stable_id(row).startswith("frames-")

    def test_stable_id_differs_for_different_rows(self) -> None:
        row_a = {"Prompt": "Question A", "Answer": "Answer A"}
        row_b = {"Prompt": "Question B", "Answer": "Answer B"}
        assert _stable_id(row_a) != _stable_id(row_b)

    def test_row_without_id_gets_stable_id(self) -> None:
        row = {"Prompt": "No id here", "Answer": "Some answer"}
        q = FramesDataset._row_to_question(row)
        assert q.id.startswith("frames-")


class TestTrack1Deferred:
    """FRAMES Track 1 is deferred to v0.2 — load_brain_questions must raise."""

    def test_load_brain_questions_raises_not_implemented(self) -> None:
        ds = FramesDataset()
        with pytest.raises(NotImplementedError) as exc_info:
            ds.load_brain_questions()
        msg = str(exc_info.value)
        assert "FRAMES" in msg
        assert "v0.2" in msg or "deferred" in msg.lower()
        assert "Wikipedia" in msg or "wiki" in msg.lower()

    def test_load_brain_questions_with_limit_still_raises(self) -> None:
        ds = FramesDataset()
        with pytest.raises(NotImplementedError):
            ds.load_brain_questions(limit=1)


class TestDefensiveParsing:
    def test_lowercase_prompt_fallback(self) -> None:
        row = {"prompt": "lowercase prompt key", "Answer": "ans", "question_id": "x1"}
        q = FramesDataset._row_to_question(row)
        assert q.question == "lowercase prompt key"

    def test_question_fallback(self) -> None:
        row = {"question": "question key", "answer": "ans", "question_id": "x2"}
        q = FramesDataset._row_to_question(row)
        assert q.question == "question key"

    def test_answer_lowercase_fallback(self) -> None:
        row = {"Prompt": "Q?", "answer": "lowercase answer", "question_id": "x3"}
        q = FramesDataset._row_to_question(row)
        assert q.expected_answer == "lowercase answer"

    def test_gold_answer_fallback(self) -> None:
        row = {"Prompt": "Q?", "gold_answer": "gold fallback", "question_id": "x4"}
        q = FramesDataset._row_to_question(row)
        assert q.expected_answer == "gold fallback"

    def test_wiki_links_as_comma_string(self) -> None:
        """Some HF schema versions return wiki_links as a CSV string."""
        row = {
            "Prompt": "Q?",
            "Answer": "A",
            "question_id": "x5",
            "wiki_links": "https://en.wikipedia.org/wiki/Foo,https://en.wikipedia.org/wiki/Bar",
        }
        q = FramesDataset._row_to_question(row)
        assert q.metadata["wiki_links"] == [
            "https://en.wikipedia.org/wiki/Foo",
            "https://en.wikipedia.org/wiki/Bar",
        ]

    def test_missing_wiki_links_defaults_to_empty_list(self) -> None:
        row = {"Prompt": "Q?", "Answer": "A", "question_id": "x6"}
        q = FramesDataset._row_to_question(row)
        assert q.metadata["wiki_links"] == []
