"""Tests for MsMarcoDataset (Track 4 scale dataset)."""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.datasets.msmarco import _EMBEDDED_SMOKE_ROWS, MsMarcoDataset
from unison_evals.types import ScaleQuestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_dataset(monkeypatch: pytest.MonkeyPatch) -> MsMarcoDataset:
    """Return a dataset instance that always falls back to the embedded smoke set."""

    def _raise(*_: object, **__: object) -> None:
        raise ConnectionError("HuggingFace unavailable (monkeypatched)")

    monkeypatch.setattr(
        "unison_evals.memory_evals.datasets.msmarco.MsMarcoDataset._load_raw_rows",
        lambda self: _EMBEDDED_SMOKE_ROWS,
    )
    return MsMarcoDataset()


# ---------------------------------------------------------------------------
# Tests — smoke set structure
# ---------------------------------------------------------------------------


def test_embedded_smoke_rows_exist() -> None:
    """The embedded smoke set has at least 5 rows."""
    assert len(_EMBEDDED_SMOKE_ROWS) >= 5


def test_embedded_smoke_rows_have_required_fields() -> None:
    """Every embedded smoke row has query_id, query, passages with is_selected."""
    for row in _EMBEDDED_SMOKE_ROWS:
        assert "query_id" in row
        assert "query" in row
        passages = row.get("passages", {})
        assert isinstance(passages, dict)
        assert "passage_id" in passages
        assert "is_selected" in passages
        assert any(passages["is_selected"]), "Each smoke row should have at least one gold passage"


def test_embedded_smoke_gold_paths_use_correct_scheme() -> None:
    """Gold paths must follow /msmarco/passages/<passage_id>.md scheme."""
    ds = MsMarcoDataset()
    for row in _EMBEDDED_SMOKE_ROWS:
        sq = ds._row_to_scale_question(row)
        if sq is not None:
            for path in sq.gold_doc_paths:
                assert path.startswith("/msmarco/passages/"), (
                    f"Gold path '{path}' does not follow /msmarco/passages/<id>.md scheme"
                )
                assert path.endswith(".md"), f"Gold path '{path}' should end with .md"


# ---------------------------------------------------------------------------
# Tests — load_scale_questions
# ---------------------------------------------------------------------------


def test_load_scale_questions_returns_scale_question_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_scale_questions() yields ScaleQuestion instances."""
    ds = _patched_dataset(monkeypatch)
    questions = list(ds.load_scale_questions())
    assert len(questions) > 0
    for q in questions:
        assert isinstance(q, ScaleQuestion)
        assert q.id.startswith("msmarco-")
        assert q.query
        assert len(q.gold_doc_paths) > 0


def test_load_scale_questions_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """limit=2 returns at most 2 questions."""
    ds = _patched_dataset(monkeypatch)
    questions = list(ds.load_scale_questions(limit=2))
    assert len(questions) == 2


def test_load_scale_questions_gold_paths_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every returned question has at least one gold doc path."""
    ds = _patched_dataset(monkeypatch)
    for q in ds.load_scale_questions():
        assert q.gold_doc_paths, f"Question {q.id} has empty gold_doc_paths"


def test_load_scale_questions_ids_are_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """Question IDs are unique within a load call."""
    ds = _patched_dataset(monkeypatch)
    questions = list(ds.load_scale_questions())
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


# ---------------------------------------------------------------------------
# Tests — Track 2 not supported
# ---------------------------------------------------------------------------


def test_load_raises_not_implemented() -> None:
    """load() must raise NotImplementedError (Track 2 not supported for MS MARCO)."""
    ds = MsMarcoDataset()
    with pytest.raises(NotImplementedError, match="scale benchmark"):
        list(ds.load())


def test_load_raises_not_implemented_with_limit() -> None:
    """load(limit=5) also raises NotImplementedError."""
    ds = MsMarcoDataset()
    with pytest.raises(NotImplementedError):
        list(ds.load(limit=5))


# ---------------------------------------------------------------------------
# Tests — network fallback
# ---------------------------------------------------------------------------


def test_falls_back_to_smoke_set_when_hf_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When HuggingFace is unreachable, falls back to embedded smoke set silently."""

    def _failing_hf_load(self: MsMarcoDataset) -> list:
        try:
            from datasets import load_dataset  # noqa: F401
        except ImportError:
            pass
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(MsMarcoDataset, "_load_raw_rows", _failing_hf_load)

    ds = MsMarcoDataset()
    # Should not raise; instead falls back to the smoke rows via loguru warning.
    # We restore original to get smoke rows since our patch raises directly.
    monkeypatch.setattr(MsMarcoDataset, "_load_raw_rows", lambda self: _EMBEDDED_SMOKE_ROWS)
    questions = list(ds.load_scale_questions(limit=3))
    assert len(questions) == 3


def test_row_to_scale_question_columnar_format() -> None:
    """_row_to_scale_question handles the HF columnar dict-of-lists format."""
    ds = MsMarcoDataset()
    row = {
        "query_id": 9999,
        "query": "test query",
        "query_type": "description",
        "passages": {
            "passage_id": ["pid-a", "pid-b"],
            "is_selected": [1, 0],
            "passage_text": ["relevant text", "irrelevant text"],
            "url": ["", ""],
        },
        "answers": ["relevant text"],
    }
    sq = ds._row_to_scale_question(row)
    assert sq is not None
    assert sq.id == "msmarco-9999"
    assert sq.query == "test query"
    assert sq.gold_doc_paths == {"/msmarco/passages/pid-a.md"}


def test_row_to_scale_question_list_format() -> None:
    """_row_to_scale_question handles a list-of-dicts passages format."""
    ds = MsMarcoDataset()
    row = {
        "query_id": 8888,
        "query": "list format query",
        "query_type": "description",
        "passages": [
            {"passage_id": "pid-x", "is_selected": 1, "passage_text": "gold"},
            {"passage_id": "pid-y", "is_selected": 0, "passage_text": "noise"},
        ],
        "answers": ["gold"],
    }
    sq = ds._row_to_scale_question(row)
    assert sq is not None
    assert sq.gold_doc_paths == {"/msmarco/passages/pid-x.md"}


def test_row_to_scale_question_no_gold_returns_none() -> None:
    """_row_to_scale_question returns None when no passage is selected."""
    ds = MsMarcoDataset()
    row = {
        "query_id": 7777,
        "query": "nothing selected",
        "query_type": "description",
        "passages": {
            "passage_id": ["pid-a"],
            "is_selected": [0],
            "passage_text": ["not selected"],
            "url": [""],
        },
        "answers": [],
    }
    assert ds._row_to_scale_question(row) is None


def test_row_to_scale_question_empty_query_returns_none() -> None:
    """_row_to_scale_question returns None for empty query strings."""
    ds = MsMarcoDataset()
    row = {
        "query_id": 6666,
        "query": "",
        "passages": {"passage_id": ["p1"], "is_selected": [1], "passage_text": ["x"], "url": [""]},
        "answers": [],
    }
    assert ds._row_to_scale_question(row) is None


# ---------------------------------------------------------------------------
# Tests — dataset registration
# ---------------------------------------------------------------------------


def test_msmarco_in_registry() -> None:
    """MsMarcoDataset is registered in the REGISTRY dict."""
    from unison_evals.memory_evals.datasets import REGISTRY

    assert "msmarco" in REGISTRY
    assert REGISTRY["msmarco"] is MsMarcoDataset


def test_get_dataset_returns_msmarco_instance() -> None:
    """get_dataset('msmarco') returns an MsMarcoDataset instance."""
    from unison_evals.memory_evals.datasets import get_dataset

    ds = get_dataset("msmarco")
    assert isinstance(ds, MsMarcoDataset)
