"""MuSiQue dataset loader — unit tests."""

from __future__ import annotations

from unison_evals.memory_evals.datasets.musique import (
    _EMBEDDED_SMOKE_ROWS,
    MuSiQueDataset,
    _format_paragraphs,
    _slug,
)
from unison_evals.types import BrainQuestion, Question

# ---------------------------------------------------------------------------
# Smoke-row parsing
# ---------------------------------------------------------------------------


def test_embedded_smoke_rows_parse_to_valid_questions() -> None:
    assert len(_EMBEDDED_SMOKE_ROWS) == 3
    for raw in _EMBEDDED_SMOKE_ROWS:
        q = MuSiQueDataset._row_to_question(raw)
        assert isinstance(q, Question)
        assert q.id, "id must be non-empty"
        assert q.question, "question must be non-empty"
        assert q.expected_answer, "expected_answer must be non-empty"
        assert q.oracle_context, "oracle_context must be non-empty"


def test_oracle_context_contains_all_paragraph_titles() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    assert q.oracle_context is not None
    for para in raw["paragraphs"]:
        assert para["title"] in q.oracle_context


def test_oracle_context_uses_markdown_headers() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    assert q.oracle_context is not None
    assert "### [" in q.oracle_context


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_metadata_preserves_paragraphs() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    assert q.metadata["paragraphs"] == raw["paragraphs"]


def test_metadata_preserves_gold_paragraph_indexes() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    gold = q.metadata["gold_paragraph_indexes"]
    expected_gold = [p["idx"] for p in raw["paragraphs"] if p["is_supporting"]]
    assert gold == expected_gold


def test_metadata_preserves_answerable_flag() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    assert q.metadata["answerable"] is True


def test_metadata_preserves_question_decomposition() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MuSiQueDataset._row_to_question(raw)
    assert q.metadata["question_decomposition"] == raw["question_decomposition"]


# ---------------------------------------------------------------------------
# load() with limit
# ---------------------------------------------------------------------------


def test_load_limit_returns_exact_count(monkeypatch) -> None:
    import unison_evals.memory_evals.datasets.musique as mod

    monkeypatch.setattr(
        mod,
        "load_dataset",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("offline")),
        raising=False,
    )

    ds = MuSiQueDataset()
    questions = list(ds.load(limit=2))
    assert len(questions) == 2


def test_load_limit_is_deterministic(monkeypatch) -> None:
    import unison_evals.memory_evals.datasets.musique as mod

    monkeypatch.setattr(
        mod,
        "load_dataset",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("offline")),
        raising=False,
    )

    ds = MuSiQueDataset()
    first = list(ds.load(limit=2))
    second = list(ds.load(limit=2))
    assert [q.id for q in first] == [q.id for q in second]


# ---------------------------------------------------------------------------
# Network fallback
# ---------------------------------------------------------------------------


def test_network_failure_falls_back_to_embedded(monkeypatch) -> None:
    import sys
    import types

    def boom(*_a, **_kw):
        raise RuntimeError("simulated network failure")

    # Patch at the datasets package level so the lazy import inside
    # _load_raw_rows picks up the mock regardless of import caching.
    fake_datasets_mod = types.ModuleType("datasets")
    fake_datasets_mod.load_dataset = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)

    ds = MuSiQueDataset()
    questions = list(ds.load(limit=2))
    assert len(questions) == 2
    assert questions[0].id == "smoke-musique-001"
    assert questions[1].id == "smoke-musique-002"


# ---------------------------------------------------------------------------
# _format_paragraphs edge cases
# ---------------------------------------------------------------------------


def test_format_paragraphs_empty_list() -> None:
    result = _format_paragraphs([])
    assert result == ""


def test_format_paragraphs_missing_title() -> None:
    paras = [{"idx": 0, "paragraph_text": "Some text.", "is_supporting": True}]
    result = _format_paragraphs(paras)
    assert "### [0]" in result
    assert "Some text." in result


def test_format_paragraphs_missing_paragraph_text() -> None:
    paras = [{"idx": 0, "title": "My Title", "is_supporting": False}]
    result = _format_paragraphs(paras)
    assert "My Title" in result
    assert "### [0]" in result


def test_format_paragraphs_both_fields_missing() -> None:
    paras = [{"idx": 5, "is_supporting": False}]
    result = _format_paragraphs(paras)
    assert "### [5]" in result


def test_format_paragraphs_multiple_paragraphs_all_present() -> None:
    paras = [
        {"idx": 0, "title": "Alpha", "paragraph_text": "Text A.", "is_supporting": True},
        {"idx": 1, "title": "Beta", "paragraph_text": "Text B.", "is_supporting": False},
    ]
    result = _format_paragraphs(paras)
    assert "Alpha" in result
    assert "Beta" in result
    assert "Text A." in result
    assert "Text B." in result


# ---------------------------------------------------------------------------
# Track 1 — load_brain_questions
# ---------------------------------------------------------------------------


def test_musique_load_brain_questions_returns_brain_question_objects(monkeypatch) -> None:
    import unison_evals.memory_evals.datasets.musique as mod

    monkeypatch.setattr(
        mod,
        "load_dataset",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("offline")),
        raising=False,
    )
    ds = MuSiQueDataset()
    bqs = list(ds.load_brain_questions(limit=2))
    assert len(bqs) == 2
    for bq in bqs:
        assert isinstance(bq, BrainQuestion)


def test_musique_brain_question_corpus_non_empty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MuSiQueDataset._row_to_brain_question(raw)
        assert bq.corpus, f"BrainQuestion {bq.id} has empty corpus"


def test_musique_brain_question_gold_paths_non_empty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MuSiQueDataset._row_to_brain_question(raw)
        assert bq.gold_doc_paths, f"BrainQuestion {bq.id} has empty gold_doc_paths"


def test_musique_brain_question_gold_paths_subset_of_corpus() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MuSiQueDataset._row_to_brain_question(raw)
        corpus_paths = {doc.path for doc in bq.corpus}
        dangling = bq.gold_doc_paths - corpus_paths
        assert not dangling, (
            f"BrainQuestion {bq.id} has dangling gold paths not in corpus: {dangling}"
        )


def test_musique_brain_question_gold_paths_only_for_supporting_paragraphs() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    bq = MuSiQueDataset._row_to_brain_question(raw)
    gold_paths = bq.gold_doc_paths
    non_gold_titles = {p["title"] for p in raw["paragraphs"] if not p["is_supporting"]}
    for doc in bq.corpus:
        if any(_slug(t) in doc.path for t in non_gold_titles):
            assert doc.path not in gold_paths, (
                f"Non-supporting paragraph {doc.path} should not be in gold paths"
            )


def test_musique_brain_question_doc_paths_use_paragraph_scheme() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    bq = MuSiQueDataset._row_to_brain_question(raw)
    for doc in bq.corpus:
        assert doc.path.startswith("/paragraphs/"), f"Unexpected path: {doc.path}"
        assert doc.path.endswith(".md"), f"Path should end in .md: {doc.path}"


def test_musique_brain_question_corpus_body_non_empty() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    bq = MuSiQueDataset._row_to_brain_question(raw)
    for doc in bq.corpus:
        assert doc.body.strip(), f"Document {doc.path} has empty body"


# ---------------------------------------------------------------------------
# _slug helper
# ---------------------------------------------------------------------------


def test_slug_lowercases() -> None:
    assert _slug("Leonardo da Vinci") == "leonardo-da-vinci"


def test_slug_replaces_special_chars() -> None:
    assert _slug("Hello, World!") == "hello-world"


def test_slug_truncates_long_titles() -> None:
    long_title = "A" * 100
    assert len(_slug(long_title)) <= 48


def test_slug_empty_string_returns_untitled() -> None:
    assert _slug("") == "untitled"
