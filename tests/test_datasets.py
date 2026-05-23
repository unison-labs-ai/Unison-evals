"""Dataset loaders return well-shaped Question objects."""

from __future__ import annotations

from unison_evals.memory_evals.datasets.longmemeval import _EMBEDDED_SMOKE_ROWS, LongMemEvalDataset
from unison_evals.types import BrainQuestion, Question


def test_longmemeval_embedded_smoke_parses() -> None:
    rows = _EMBEDDED_SMOKE_ROWS
    assert len(rows) == 3
    for raw in rows:
        q = LongMemEvalDataset._row_to_question(raw)
        assert isinstance(q, Question)
        assert q.id
        assert q.question
        assert q.expected_answer
        assert q.oracle_context  # smoke rows always have at least one session
        assert "## Session" in q.oracle_context


def test_longmemeval_load_with_limit_uses_embedded_when_offline(monkeypatch) -> None:
    """Force the network path to fail → loader falls back to embedded."""
    import unison_evals.memory_evals.datasets.longmemeval as mod

    def boom(*_a, **_kw):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(mod, "load_dataset", boom, raising=False)

    ds = LongMemEvalDataset()
    questions = list(ds.load(limit=2))
    assert len(questions) == 2
    assert questions[0].id == "smoke-001"
    assert questions[1].id == "smoke-002"


def test_question_metadata_passthrough() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[1]
    q = LongMemEvalDataset._row_to_question(raw)
    assert q.metadata["question_type"] == "knowledge-update"
    assert q.metadata["answer_session_ids"] == ["s2"]


# ---------------------------------------------------------------------------
# LongMemEval — Track 1 (load_brain_questions)
# ---------------------------------------------------------------------------


def test_longmemeval_load_brain_questions_returns_brain_question_objects(monkeypatch) -> None:
    ds = LongMemEvalDataset()
    # Force offline fallback so no HF download is needed. monkeypatch.setitem
    # snapshots and restores sys.modules["datasets"] cleanly, so this test can't
    # leak a fake module into later tests.
    import sys
    import types

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    bqs = list(ds.load_brain_questions(limit=2))

    assert len(bqs) == 2
    for bq in bqs:
        assert isinstance(bq, BrainQuestion)


def test_longmemeval_brain_question_corpus_non_empty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = LongMemEvalDataset._row_to_brain_question(raw)
        assert bq.corpus, f"BrainQuestion {bq.id} has empty corpus"


def test_longmemeval_brain_question_gold_paths_subset_of_corpus() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = LongMemEvalDataset._row_to_brain_question(raw)
        corpus_paths = {doc.path for doc in bq.corpus}
        dangling = bq.gold_doc_paths - corpus_paths
        assert not dangling, (
            f"BrainQuestion {bq.id} has dangling gold paths not in corpus: {dangling}"
        )


def test_longmemeval_brain_question_gold_paths_match_answer_sessions() -> None:
    """smoke-001 has answer_session_ids=['s1'] → gold path should include /sessions/s1.md."""
    raw = _EMBEDDED_SMOKE_ROWS[0]
    bq = LongMemEvalDataset._row_to_brain_question(raw)
    assert "/sessions/s1.md" in bq.gold_doc_paths, (
        f"Expected /sessions/s1.md in gold paths, got: {bq.gold_doc_paths}"
    )


def test_longmemeval_brain_question_multi_session_gold() -> None:
    """smoke-003 has answer_session_ids=['s1','s2'] → both paths must be gold."""
    raw = _EMBEDDED_SMOKE_ROWS[2]
    bq = LongMemEvalDataset._row_to_brain_question(raw)
    assert "/sessions/s1.md" in bq.gold_doc_paths
    assert "/sessions/s2.md" in bq.gold_doc_paths


def test_longmemeval_brain_question_missing_answer_session_ids_gives_empty_gold() -> None:
    """Row without answer_session_ids should produce empty gold_doc_paths."""
    raw = dict(_EMBEDDED_SMOKE_ROWS[0])
    raw.pop("answer_session_ids", None)
    bq = LongMemEvalDataset._row_to_brain_question(raw)
    assert bq.gold_doc_paths == set(), f"Expected empty gold paths, got: {bq.gold_doc_paths}"


def test_longmemeval_brain_question_doc_paths_use_session_id_scheme() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    bq = LongMemEvalDataset._row_to_brain_question(raw)
    for doc in bq.corpus:
        assert doc.path.startswith("/sessions/"), f"Unexpected path: {doc.path}"
        assert doc.path.endswith(".md"), f"Path should end in .md: {doc.path}"
