"""MemoryAgentBench dataset loader — unit tests."""

from __future__ import annotations

from unison_evals.memory_evals.datasets.memoryagentbench import (
    _EMBEDDED_SMOKE_ROWS,
    MemoryAgentBenchDataset,
    _split_trajectory_turns,
    _stable_id,
)
from unison_evals.types import BrainQuestion, Question

# ---------------------------------------------------------------------------
# Smoke-row parsing
# ---------------------------------------------------------------------------


def test_embedded_smoke_rows_parse_to_valid_questions() -> None:
    """Every smoke row must parse to a well-formed Question."""
    assert len(_EMBEDDED_SMOKE_ROWS) == 3
    for raw in _EMBEDDED_SMOKE_ROWS:
        q = MemoryAgentBenchDataset._row_to_question(raw)
        assert isinstance(q, Question)
        assert q.id
        assert q.question
        assert q.expected_answer
        assert q.oracle_context


def test_oracle_context_includes_trajectory_text() -> None:
    """oracle_context should contain the trajectory content."""
    for raw in _EMBEDDED_SMOKE_ROWS:
        q = MemoryAgentBenchDataset._row_to_question(raw)
        assert q.oracle_context is not None
        assert "Trajectory" in q.oracle_context
        assert len(q.oracle_context) > 20


def test_metadata_preserves_memory_ability_tag() -> None:
    """metadata["memory_ability"] must be present and non-empty."""
    for raw in _EMBEDDED_SMOKE_ROWS:
        q = MemoryAgentBenchDataset._row_to_question(raw)
        assert "memory_ability" in q.metadata
        assert q.metadata["memory_ability"]


def test_smoke_rows_cover_expected_abilities() -> None:
    """Smoke set must include accurate_retrieval, long_range_understanding,
    and conflict_resolution."""
    abilities = {
        MemoryAgentBenchDataset._row_to_question(r).metadata["memory_ability"]
        for r in _EMBEDDED_SMOKE_ROWS
    }
    assert "accurate_retrieval" in abilities
    assert "long_range_understanding" in abilities
    assert "conflict_resolution" in abilities


def test_each_smoke_row_expected_answer_nonempty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        q = MemoryAgentBenchDataset._row_to_question(raw)
        assert q.expected_answer.strip() != ""


# ---------------------------------------------------------------------------
# Conflict resolution sanity: the gold answer is the LATEST contradictory fact
# ---------------------------------------------------------------------------


def test_conflict_resolution_gold_answer_is_latest_fact() -> None:
    """The conflict_resolution smoke row's gold answer must match the most
    recent contradictory update, not the earlier (stale) fact.

    Row smoke-mab-003:
      Turn 1 claims launch = 2026-06-01
      Turn 3 updates launch = 2026-09-15  ← this is the gold answer
    """
    conflict_row = next(
        r for r in _EMBEDDED_SMOKE_ROWS if r["memory_ability"] == "conflict_resolution"
    )
    q = MemoryAgentBenchDataset._row_to_question(conflict_row)
    assert q.expected_answer == "2026-09-15", (
        f"Conflict resolution gold answer should be the latest fact (2026-09-15), "
        f"got: {q.expected_answer!r}"
    )
    # The stale date must NOT be the gold answer.
    assert "2026-06-01" not in q.expected_answer


# ---------------------------------------------------------------------------
# load() limit
# ---------------------------------------------------------------------------


def test_load_limit_returns_exactly_n(monkeypatch) -> None:
    """load(limit=2) must return exactly 2 questions from the smoke set."""
    import unison_evals.memory_evals.datasets.memoryagentbench as mod

    monkeypatch.setattr(mod, "load_dataset", _raise_network_error)

    ds = MemoryAgentBenchDataset()
    questions = list(ds.load(limit=2))
    assert len(questions) == 2


def test_load_no_limit_offline_returns_all_smoke_rows(monkeypatch) -> None:
    """When offline, all 3 embedded smoke rows should be returned."""
    import unison_evals.memory_evals.datasets.memoryagentbench as mod

    monkeypatch.setattr(mod, "load_dataset", _raise_network_error)

    ds = MemoryAgentBenchDataset()
    questions = list(ds.load())
    assert len(questions) == len(_EMBEDDED_SMOKE_ROWS)


# ---------------------------------------------------------------------------
# Network fallback
# ---------------------------------------------------------------------------


def test_network_failure_falls_back_to_embedded(monkeypatch) -> None:
    """If load_dataset raises, the loader must silently fall back to the
    embedded smoke set and still yield valid Question objects."""
    import unison_evals.memory_evals.datasets.memoryagentbench as mod

    monkeypatch.setattr(mod, "load_dataset", _raise_network_error)

    ds = MemoryAgentBenchDataset()
    questions = list(ds.load())
    assert len(questions) == len(_EMBEDDED_SMOKE_ROWS)
    for q in questions:
        assert isinstance(q, Question)
        assert q.id
        assert q.question
        assert q.expected_answer


# ---------------------------------------------------------------------------
# Stable ID
# ---------------------------------------------------------------------------


def test_stable_id_is_deterministic() -> None:
    """_stable_id must return the same string for the same row content."""
    row = {"question": "What is X?", "answer": "Y"}
    id1 = _stable_id(row)
    id2 = _stable_id(row)
    assert id1 == id2
    assert id1.startswith("mab-")


def test_stable_id_differs_for_different_rows() -> None:
    row_a = {"question": "What is X?", "answer": "Y"}
    row_b = {"question": "What is Z?", "answer": "W"}
    assert _stable_id(row_a) != _stable_id(row_b)


def test_row_without_question_id_gets_stable_id() -> None:
    """A row lacking question_id / id / qid must still get a deterministic id."""
    row = {
        "question": "No id here",
        "answer": "42",
        "oracle_context": "## Trajectory\n\nsome context",
        "memory_ability": "accurate_retrieval",
        "turn_count": 1,
        "source_row_id": None,
    }
    q = MemoryAgentBenchDataset._row_to_question(row)
    assert q.id.startswith("mab-")


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_metadata_turn_count_preserved() -> None:
    raw = _EMBEDDED_SMOKE_ROWS[0]
    q = MemoryAgentBenchDataset._row_to_question(raw)
    assert "turn_count" in q.metadata


# ---------------------------------------------------------------------------
# Track 1 — load_brain_questions
# ---------------------------------------------------------------------------


def test_memoryagentbench_load_brain_questions_returns_brain_question_objects(
    monkeypatch,
) -> None:
    import unison_evals.memory_evals.datasets.memoryagentbench as mod

    monkeypatch.setattr(mod, "load_dataset", _raise_network_error)
    ds = MemoryAgentBenchDataset()
    bqs = list(ds.load_brain_questions(limit=2))
    assert len(bqs) == 2
    for bq in bqs:
        assert isinstance(bq, BrainQuestion)


def test_memoryagentbench_brain_question_corpus_non_empty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MemoryAgentBenchDataset._row_to_brain_question(raw)
        assert bq.corpus, f"BrainQuestion {bq.id} has empty corpus"


def test_memoryagentbench_brain_question_gold_paths_subset_of_corpus() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MemoryAgentBenchDataset._row_to_brain_question(raw)
        corpus_paths = {doc.path for doc in bq.corpus}
        dangling = bq.gold_doc_paths - corpus_paths
        assert not dangling, (
            f"BrainQuestion {bq.id} has dangling gold paths not in corpus: {dangling}"
        )


def test_memoryagentbench_brain_question_accurate_retrieval_has_gold() -> None:
    """The accurate_retrieval smoke row contains the answer in the trajectory."""
    raw = next(r for r in _EMBEDDED_SMOKE_ROWS if r["memory_ability"] == "accurate_retrieval")
    bq = MemoryAgentBenchDataset._row_to_brain_question(raw)
    assert bq.gold_doc_paths, (
        f"Accurate retrieval row should find the answer '9:45 PM' in at least one turn; "
        f"corpus={[d.body for d in bq.corpus]}"
    )


def test_memoryagentbench_brain_question_doc_paths_use_turn_scheme() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MemoryAgentBenchDataset._row_to_brain_question(raw)
        for doc in bq.corpus:
            assert doc.path.startswith("/turns/"), f"Unexpected path: {doc.path}"
            assert doc.path.endswith(".md"), f"Path should end in .md: {doc.path}"


def test_memoryagentbench_brain_question_corpus_body_non_empty() -> None:
    for raw in _EMBEDDED_SMOKE_ROWS:
        bq = MemoryAgentBenchDataset._row_to_brain_question(raw)
        for doc in bq.corpus:
            assert doc.body.strip(), f"Document {doc.path} has empty body"


# ---------------------------------------------------------------------------
# _split_trajectory_turns helper
# ---------------------------------------------------------------------------


def test_split_trajectory_turns_splits_on_newlines() -> None:
    text = "Turn 1 USER: Hello\nTurn 2 ASSISTANT: Hi"
    turns = _split_trajectory_turns(text)
    assert len(turns) == 2
    assert turns[0] == "Turn 1 USER: Hello"
    assert turns[1] == "Turn 2 ASSISTANT: Hi"


def test_split_trajectory_turns_drops_blank_lines() -> None:
    text = "Turn 1 USER: Hello\n\nTurn 2 ASSISTANT: Hi\n"
    turns = _split_trajectory_turns(text)
    assert len(turns) == 2


def test_split_trajectory_turns_empty_string() -> None:
    assert _split_trajectory_turns("") == []


def test_split_trajectory_turns_single_line() -> None:
    turns = _split_trajectory_turns("Single line text")
    assert len(turns) == 1
    assert turns[0] == "Single line text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_network_error(*_a: object, **_kw: object) -> None:
    raise RuntimeError("simulated network failure")
