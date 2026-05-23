"""Tests for BitempoQADataset.

Covers:
- Corpus JSONL loads without IO errors and has ≥100 entries
- Questions JSONL loads without IO errors and has ≥100 entries
- load(limit=5) returns exactly 5 valid Question objects
- Each Question has a non-empty oracle_context that includes the relevant facts
- All four question_types are represented
- All three difficulty levels are represented
- as_of questions embed the as_of date in oracle_context
- No dangling fact_id references (every fact_id in questions resolves in corpus)
- Data-integrity: every terminated fact (valid_to != null) has a successor in corpus
"""

from __future__ import annotations

import json
from pathlib import Path

from unison_evals.memory_evals.datasets.bitempoqa import BitempoQADataset
from unison_evals.types import BrainQuestion

DATA_DIR = Path(__file__).parent.parent / "data" / "bitempoqa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus_raw() -> list[dict]:
    path = DATA_DIR / "corpus.jsonl"
    facts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def _load_questions_raw() -> list[dict]:
    path = DATA_DIR / "questions.jsonl"
    questions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


# ---------------------------------------------------------------------------
# File-level assertions
# ---------------------------------------------------------------------------


def test_corpus_file_exists():
    assert (DATA_DIR / "corpus.jsonl").is_file()


def test_questions_file_exists():
    assert (DATA_DIR / "questions.jsonl").is_file()


def test_corpus_has_at_least_100_entries():
    facts = _load_corpus_raw()
    assert len(facts) >= 100, f"Expected ≥100 corpus facts, got {len(facts)}"


def test_questions_has_at_least_100_entries():
    questions = _load_questions_raw()
    assert len(questions) >= 100, f"Expected ≥100 questions, got {len(questions)}"


def test_corpus_schema():
    required = {
        "fact_id",
        "subject",
        "predicate",
        "object",
        "valid_from",
        "valid_to",
        "supersedes",
        "source_id",
    }
    for fact in _load_corpus_raw():
        missing = required - set(fact.keys())
        assert not missing, f"Fact {fact.get('fact_id')} missing fields: {missing}"


def test_questions_schema():
    required = {
        "id",
        "question",
        "expected_answer",
        "as_of",
        "fact_ids",
        "question_type",
        "difficulty",
    }
    for q in _load_questions_raw():
        missing = required - set(q.keys())
        assert not missing, f"Question {q.get('id')} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Loader / Question shape
# ---------------------------------------------------------------------------


def test_load_limit_returns_correct_count():
    ds = BitempoQADataset()
    questions = list(ds.load(limit=5))
    assert len(questions) == 5


def test_load_no_limit_returns_all():
    ds = BitempoQADataset()
    questions = list(ds.load())
    assert len(questions) >= 100


def test_question_fields_populated():
    ds = BitempoQADataset()
    for q in ds.load(limit=10):
        assert q.id, "Question must have a non-empty id"
        assert q.question, "Question must have a non-empty question text"
        assert q.expected_answer, "Question must have a non-empty expected_answer"


def test_oracle_context_non_empty():
    ds = BitempoQADataset()
    for q in ds.load(limit=10):
        assert q.oracle_context, f"Question {q.id} has empty oracle_context"


def test_oracle_context_contains_relevant_fact_text():
    """oracle_context must mention at least one fact object or subject."""
    ds = BitempoQADataset()
    corpus = ds._load_corpus()
    for q in list(ds.load(limit=20)):
        fact_ids = q.metadata.get("fact_ids", [])
        assert fact_ids, f"Question {q.id} has no fact_ids in metadata"
        context = q.oracle_context or ""
        first_fact = corpus.get(fact_ids[0])
        assert first_fact is not None, f"fact_id {fact_ids[0]} not in corpus"
        assert first_fact["object"] in context or first_fact["subject"] in context, (
            f"Question {q.id}: oracle_context does not mention expected fact content. "
            f"Context: {context!r}"
        )


# ---------------------------------------------------------------------------
# Coverage: all question_types and difficulty levels
# ---------------------------------------------------------------------------


def test_all_question_types_present():
    ds = BitempoQADataset()
    types_found: set[str] = set()
    for q in ds.load():
        qt = q.metadata.get("question_type")
        if qt:
            types_found.add(qt)

    expected = {"current_truth", "historical_truth", "predecessor", "transition"}
    assert expected == types_found, f"Expected question types {expected}, found {types_found}"


def test_all_difficulty_levels_present():
    ds = BitempoQADataset()
    levels_found: set[int] = set()
    for q in ds.load():
        d = q.metadata.get("difficulty")
        if d is not None:
            levels_found.add(d)

    assert levels_found == {1, 2, 3}, f"Expected difficulty levels {{1,2,3}}, found {levels_found}"


def test_question_type_distribution_roughly_even():
    """Each type should have at least 15 questions (generous lower bound on 25 target)."""
    ds = BitempoQADataset()
    counts: dict[str, int] = {}
    for q in ds.load():
        qt = q.metadata.get("question_type", "")
        counts[qt] = counts.get(qt, 0) + 1

    for qt, count in counts.items():
        assert count >= 15, f"Question type '{qt}' only has {count} questions, expected ≥15"


# ---------------------------------------------------------------------------
# as_of questions
# ---------------------------------------------------------------------------


def test_as_of_questions_embed_date_in_context():
    """historical_truth questions must have as_of set and the date in oracle_context."""
    ds = BitempoQADataset()
    historical = [q for q in ds.load() if q.metadata.get("question_type") == "historical_truth"]
    assert historical, "No historical_truth questions found"
    for q in historical:
        as_of = q.metadata.get("as_of")
        assert as_of, f"Question {q.id} is historical_truth but has no as_of"
        context = q.oracle_context or ""
        assert as_of in context, (
            f"Question {q.id}: as_of date {as_of!r} not found in oracle_context: {context!r}"
        )


def test_current_truth_questions_have_no_as_of():
    ds = BitempoQADataset()
    current = [q for q in ds.load() if q.metadata.get("question_type") == "current_truth"]
    assert current, "No current_truth questions found"
    for q in current:
        as_of = q.metadata.get("as_of")
        assert as_of is None, f"Question {q.id} is current_truth but has as_of={as_of!r}"


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_no_dangling_fact_id_references():
    """Every fact_id referenced in questions must exist in corpus."""
    facts_raw = _load_corpus_raw()
    corpus_ids = {f["fact_id"] for f in facts_raw}
    questions_raw = _load_questions_raw()
    for q in questions_raw:
        for fid in q.get("fact_ids", []):
            assert fid in corpus_ids, (
                f"Question {q['id']} references fact_id {fid!r} which is not in corpus"
            )


def test_terminated_facts_have_successors():
    """A fact with valid_to != null must have a successor whose supersedes == that fact_id."""
    facts_raw = _load_corpus_raw()
    supersedes_set = {f["supersedes"] for f in facts_raw if f["supersedes"] is not None}
    for fact in facts_raw:
        if fact["valid_to"] is not None:
            assert fact["fact_id"] in supersedes_set, (
                f"Fact {fact['fact_id']} has valid_to={fact['valid_to']!r} "
                f"but no successor fact supersedes it"
            )


def test_successor_valid_from_matches_predecessor_valid_to():
    """For each supersession chain: successor.valid_from == predecessor.valid_to."""
    facts_by_id = {f["fact_id"]: f for f in _load_corpus_raw()}
    for fact in facts_by_id.values():
        pred_id = fact.get("supersedes")
        if pred_id is None:
            continue
        pred = facts_by_id.get(pred_id)
        assert pred is not None, f"Fact {fact['fact_id']} supersedes non-existent {pred_id}"
        assert fact["valid_from"] == pred["valid_to"], (
            f"Chain break: {fact['fact_id']}.valid_from={fact['valid_from']!r} != "
            f"{pred_id}.valid_to={pred['valid_to']!r}"
        )


# ---------------------------------------------------------------------------
# Subject coverage
# ---------------------------------------------------------------------------


def test_questions_cover_at_least_20_subjects():
    """Questions must span at least 20 distinct subjects (via fact lookups)."""
    facts_raw = _load_corpus_raw()
    corpus = {f["fact_id"]: f for f in facts_raw}
    questions_raw = _load_questions_raw()
    subjects: set[str] = set()
    for q in questions_raw:
        for fid in q.get("fact_ids", []):
            fact = corpus.get(fid)
            if fact:
                subjects.add(fact["subject"])
    assert len(subjects) >= 20, f"Expected ≥20 distinct subjects, found {len(subjects)}: {subjects}"


# ---------------------------------------------------------------------------
# Track 1 — load_brain_questions
# ---------------------------------------------------------------------------


def test_load_brain_questions_returns_brain_question_objects():
    ds = BitempoQADataset()
    bqs = list(ds.load_brain_questions(limit=2))
    assert len(bqs) == 2
    for bq in bqs:
        assert isinstance(bq, BrainQuestion)


def test_load_brain_questions_corpus_non_empty():
    ds = BitempoQADataset()
    for bq in ds.load_brain_questions(limit=2):
        assert bq.corpus, f"BrainQuestion {bq.id} has empty corpus"


def test_load_brain_questions_gold_paths_non_empty():
    ds = BitempoQADataset()
    for bq in ds.load_brain_questions(limit=5):
        assert bq.gold_doc_paths, f"BrainQuestion {bq.id} has empty gold_doc_paths"


def test_load_brain_questions_gold_paths_subset_of_corpus():
    ds = BitempoQADataset()
    for bq in ds.load_brain_questions(limit=5):
        corpus_paths = {doc.path for doc in bq.corpus}
        dangling = bq.gold_doc_paths - corpus_paths
        assert not dangling, (
            f"BrainQuestion {bq.id} has dangling gold paths not in corpus: {dangling}"
        )


def test_load_brain_questions_paths_use_fact_id_scheme():
    ds = BitempoQADataset()
    bq = next(iter(ds.load_brain_questions(limit=1)))
    for doc in bq.corpus:
        assert doc.path.startswith("/facts/"), f"Unexpected path scheme: {doc.path}"
        assert doc.path.endswith(".md"), f"Path should end in .md: {doc.path}"


def test_load_brain_questions_corpus_body_non_empty():
    ds = BitempoQADataset()
    bq = next(iter(ds.load_brain_questions(limit=1)))
    for doc in bq.corpus:
        assert doc.body.strip(), f"Document {doc.path} has empty body"


def test_load_brain_questions_gold_paths_match_fact_ids():
    """Gold paths must correspond to the question's fact_ids."""
    ds = BitempoQADataset()
    for bq in ds.load_brain_questions(limit=10):
        fact_ids = bq.metadata.get("fact_ids", [])
        expected_gold = {f"/facts/{fid}.md" for fid in fact_ids}
        assert bq.gold_doc_paths == expected_gold, (
            f"BrainQuestion {bq.id} gold paths {bq.gold_doc_paths!r} "
            f"do not match expected {expected_gold!r}"
        )
