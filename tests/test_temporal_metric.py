"""Unit tests for metrics.temporal.temporal_correct_at_1."""

from __future__ import annotations

import pytest

from unison_evals.memory_evals.metrics.temporal import temporal_correct_at_1

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_retrieved_returns_zero() -> None:
    assert temporal_correct_at_1([], {"/facts/f001.md"}, {"/facts/f001.md": "f001"}) == 0.0


def test_empty_gold_returns_zero() -> None:
    assert temporal_correct_at_1(["/facts/f001.md"], set(), {"/facts/f001.md": "f001"}) == 0.0


def test_both_empty_returns_zero() -> None:
    assert temporal_correct_at_1([], set(), {}) == 0.0


# ---------------------------------------------------------------------------
# No version constraint (plain hit@1)
# ---------------------------------------------------------------------------


def test_no_version_constraint_top1_in_gold_returns_one() -> None:
    """Empty expected_versions → degrades to hit@1."""
    assert temporal_correct_at_1(["/facts/f001.md"], {"/facts/f001.md"}, {}) == 1.0


def test_no_version_constraint_top1_not_in_gold_returns_zero() -> None:
    assert temporal_correct_at_1(["/facts/f002.md"], {"/facts/f001.md"}, {}) == 0.0


def test_no_version_constraint_second_doc_in_gold_does_not_count() -> None:
    """Only rank-1 matters."""
    assert (
        temporal_correct_at_1(["/facts/f002.md", "/facts/f001.md"], {"/facts/f001.md"}, {}) == 0.0
    )


# ---------------------------------------------------------------------------
# With version constraint
# ---------------------------------------------------------------------------


def test_top1_in_gold_correct_version_returns_one() -> None:
    """top-1 path contains the expected fact_id → full score."""
    retrieved = ["/facts/f001.md"]
    gold = {"/facts/f001.md"}
    expected_versions = {"/facts/f001.md": "f001"}
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 1.0


def test_top1_in_gold_wrong_version_returns_half() -> None:
    """top-1 path is in gold but encodes wrong fact_id → partial score."""
    # Simulates: question asks as_of 2022-06-01 (gold=f001), but system
    # returned f002 (superseding fact). f002 IS a gold path here (we're
    # testing the version mismatch case where the gold set was intentionally
    # set to include f002 but expected_versions points to f001).
    retrieved = ["/facts/f002.md"]
    gold = {"/facts/f002.md"}  # in gold (entity correct)
    expected_versions = {"/facts/f002.md": "f001"}  # but expected version is f001
    # path "/facts/f002.md" does NOT contain "f001" → partial score
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 0.5


def test_top1_in_gold_correct_version_substring_check() -> None:
    """Version label is matched as substring of the doc_path."""
    retrieved = ["/facts/f042.md"]
    gold = {"/facts/f042.md"}
    expected_versions = {"/facts/f042.md": "f042"}
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 1.0


def test_top1_not_in_gold_returns_zero_regardless_of_version() -> None:
    retrieved = ["/facts/f002.md"]
    gold = {"/facts/f001.md"}
    expected_versions = {"/facts/f001.md": "f001"}
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 0.0


def test_top1_in_gold_not_in_expected_versions_returns_half() -> None:
    """top-1 in gold but not keyed in expected_versions → partial credit."""
    retrieved = ["/facts/f001.md"]
    gold = {"/facts/f001.md"}
    expected_versions = {"/facts/f003.md": "f003"}  # f001 not mentioned
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 0.5


# ---------------------------------------------------------------------------
# Multi-candidate ranking (only top-1 matters)
# ---------------------------------------------------------------------------


def test_correct_version_at_rank2_scores_zero() -> None:
    """Having the correct version at rank 2 is still 0.0 — metric is @1."""
    retrieved = ["/facts/f002.md", "/facts/f001.md"]
    gold = {"/facts/f001.md"}
    expected_versions = {"/facts/f001.md": "f001"}
    # top-1 is f002, which is NOT in gold
    assert temporal_correct_at_1(retrieved, gold, expected_versions) == 0.0


# ---------------------------------------------------------------------------
# Score range invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved,gold,expected_versions,expected_score",
    [
        (["/facts/f001.md"], {"/facts/f001.md"}, {"/facts/f001.md": "f001"}, 1.0),
        (["/facts/f001.md"], {"/facts/f001.md"}, {}, 1.0),
        (["/facts/f002.md"], {"/facts/f001.md"}, {"/facts/f001.md": "f001"}, 0.0),
        ([], {"/facts/f001.md"}, {"/facts/f001.md": "f001"}, 0.0),
    ],
)
def test_score_in_valid_range(
    retrieved: list[str],
    gold: set[str],
    expected_versions: dict[str, str],
    expected_score: float,
) -> None:
    score = temporal_correct_at_1(retrieved, gold, expected_versions)
    assert score == expected_score
    assert 0.0 <= score <= 1.0
