"""Temporal correctness metrics for BITEMPORAL brain eval mode.

temporal_correct_at_1:
  Checks whether the top-1 retrieved document is not only in the gold set
  but also the version-correct document for the question's as_of timestamp.

  Score:
    1.0 — top-1 doc is in gold AND its version label matches expected_versions
    0.5 — top-1 doc is in gold but wrong version (correct entity, stale/future fact)
    0.0 — top-1 doc not in gold, or retrieved list is empty

  When expected_versions is empty (non-temporal question), falls back to
  plain hit@1: 1.0 if top-1 in gold, else 0.0. This lets the metric be
  computed uniformly over all questions in a BITEMPORAL run.
"""

from __future__ import annotations


def temporal_correct_at_1(
    retrieved: list[str],
    gold: set[str],
    expected_versions: dict[str, str],
) -> float:
    """Score the top-1 retrieved doc for temporal correctness.

    Args:
        retrieved: Ranked list of doc_paths (index 0 = rank 1).
        gold: Set of doc_paths that are relevant to the question.
        expected_versions: Maps doc_path → version label (e.g. the fact_id of
            the temporally-correct document).  When non-empty the metric
            enforces version correctness in addition to path membership.
            When empty the metric degrades to plain hit@1.

    Returns:
        1.0, 0.5, or 0.0 as described in the module docstring.
    """
    if not retrieved or not gold:
        return 0.0

    top1 = retrieved[0]

    if top1 not in gold:
        return 0.0

    # No version constraint — plain hit@1.
    if not expected_versions:
        return 1.0

    # Version constraint: check whether the top-1 doc has the expected version.
    expected = expected_versions.get(top1)
    if expected is None:
        # top1 is in gold but not in expected_versions — partial credit because
        # the entity is right even if we can't verify the version.
        return 0.5

    # The adapter is expected to surface the version label in chunk metadata.
    # For now we treat the doc_path itself as a version proxy: if the path
    # matches the expected_versions key AND the value equals the top1 path
    # (i.e. the version IS the path), score 1.0; otherwise 0.5.
    # Adapters that version docs by path (e.g. /facts/f001.md vs /facts/f002.md)
    # will naturally produce the right path if retrieval is correct.
    #
    # For BitempoQA the version label is the fact_id (e.g. "f001").
    # The doc path is "/facts/f001.md".  So we check that the path encodes
    # the expected fact_id.
    if expected in top1:
        return 1.0
    return 0.5
