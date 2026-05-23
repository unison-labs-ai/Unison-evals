"""Unit tests for retrieval metric functions."""

from __future__ import annotations

import math

from unison_evals.memory_evals.metrics.retrieval import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_full_overlap() -> None:
    assert recall_at_k(["/a.md", "/b.md", "/c.md"], {"/a.md", "/b.md"}, k=2) == 1.0


def test_recall_at_k_partial_overlap() -> None:
    # gold={a,b,c}, top-2=[a,d] → 1/3
    result = recall_at_k(["/a.md", "/d.md", "/b.md", "/c.md"], {"/a.md", "/b.md", "/c.md"}, k=2)
    assert math.isclose(result, 1 / 3)


def test_recall_at_k_no_overlap() -> None:
    assert recall_at_k(["/x.md", "/y.md"], {"/a.md"}, k=2) == 0.0


def test_recall_at_k_k_larger_than_retrieved() -> None:
    # k=10 but only 2 docs retrieved, both gold → recall=1.0
    assert recall_at_k(["/a.md", "/b.md"], {"/a.md", "/b.md"}, k=10) == 1.0


def test_recall_at_k_k_zero() -> None:
    # k=0 → top-0 is empty, no overlap possible
    assert recall_at_k(["/a.md"], {"/a.md"}, k=0) == 0.0


def test_recall_at_k_empty_retrieved() -> None:
    assert recall_at_k([], {"/a.md"}, k=5) == 0.0


def test_recall_at_k_empty_gold() -> None:
    assert recall_at_k(["/a.md"], set(), k=5) == 0.0


def test_recall_at_k_both_empty() -> None:
    assert recall_at_k([], set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------


def test_hit_at_k_found_at_rank_1() -> None:
    assert hit_at_k(["/a.md", "/b.md"], {"/a.md"}, k=1) == 1.0


def test_hit_at_k_found_at_rank_k() -> None:
    assert hit_at_k(["/x.md", "/y.md", "/a.md"], {"/a.md"}, k=3) == 1.0


def test_hit_at_k_not_found_within_k() -> None:
    assert hit_at_k(["/x.md", "/y.md", "/a.md"], {"/a.md"}, k=2) == 0.0


def test_hit_at_k_empty_retrieved() -> None:
    assert hit_at_k([], {"/a.md"}, k=5) == 0.0


def test_hit_at_k_empty_gold() -> None:
    assert hit_at_k(["/a.md"], set(), k=5) == 0.0


def test_hit_at_k_both_empty() -> None:
    assert hit_at_k([], set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_gold_at_rank_1() -> None:
    assert mrr(["/a.md", "/b.md", "/c.md"], {"/a.md"}) == 1.0


def test_mrr_gold_at_rank_5() -> None:
    retrieved = ["/x.md", "/y.md", "/z.md", "/w.md", "/a.md"]
    assert math.isclose(mrr(retrieved, {"/a.md"}), 1 / 5)


def test_mrr_gold_not_present() -> None:
    assert mrr(["/x.md", "/y.md"], {"/a.md"}) == 0.0


def test_mrr_multiple_gold_uses_first_hit() -> None:
    # a is rank 2, b is rank 4 → MRR = 1/2
    retrieved = ["/x.md", "/a.md", "/y.md", "/b.md"]
    assert math.isclose(mrr(retrieved, {"/a.md", "/b.md"}), 1 / 2)


def test_mrr_empty_retrieved() -> None:
    assert mrr([], {"/a.md"}) == 0.0


def test_mrr_empty_gold() -> None:
    assert mrr(["/a.md"], set()) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_perfect_ranking() -> None:
    # All gold docs at top → nDCG = 1.0
    assert ndcg_at_k(["/a.md", "/b.md", "/c.md"], {"/a.md", "/b.md"}, k=2) == 1.0


def test_ndcg_at_k_hand_computed() -> None:
    # retrieved = [a, x, b], gold = {a, b}, k=3
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309...
    retrieved = ["/a.md", "/x.md", "/b.md"]
    gold = {"/a.md", "/b.md"}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)  # rank 1, rank 3
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal: 2 gold docs at positions 1, 2
    expected = dcg / idcg
    assert math.isclose(ndcg_at_k(retrieved, gold, k=3), expected, rel_tol=1e-9)


def test_ndcg_at_k_no_overlap() -> None:
    assert ndcg_at_k(["/x.md", "/y.md"], {"/a.md"}, k=2) == 0.0


def test_ndcg_at_k_k_smaller_than_all_gold() -> None:
    # gold has 5 docs, k=2, only top-2 can contribute
    gold = {f"/{i}.md" for i in range(5)}
    retrieved = ["/0.md", "/1.md"] + [f"/extra{i}.md" for i in range(8)]
    # DCG = 1/log2(2) + 1/log2(3); IDCG = same (only 2 ideal positions)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(retrieved, gold, k=2), dcg / idcg)


def test_ndcg_at_k_empty_retrieved() -> None:
    assert ndcg_at_k([], {"/a.md"}, k=5) == 0.0


def test_ndcg_at_k_empty_gold() -> None:
    assert ndcg_at_k(["/a.md"], set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_all_gold() -> None:
    assert precision_at_k(["/a.md", "/b.md"], {"/a.md", "/b.md"}, k=2) == 1.0


def test_precision_at_k_half_gold() -> None:
    # top-4: a(gold), x, b(gold), y → 2/4 = 0.5
    assert precision_at_k(["/a.md", "/x.md", "/b.md", "/y.md"], {"/a.md", "/b.md"}, k=4) == 0.5


def test_precision_at_k_none_gold() -> None:
    assert precision_at_k(["/x.md", "/y.md"], {"/a.md"}, k=2) == 0.0


def test_precision_at_k_k_larger_than_list() -> None:
    # retrieved=[a,b], k=5 → 2/5 = 0.4
    result = precision_at_k(["/a.md", "/b.md"], {"/a.md", "/b.md"}, k=5)
    assert math.isclose(result, 2 / 5)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], {"/a.md"}, k=5) == 0.0


def test_precision_at_k_empty_gold() -> None:
    assert precision_at_k(["/a.md"], set(), k=5) == 0.0
