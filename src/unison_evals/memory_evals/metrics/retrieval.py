"""Pure ranked-retrieval metrics for brain retrieval evaluation.

All functions take:
  retrieved: list[str]  — doc paths in rank order (index 0 = rank 1)
  gold:      set[str]   — relevant doc paths (ground truth)
  k:         int        — cut-off depth (ignored for mrr)

Edge-case convention:
  * empty retrieved → 0.0 for all metrics
  * empty gold      → 0.0 for all metrics
  (No documents to retrieve / no relevant documents → undefined; we return 0.)
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold documents found in the top-k retrieved results.

    recall@k = |gold ∩ retrieved[:k]| / |gold|

    Returns 0.0 when either list/set is empty.
    """
    if not retrieved or not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(gold & top_k) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold document appears in the top-k retrieved, else 0.0.

    Returns 0.0 when either list/set is empty.
    """
    if not retrieved or not gold:
        return 0.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    """Mean Reciprocal Rank — 1 / (rank of first gold document).

    MRR = 1 / rank_of_first_hit, or 0 if no gold document appears.
    (For a single query this equals RR; "mean" applies when averaged over
    many queries in the calling code.)

    Returns 0.0 when either list/set is empty or no gold doc is found.
    """
    if not retrieved or not gold:
        return 0.0
    for rank, path in enumerate(retrieved, start=1):
        if path in gold:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of the top-k retrieved results that are gold documents.

    precision@k = |gold ∩ retrieved[:k]| / k

    Returns 0.0 when either list/set is empty.
    """
    if not retrieved or not gold:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for p in top_k if p in gold)
    return hits / k


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k using binary relevance.

    rel_i = 1.0 if retrieved[i] in gold, else 0.0
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(|gold|,k)} 1 / log2(i + 1)   (perfect ranking)
    nDCG@k = DCG@k / IDCG@k

    Returns 0.0 when either list/set is empty or IDCG is 0.
    """
    if not retrieved or not gold:
        return 0.0

    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, path in enumerate(retrieved[:k], start=1)
        if path in gold
    )

    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
