"""Statistical helpers for head-to-head eval comparison.

What's here and why:
  - bootstrap_ci   — non-parametric 95% confidence interval for any aggregate
                     metric (pass-rate, recall@10, mean latency, etc.).
                     1000-resample default is the community convention. Use
                     this instead of point estimates when reporting any
                     numeric metric — a 4pp accuracy gap with overlapping
                     CIs is not a real win.

  - mcnemar_test   — paired-binary significance test. The right test when
                     two systems answer the same questions and you want to
                     know whether A beats B. Counts only discordant pairs
                     (A correct + B wrong) vs (A wrong + B correct).
                     Returns chi-squared statistic and a two-tailed p-value.

  - paired_t_test  — paired-continuous significance test. For continuous
                     score deltas (latency differences, cost differences,
                     LLM-judge scores). Two-tailed p-value via normal-
                     distribution approximation; for small n (<30) the
                     approximation overstates significance slightly — note
                     this in any publication-grade report.

  - cohens_d       — effect size for paired deltas. p<0.05 with d<0.2 is
                     statistically significant but practically meaningless.
                     Always report d alongside p. Conventional reading:
                     <0.2 negligible · 0.2-0.5 small · 0.5-0.8 medium · >0.8 large.

All functions are pure (no side effects, no global state, no I/O). bootstrap
uses a seeded RNG for reproducibility — same input + same seed → same CI.

References:
  - Anthropic, "A statistical approach to model evaluations"
    https://www.anthropic.com/research/statistical-approach-to-model-evals
  - McNemar (1947), "Note on the sampling error of the difference between
    correlated proportions or percentages"
  - Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences"
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Final

# Default bootstrap settings — community convention.
_DEFAULT_RESAMPLES: Final = 1000
_DEFAULT_CONFIDENCE: Final = 0.95
_DEFAULT_SEED: Final = 42


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = _DEFAULT_CONFIDENCE,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Return the bootstrap (low, high) percentile CI for the mean of `values`.

    For pass-rate, pass `values` as a list of 0.0/1.0 floats (one per question).
    For latency, pass per-question latencies in ms. For recall@10, pass per-Q
    recall scores.

    Empty list returns (0.0, 0.0). Single-element list returns (v, v) — no
    variance to estimate.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    resampled_means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(sample) / n)
    resampled_means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, int(alpha * n_resamples))
    high_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1)
    return (resampled_means[low_idx], resampled_means[high_idx])


def mcnemar_test(
    a_outcomes: list[bool],
    b_outcomes: list[bool],
) -> dict[str, float]:
    """McNemar's test for paired binary outcomes.

    a_outcomes[i] and b_outcomes[i] are the pass/fail of system A and B on
    question i. The two lists must have the same length.

    Returns:
        chi2     — chi-squared statistic with 1 degree of freedom
        p_value  — two-tailed p-value (via chi-squared(1) survival fn)
        b        — count of (A correct, B wrong) pairs
        c        — count of (A wrong, B correct) pairs

    Only discordant pairs carry information; concordant pairs are ignored
    (this is the whole point — it controls for question difficulty).

    Edge case: if b + c == 0 (no disagreements at all), returns p=1.0.
    """
    if len(a_outcomes) != len(b_outcomes):
        raise ValueError(f"paired lengths must match: a={len(a_outcomes)} b={len(b_outcomes)}")
    b_count = sum(1 for a, b in zip(a_outcomes, b_outcomes, strict=True) if a and not b)
    c_count = sum(1 for a, b in zip(a_outcomes, b_outcomes, strict=True) if not a and b)
    discordant = b_count + c_count
    if discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": float(b_count), "c": float(c_count)}
    chi2 = ((b_count - c_count) ** 2) / discordant
    # chi-squared(1) survival function = erfc(sqrt(chi2/2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "chi2": chi2,
        "p_value": p_value,
        "b": float(b_count),
        "c": float(c_count),
    }


def paired_t_test(deltas: list[float]) -> dict[str, float]:
    """Paired t-test for continuous deltas (score_A - score_B per question).

    Returns:
        t        — t-statistic
        df       — degrees of freedom (n - 1)
        p_value  — two-tailed p-value via normal-distribution approximation
                   (good for n >= 30; slightly over-significant for smaller n)

    Edge cases:
        n < 2          → returns t=0, p=1
        std(deltas)==0 → returns p=0 if mean != 0 (perfectly separated),
                                  p=1 if mean == 0 (no effect)
    """
    n = len(deltas)
    if n < 2:
        return {"t": 0.0, "df": float(max(0, n - 1)), "p_value": 1.0}
    m = statistics.mean(deltas)
    s = statistics.stdev(deltas)
    if s == 0.0:
        if m == 0.0:
            return {"t": 0.0, "df": float(n - 1), "p_value": 1.0}
        return {"t": math.inf, "df": float(n - 1), "p_value": 0.0}
    t = m / (s / math.sqrt(n))
    # two-tailed p-value, normal approximation
    p_value = math.erfc(abs(t) / math.sqrt(2.0))
    return {"t": t, "df": float(n - 1), "p_value": p_value}


def cohens_d(deltas: list[float]) -> float:
    """Effect size for paired deltas: mean / std.

    Returns 0.0 for empty or single-value lists. Returns +inf when there is
    a non-zero mean with zero variance.

    Conventional reading (Cohen 1988):
        |d| < 0.2  negligible
        0.2-0.5   small
        0.5-0.8   medium
        > 0.8     large
    """
    if len(deltas) < 2:
        return 0.0
    s = statistics.stdev(deltas)
    if s == 0.0:
        m = statistics.mean(deltas)
        if m == 0.0:
            return 0.0
        return math.inf if m > 0 else -math.inf
    return statistics.mean(deltas) / s
