"""Tests for metrics/stats.py — bootstrap CI, McNemar's, paired t, Cohen's d."""

from __future__ import annotations

import math

import pytest

from unison_evals.memory_evals.metrics.stats import (
    bootstrap_ci,
    cohens_d,
    mcnemar_test,
    paired_t_test,
)

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_empty_returns_zero_zero() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_single_value_returns_value_value() -> None:
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_constant_values_returns_constant() -> None:
    low, high = bootstrap_ci([0.5] * 50)
    assert low == 0.5 and high == 0.5


def test_bootstrap_50_50_passes_brackets_mean() -> None:
    values = [1.0] * 50 + [0.0] * 50
    low, high = bootstrap_ci(values, n_resamples=2000)
    # Mean is 0.5; CI should contain it and not be wildly wide.
    assert 0.35 < low < 0.5 < high < 0.65


def test_bootstrap_deterministic_with_seed() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_ci_low_le_high() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    low, high = bootstrap_ci(values)
    assert low <= high


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------


def test_mcnemar_no_disagreements_returns_p_one() -> None:
    a = [True, True, False, False]
    res = mcnemar_test(a, list(a))
    assert res["p_value"] == 1.0
    assert res["chi2"] == 0.0


def test_mcnemar_all_discordant_a_wins() -> None:
    # A is correct on every question B is wrong on, and vice versa
    a = [True, True, True, True, True]
    b = [False, False, False, False, False]
    res = mcnemar_test(a, b)
    # b=5 c=0 → chi2 = (5-0)^2 / 5 = 5
    assert res["chi2"] == pytest.approx(5.0)
    assert res["b"] == 5.0
    assert res["c"] == 0.0
    # p should be small (significant)
    assert res["p_value"] < 0.05


def test_mcnemar_balanced_disagreements_returns_high_p() -> None:
    # Equal counts of (A right, B wrong) and (A wrong, B right) → no win
    a = [True, True, False, False]
    b = [False, False, True, True]
    res = mcnemar_test(a, b)
    assert res["chi2"] == 0.0
    assert res["p_value"] == 1.0


def test_mcnemar_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        mcnemar_test([True, False], [True])


# ---------------------------------------------------------------------------
# paired_t_test
# ---------------------------------------------------------------------------


def test_paired_t_empty_or_single() -> None:
    assert paired_t_test([])["p_value"] == 1.0
    assert paired_t_test([0.5])["p_value"] == 1.0


def test_paired_t_all_zero_deltas_returns_p_one() -> None:
    res = paired_t_test([0.0, 0.0, 0.0, 0.0])
    assert res["t"] == 0.0
    assert res["p_value"] == 1.0


def test_paired_t_constant_nonzero_delta_returns_inf_t() -> None:
    res = paired_t_test([0.1, 0.1, 0.1, 0.1])
    assert math.isinf(res["t"])
    assert res["p_value"] == 0.0


def test_paired_t_significant_positive_delta() -> None:
    # Strong positive deltas — should give a small p
    deltas = [0.2, 0.15, 0.25, 0.18, 0.22, 0.19, 0.21, 0.16, 0.23, 0.20]
    res = paired_t_test(deltas)
    assert res["t"] > 0
    assert res["p_value"] < 0.001


def test_paired_t_no_effect_returns_high_p() -> None:
    # Symmetric around zero — should give a large p
    deltas = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05]
    res = paired_t_test(deltas)
    assert res["p_value"] > 0.5


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_empty_or_single_returns_zero() -> None:
    assert cohens_d([]) == 0.0
    assert cohens_d([0.5]) == 0.0


def test_cohens_d_constant_zero_returns_zero() -> None:
    assert cohens_d([0.0, 0.0, 0.0]) == 0.0


def test_cohens_d_constant_nonzero_returns_inf() -> None:
    d = cohens_d([0.1, 0.1, 0.1])
    assert math.isinf(d) and d > 0


def test_cohens_d_large_effect() -> None:
    deltas = [1.0, 1.1, 0.95, 1.05, 0.98, 1.02]
    d = cohens_d(deltas)
    # Mean ~1.0, std ~0.05 → d ~20 (very large)
    assert d > 5.0


def test_cohens_d_small_effect() -> None:
    # Mean small, std large
    deltas = [0.05, -0.5, 0.6, -0.4, 0.5, -0.45, 0.4]
    d = cohens_d(deltas)
    assert abs(d) < 0.5
