"""Tests for krauss.regimes.analysis.

Focus areas:
  - sharpe(): edge cases (zero std, empty) return 0 without raising
  - cross_sectional_zscore(): within-day, no-lookahead
  - bootstrap_sharpe_ci(): reproducibility (fixed seed), sensible ordering
  - rule_cash_in_regime(): zeroes returns only on target-regime days

The integration-level tests — regime_stats() vs known notebook outputs —
live in the notebooks themselves (they print numbers that match hardcoded
expected values). Unit tests here stay on the primitives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from krauss.regimes.analysis import (
    bootstrap_sharpe_ci,
    cross_sectional_zscore,
    rule_cash_in_regime,
    sharpe,
)
from krauss.regimes.vix_regimes import RegimeConfig, label_vix_regimes


# --------------------------------- sharpe ------------------------------------


def test_sharpe_empty_is_zero():
    assert sharpe(pd.Series([], dtype=float)) == 0.0


def test_sharpe_constant_is_zero():
    # Zero std -> div by zero guard -> 0.0
    assert sharpe(pd.Series([0.001] * 100)) == 0.0


def test_sharpe_sign_follows_mean():
    # All-positive returns above rf: positive Sharpe.
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.normal(0.001, 0.01, 1000))  # ~25% ann ret, ~16% ann std
    assert sharpe(pos) > 0
    neg = pd.Series(rng.normal(-0.001, 0.01, 1000))
    assert sharpe(neg) < 0


# --------------------------- cross-sectional z-score --------------------------


def test_cs_zscore_within_day_only():
    # Two days, 3 stocks each. z-score within day 1 shouldn't depend on day 2.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
        "score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    z = cross_sectional_zscore(df, "score")
    # Day 1 z-scores: (1-2)/1, (2-2)/1, (3-2)/1  = -1, 0, 1
    np.testing.assert_allclose(z.iloc[:3].values, [-1.0, 0.0, 1.0])
    # Day 2 has different scale but same relative pattern.
    np.testing.assert_allclose(z.iloc[3:].values, [-1.0, 0.0, 1.0])


def test_cs_zscore_constant_day_becomes_nan():
    # If all scores on a day are identical, std is zero -> z is NaN (safer
    # than dividing by zero).
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3),
        "score": [5.0, 5.0, 5.0],
    })
    z = cross_sectional_zscore(df, "score")
    assert z.isna().all()


# --------------------------- bootstrap CI ------------------------------------


def test_bootstrap_sharpe_ci_reproducible():
    rng = np.random.default_rng(123)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    a = bootstrap_sharpe_ci(returns, n_boot=200, seed=7)
    b = bootstrap_sharpe_ci(returns, n_boot=200, seed=7)
    assert a == b, "Same seed must give same CI"


def test_bootstrap_sharpe_ci_ordering():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    point, lo, hi = bootstrap_sharpe_ci(returns, n_boot=500, seed=1)
    assert lo <= point <= hi, "CI should bracket the point estimate"
    assert lo < hi, "CI must have positive width"


def test_bootstrap_sharpe_ci_empty():
    # Empty input shouldn't crash
    point, lo, hi = bootstrap_sharpe_ci(pd.Series([], dtype=float))
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


# -------------------------- rule_cash_in_regime ------------------------------


def _fake_daily_and_regimes():
    """Build a daily frame and matching regime frame for rule tests."""
    dates = pd.bdate_range("2000-01-03", periods=30)
    # VIX that rises monotonically so later days land in progressively higher
    # regimes under standard thresholds.
    vix = pd.DataFrame({
        "date": dates,
        "vix": np.linspace(15.0, 35.0, 30),
    })
    regimes = label_vix_regimes(vix, RegimeConfig())
    daily = pd.DataFrame({
        "date": dates,
        "port_ret_net": np.full(30, 0.01),  # 100 bps/day everywhere
        "long_ret": np.full(30, 0.015),
        "short_ret": np.full(30, -0.005),
    })
    return daily, regimes


def test_rule_zeros_target_regime_only():
    daily, regimes = _fake_daily_and_regimes()
    cash_rule = rule_cash_in_regime(daily, regimes, "high_vol")
    merged = daily.merge(
        regimes[["date", "regime"]], on="date", how="left"
    ).assign(rule=cash_rule.values)
    # On high_vol days: rule = 0
    hv = merged[merged["regime"] == "high_vol"]
    assert (hv["rule"] == 0.0).all(), "high_vol days should be zeroed"
    # On non-high_vol days: rule unchanged
    not_hv = merged[merged["regime"] != "high_vol"]
    np.testing.assert_allclose(
        not_hv["rule"].values, not_hv["port_ret_net"].values
    )


def test_rule_works_with_preattached_regime():
    """If the caller has already merged a `regime` column onto daily, the
    helper shouldn't blow up with merge-suffix column naming (KeyError)."""
    from krauss.regimes.vix_regimes import attach_regime
    daily, regimes = _fake_daily_and_regimes()
    pre_attached = attach_regime(daily, regimes)
    # Should not raise
    result = rule_cash_in_regime(pre_attached, regimes, "high_vol")
    assert len(result) == len(pre_attached)
