"""Tests for krauss.regimes.vix_regimes.

Focus: the no-lookahead invariant (the most important property for a paper
claim to be defensible) and basic correctness of the labeling rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krauss.regimes.vix_regimes import (
    RegimeConfig,
    attach_regime,
    label_vix_regimes,
    regime_coverage,
    split_returns_by_regime,
)


def _synth_vix(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Synthetic VIX: sinusoid ranging roughly 10 to 40 over `n` days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    base = 22 + 12 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = rng.normal(0, 1.5, n)
    return pd.DataFrame({"date": dates, "vix": base + noise})


def test_config_validation():
    with pytest.raises(ValueError, match="low_threshold"):
        RegimeConfig(low_threshold=30, high_threshold=20)
    with pytest.raises(ValueError, match="smoothing_window"):
        RegimeConfig(smoothing_window=0)


def test_no_lookahead_invariant():
    """Perturbing VIX on day t must not change the regime label on day t."""
    vix = _synth_vix()
    base = label_vix_regimes(vix)

    # Perturb every row's VIX by a huge amount and re-label.
    # If the labeler is causal, the label on day t depends only on days < t,
    # so flipping VIX on day t cannot change day-t's own label.
    perturbed = vix.copy()
    perturbed["vix"] = perturbed["vix"] + 100.0  # force above high_threshold
    bumped = label_vix_regimes(perturbed)

    # Day t's label in `bumped` is computed from days < t in `perturbed`,
    # which ARE different from `vix`. So most labels will change.
    # But this test is about a different invariant: bump ONLY day t, check
    # label on day t is unchanged.
    for i in range(5, len(vix)):  # skip first rows with NaN regime
        perturbed_one = vix.copy()
        perturbed_one.loc[i, "vix"] = 500.0  # absurd spike on exactly day i
        relabel = label_vix_regimes(perturbed_one)
        assert relabel.loc[i, "regime"] == base.loc[i, "regime"], (
            f"Day {i} label changed after perturbing VIX on same day"
        )


def test_future_vix_does_not_affect_past_labels():
    """Appending future VIX rows must not change historical regime labels."""
    vix = _synth_vix(n=30)
    base = label_vix_regimes(vix).set_index("date")["regime"]

    # Append a giant future spike and relabel.
    extra = pd.DataFrame({
        "date": pd.bdate_range(vix["date"].iloc[-1] + pd.Timedelta(days=1), periods=5),
        "vix": [999.0] * 5,
    })
    extended = pd.concat([vix, extra], ignore_index=True)
    relabel = label_vix_regimes(extended).set_index("date")["regime"]

    common = base.index.intersection(relabel.index)
    pd.testing.assert_series_equal(
        base.loc[common], relabel.loc[common], check_names=False,
    )


def test_thresholds_and_smoothing():
    """Basic rule check: constant VIX below/between/above thresholds -> fixed regime."""
    for level, expected in [(10.0, "low_vol"), (25.0, "mid_vol"), (40.0, "high_vol")]:
        vix = pd.DataFrame({
            "date": pd.bdate_range("2000-01-03", periods=20),
            "vix": [level] * 20,
        })
        labeled = label_vix_regimes(vix)
        # First `smoothing_window` rows are NaN (insufficient trailing history).
        # From row `smoothing_window` onward, all labels equal expected.
        cfg = RegimeConfig()
        tail = labeled["regime"].iloc[cfg.smoothing_window:]
        assert (tail == expected).all(), (
            f"VIX={level} should yield {expected}, got {tail.unique()}"
        )


def test_first_n_labels_are_nan():
    """With smoothing_window=5, the first 5 labels must be NaN."""
    vix = _synth_vix()
    labeled = label_vix_regimes(vix, RegimeConfig(smoothing_window=5))
    assert labeled["regime"].iloc[:5].isna().all()
    assert labeled["regime"].iloc[5:].notna().all()


def test_regime_coverage_sums_to_total():
    vix = _synth_vix(n=100)
    labeled = label_vix_regimes(vix)
    cov = regime_coverage(labeled)
    assert cov["n_days"].sum() == len(labeled)
    assert abs(cov["share"].sum() - 1.0) < 1e-9


def test_attach_regime_exact_match():
    vix = _synth_vix(n=30)
    labeled = label_vix_regimes(vix)
    trades = vix[["date"]].iloc[10:20].copy()
    attached = attach_regime(trades, labeled)
    assert len(attached) == 10
    assert "regime" in attached.columns
    assert "vix" in attached.columns
    assert "vix_smooth" in attached.columns


def test_split_returns_by_regime():
    vix = pd.DataFrame({
        "date": pd.bdate_range("2000-01-03", periods=20),
        "vix": [15.0] * 10 + [35.0] * 10,
    })
    labeled = label_vix_regimes(vix)
    daily = pd.DataFrame({
        "date": vix["date"],
        "port_ret": np.arange(20) * 0.001,
    })
    splits = split_returns_by_regime(daily, labeled)
    # Days 0-4 NaN, days 5-9 low_vol (VIX=15), days 10-19 transition into high_vol.
    # The exact split depends on smoothing; just sanity-check non-empty and disjoint.
    total = sum(len(v) for v in splits.values())
    labeled_days = labeled["regime"].notna().sum()
    assert total == labeled_days
