"""
Reusable analysis helpers for regime-conditional backtests.

This module exists so new models, scores, thresholds, or sub-period checks
don't have to re-implement the same plumbing that lives inside the three
regime notebooks. Import and call.

Typical usage:

    from krauss.backtest.ranking import rank_and_select
    from krauss.backtest.portfolio import build_daily_portfolios, aggregate_portfolio_returns
    from krauss.backtest.costs import compute_turnover, apply_transaction_costs
    from krauss.regimes.vix_regimes import RegimeConfig, label_vix_regimes
    from krauss.regimes.analysis import backtest_k10, regime_stats, bootstrap_sharpe_ci

    regimes = label_vix_regimes(vix)
    daily = backtest_k10(predictions, 'zcomp_ens1', returns)
    stats = regime_stats(daily, regimes)              # dict of per-regime metrics
    ci   = bootstrap_sharpe_ci(daily['port_ret_net']) # (lo, hi) 95% CI

The helpers are deliberately dumb: small, pure, each with one job. The
notebooks stitch them together for narrative flow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from krauss.backtest.ranking import rank_and_select
from krauss.backtest.portfolio import (
    build_daily_portfolios,
    aggregate_portfolio_returns,
)
from krauss.backtest.costs import compute_turnover, apply_transaction_costs
from krauss.regimes.vix_regimes import REGIME_ORDER, attach_regime


# ----------------------------- basic primitives ------------------------------


def sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
    rf_annual: float = 0.02,
) -> float:
    """Annualized Sharpe ratio of a daily-return series.

    Returns 0.0 if std is zero or the input is empty, rather than raising.
    """
    if returns is None or len(returns) == 0:
        return 0.0
    mean_ann = returns.mean() * periods_per_year
    std_ann = returns.std() * np.sqrt(periods_per_year)
    # Tolerance guard against floating-point noise (e.g. a constant series'
    # computed std can be ~1e-19 rather than exactly 0). 1e-12 comfortably
    # rejects that while still accepting any real-world Sharpe.
    if std_ann < 1e-12 or not np.isfinite(std_ann):
        return 0.0
    return (mean_ann - rf_annual) / std_ann


def cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score column `col` within each trading day (no lookahead).

    Standard helper so notebooks stop reimplementing the same groupby chain.
    """
    mu = df.groupby("date")[col].transform("mean")
    sigma = df.groupby("date")[col].transform("std")
    return (df[col] - mu) / sigma.replace(0, np.nan)


# ------------------------------ backtest wrapper -----------------------------


def run_backtest(
    predictions: pd.DataFrame,
    score_col: str,
    returns: pd.DataFrame,
    k: int = 10,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Full Phase-1-style backtest: rank -> build -> aggregate -> costs.

    Returns a daily DataFrame with columns including `port_ret`, `port_ret_net`,
    `long_ret`, `short_ret`, `cost`, and `date`. Matches the schema used by
    every composite notebook in this repo.
    """
    sel = rank_and_select(predictions, k=k, score_col=score_col)
    hold = build_daily_portfolios(sel, returns, k=k)
    daily = aggregate_portfolio_returns(hold)
    turn = compute_turnover(hold, k=k)
    daily = apply_transaction_costs(daily, turn, cost_bps)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def backtest_k10(
    predictions: pd.DataFrame,
    score_col: str,
    returns: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Convenience wrapper: run_backtest with k=10 (the choice Parts 1–3 adopt)."""
    return run_backtest(predictions, score_col, returns, k=10, cost_bps=cost_bps)


# ------------------------------ regime summaries -----------------------------


def regime_stats(
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    ret_col: str = "port_ret_net",
    periods_per_year: int = 252,
    rf_annual: float = 0.02,
) -> dict[str, Any]:
    """Compute overall + per-regime stats for a daily strategy series.

    Returns a flat dict like:
        {
            "overall_Sharpe": 1.92,
            "overall_ret_bps": 30.6,
            "low_vol_Sharpe": 2.42, "low_vol_ret_bps": 22.5,
                "low_vol_long_bps": 19.9, "low_vol_short_bps": -14.3,
                "low_vol_n_days": 3393,
            "mid_vol_...": ...,
            "high_vol_...": ...,
            "n_days_labeled": 5748,
        }

    Use `regime_stats_df` below for a multi-model comparison table.
    """
    d = attach_regime(daily, regimes).dropna(subset=["regime"]).copy()
    out: dict[str, Any] = {
        "overall_Sharpe": sharpe(d[ret_col], periods_per_year, rf_annual),
        "overall_ret_bps": d[ret_col].mean() * 10_000,
        "n_days_labeled": len(d),
    }
    for r in REGIME_ORDER:
        sub = d[d["regime"] == r]
        if not len(sub):
            continue
        out[f"{r}_Sharpe"] = sharpe(sub[ret_col], periods_per_year, rf_annual)
        out[f"{r}_ret_bps"] = sub[ret_col].mean() * 10_000
        out[f"{r}_long_bps"] = sub["long_ret"].mean() * 10_000
        out[f"{r}_short_bps"] = sub["short_ret"].mean() * 10_000
        out[f"{r}_n_days"] = len(sub)
    return out


def regime_stats_df(
    models: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
    *,
    ret_col: str = "port_ret_net",
    drop_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame from many model backtests.

    Parameters
    ----------
    models : dict mapping model label -> daily DataFrame (output of run_backtest)
    regimes : output of label_vix_regimes
    drop_window : optional (start_date, end_date) to exclude before summarizing,
        e.g. ('2008-09-01', '2009-03-31') for the crisis robustness check.

    Returns
    -------
    DataFrame indexed by model label with one row of stats per model.
    """
    rows = []
    for name, daily in models.items():
        if drop_window is not None:
            start, end = pd.Timestamp(drop_window[0]), pd.Timestamp(drop_window[1])
            mask = ~pd.to_datetime(daily["date"]).between(start, end)
            daily = daily[mask]
        row = regime_stats(daily, regimes, ret_col=ret_col)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# --------------------------- bootstrap confidence ----------------------------


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap CI on annualized Sharpe.

    Returns (point_estimate, ci_low, ci_high).

    Uses 1-day blocks (iid bootstrap) — fine for this strategy because daily
    returns have low autocorrelation. Switch to block bootstrap (block len
    ~10) if needed for a residual-autocorrelation defense in a later paper
    draft; the point estimate is unchanged either way.
    """
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = returns.values
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = sharpe(pd.Series(arr[idx]))
    point = sharpe(returns)
    tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, tail * 100))
    hi = float(np.percentile(boots, (1.0 - tail) * 100))
    return point, lo, hi


# ---------------------------- trading rule helpers ---------------------------


def rule_cash_in_regime(
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    target_regime: str,
    ret_col: str = "port_ret_net",
) -> pd.Series:
    """Return the daily P&L series of a rule that goes to cash on `target_regime` days.

    Trading on any day not in `target_regime` is unchanged; on `target_regime`
    days the return is zero. Cost of skipping is approximated as zero (no
    turnover on skip days), which slightly overstates the improvement when the
    next non-skip day involves rebuilding the portfolio. For a sizing-only
    analysis this is close enough.

    `daily` may or may not already have a `regime` column; this handles both.
    """
    if "regime" in daily.columns:
        d = daily
    else:
        d = attach_regime(daily, regimes)
    return pd.Series(
        np.where(d["regime"] == target_regime, 0.0, d[ret_col]),
        index=d.index,
    )
