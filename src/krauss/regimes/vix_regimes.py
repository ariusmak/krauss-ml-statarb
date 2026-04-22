"""
VIX-based regime labeling and regime-conditional backtesting.

This module supports Phase 2 extension analysis: does the optimal portfolio
size k depend on the volatility regime? Following Krauss et al. (2017),
who use VIX > 30 as a crisis indicator in their Table 4 factor regression,
we extend that threshold into a three-regime classification and test
regime-conditional portfolio sizing.

Regime definitions (fixed thresholds, not data-driven):
    - low_vol:  smoothed VIX < 20   (~ long-run median, calm markets)
    - mid_vol:  20 <= smoothed VIX <= 30  (elevated but not crisis)
    - high_vol: smoothed VIX > 30   (paper's crisis threshold, ~10% of days)

The smoothing (default: 5-day trailing mean of VIX close) reduces regime
whipsawing on single-day VIX spikes. The trailing window is strictly
backward-looking: the label for trading day t is computed using VIX
closes through day t-1, which is what a trader would have at signal time
on day t. No lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RegimeName = Literal["low_vol", "mid_vol", "high_vol"]
REGIME_ORDER: list[RegimeName] = ["low_vol", "mid_vol", "high_vol"]


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for VIX regime labeling.

    Parameters
    ----------
    low_threshold : float
        Smoothed VIX < this value -> low_vol regime. Default 20.0.
    high_threshold : float
        Smoothed VIX > this value -> high_vol regime. Default 30.0
        (matches Krauss et al. 2017 Table 4 crisis indicator).
    smoothing_window : int
        Number of trading days in the trailing mean applied to VIX before
        thresholding. Default 5. Use 1 for no smoothing (raw daily close).
    """

    low_threshold: float = 20.0
    high_threshold: float = 30.0
    smoothing_window: int = 5

    def __post_init__(self):
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < "
                f"high_threshold ({self.high_threshold})"
            )
        if self.smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be >= 1, got {self.smoothing_window}"
            )


def label_vix_regimes(
    vix: pd.DataFrame,
    config: RegimeConfig | None = None,
    *,
    date_col: str = "date",
    vix_col: str = "vix",
) -> pd.DataFrame:
    """Label each trading day with a VIX regime, using only past information.

    The regime for trading day t is determined by the trailing mean of VIX
    closes strictly before t (i.e., closes on days < t). This means the
    label Claude would assign at signal time on day t is causally correct:
    no VIX value from day t itself or later is used.

    Parameters
    ----------
    vix : pd.DataFrame
        Daily VIX data. Must contain columns `date_col` and `vix_col`.
    config : RegimeConfig, optional
        Threshold and smoothing configuration. Defaults to paper-aligned
        values: 20/30 thresholds, 5-day trailing mean.
    date_col, vix_col : str
        Column names in `vix`. Defaults match scripts/fetch_vix.py output.

    Returns
    -------
    pd.DataFrame
        Columns: date, vix, vix_smooth, regime.
        `vix_smooth` is the trailing (lookahead-free) average used for
        thresholding. `regime` is one of {"low_vol", "mid_vol", "high_vol"}
        or NaN for the first `smoothing_window` rows (insufficient history).
    """
    config = config or RegimeConfig()
    df = (
        vix[[date_col, vix_col]]
        .rename(columns={date_col: "date", vix_col: "vix"})
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    df["date"] = pd.to_datetime(df["date"])

    # Trailing mean STRICTLY excluding day t. shift(1) drops the current
    # day's VIX close out of the window, so the label is causal.
    df["vix_smooth"] = (
        df["vix"]
        .shift(1)
        .rolling(window=config.smoothing_window, min_periods=config.smoothing_window)
        .mean()
    )

    regime = pd.Series(pd.NA, index=df.index, dtype="object")
    mask_low = df["vix_smooth"] < config.low_threshold
    mask_high = df["vix_smooth"] > config.high_threshold
    mask_mid = (~mask_low) & (~mask_high) & df["vix_smooth"].notna()
    regime.loc[mask_low] = "low_vol"
    regime.loc[mask_mid] = "mid_vol"
    regime.loc[mask_high] = "high_vol"
    df["regime"] = regime

    return df


def attach_regime(
    trade_dates: pd.DataFrame | pd.Series,
    regime_df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Join regime labels onto a panel of trading dates.

    Because VIX is only quoted on NYSE/CBOE trading days and the strategy
    trades on the same calendar, the join is exact-match on date. Any trade
    date without a VIX observation (rare edge case — e.g., a day CBOE was
    closed but NYSE wasn't) will get NaN regime.

    Parameters
    ----------
    trade_dates : pd.DataFrame or pd.Series
        Must have a `date_col` column (or be a Series of dates).
    regime_df : pd.DataFrame
        Output of `label_vix_regimes`.
    date_col : str
        Date column name in `trade_dates`.

    Returns
    -------
    pd.DataFrame
        `trade_dates` with added columns: vix, vix_smooth, regime.
    """
    if isinstance(trade_dates, pd.Series):
        trade_dates = trade_dates.to_frame(name=date_col)

    left = trade_dates.copy()
    left[date_col] = pd.to_datetime(left[date_col])

    right = regime_df[["date", "vix", "vix_smooth", "regime"]].copy()
    right["date"] = pd.to_datetime(right["date"])

    merged = left.merge(right, left_on=date_col, right_on="date", how="left")
    # Drop duplicate date col if names collide
    if date_col != "date" and "date" in merged.columns:
        merged = merged.drop(columns=["date"])
    return merged


def regime_coverage(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table: trading day counts and share by regime.

    Useful as a diagnostic before running regime-conditional backtests.
    Rare regimes (< ~5% of days) make per-regime return estimates noisy.
    """
    counts = regime_df["regime"].value_counts(dropna=False).rename("n_days")
    total = int(counts.sum())

    # Build the ordered index: named regimes in canonical order, then any
    # NaN bucket last. We can't use reindex() with NaN keys — it looks up by
    # value equality and NaN != NaN, so NaN rows get dropped silently.
    rows = []
    for r in REGIME_ORDER:
        n = int(counts.get(r, 0))
        rows.append((r, n))
    nan_count = int(regime_df["regime"].isna().sum())
    if nan_count:
        rows.append(("unlabeled", nan_count))

    out = pd.DataFrame(rows, columns=["regime", "n_days"]).set_index("regime")
    out["share"] = out["n_days"] / total if total else 0.0
    return out


def split_returns_by_regime(
    daily_returns: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> dict[RegimeName, pd.DataFrame]:
    """Partition a daily strategy return series into per-regime frames.

    The returned dict maps regime name -> DataFrame of rows from
    `daily_returns` where that regime was active. Dates with unlabeled
    regime (NaN) are dropped.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Output of the backtest pipeline (e.g., `apply_transaction_costs`).
        Must contain `date_col` and the usual return columns.
    regime_df : pd.DataFrame
        Output of `label_vix_regimes`.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    merged = attach_regime(daily_returns, regime_df, date_col=date_col)
    out: dict[RegimeName, pd.DataFrame] = {}
    for r in REGIME_ORDER:
        sub = merged[merged["regime"] == r].copy()
        if len(sub):
            out[r] = sub
    return out
