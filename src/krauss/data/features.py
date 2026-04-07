"""
Feature generation: 31 lagged return features per the paper.

Features:
    R1, R2, ..., R20   — daily simple returns over 1..20 day lookbacks
    R40, R60, R80, R100, R120, R140, R160, R180, R200, R220, R240
                        — multi-period simple returns

Definition:
    R_{t,m} = P_t / P_{t-m} - 1

    where P is a return-consistent (total-return) price level reconstructed
    from daily holding-period returns.

The longest lookback is 240 trading days, which is why the paper states
that the first 240 days of each training window are consumed by feature
construction.
"""

import pandas as pd
import numpy as np

# All lag periods used in the paper
DAILY_LAGS = list(range(1, 21))  # R1 .. R20
MULTI_PERIOD_LAGS = list(range(40, 241, 20))  # R40, R60, ..., R240
ALL_LAGS = DAILY_LAGS + MULTI_PERIOD_LAGS  # 31 features total
FEATURE_COLS = [f"R{lag}" for lag in ALL_LAGS]


def _build_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a cumulative total-return price index from daily returns.

    Starting from P_0 = 1 for each PERMNO, compound forward:
        P_t = P_{t-1} * (1 + ret_t)

    Parameters
    ----------
    returns : pd.DataFrame
        Columns: permno, date, ret
        Must be sorted by (permno, date).

    Returns
    -------
    pd.DataFrame
        permno, date, ret, price_idx
    """
    df = returns.copy()
    df["price_idx"] = df.groupby("permno")["ret"].transform(
        lambda r: (1 + r).cumprod()
    )
    return df


def compute_lagged_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 31 lagged return features for each (permno, date).

    R_{t,m} = price_idx_t / price_idx_{t-m} - 1

    Parameters
    ----------
    returns : pd.DataFrame
        Columns: permno, date, ret
        Sorted by (permno, date). Should be the full return panel
        (not just eligible dates) so lookbacks can reach back far enough.

    Returns
    -------
    pd.DataFrame
        permno, date, R1, R2, ..., R20, R40, R60, ..., R240
        Rows with insufficient history for all 31 features are dropped.
    """
    df = _build_price_index(returns)

    # Compute each lagged return using the price index
    for lag in ALL_LAGS:
        col = f"R{lag}"
        df[col] = df.groupby("permno")["price_idx"].transform(
            lambda p: p / p.shift(lag) - 1
        )

    # Keep only feature columns + identifiers
    feature_cols = [f"R{lag}" for lag in ALL_LAGS]
    result = df[["permno", "date"] + feature_cols].copy()

    # Drop rows where any feature is missing (insufficient lookback)
    result = result.dropna(subset=feature_cols).reset_index(drop=True)

    return result
