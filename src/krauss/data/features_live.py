"""
Live-update feature construction.

This module produces the same 31 lagged-return features the Phase 2 models
were trained on, from the unified returns panel
(``data/processed/returns_unified.parquet``) instead of the legacy training
parquets.  It is the entry point used by the live-refresh path:

    returns_unified.parquet  ──>  features_live.compute_features(...)
                                          │
                                          ▼
                                  31-feature panel at any date range,
                                  keyed by (date, infocode), ready to feed
                                  into the frozen period-33 models.

The numerical recipe is the same as ``krauss.data.features.compute_lagged_returns``
but generalised to operate on an ``id_col`` parameter (``infocode`` here)
and to take an explicit date range.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from krauss.data.features import ALL_LAGS, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[3]
RETURNS_UNIFIED_PATH = ROOT / "data" / "processed" / "returns_unified.parquet"

# Lookback room needed before the requested start date so that R240 is
# computable on the first requested feature row.  Paper / Phase 2 use
# 240 trading days.
MAX_LOOKBACK = max(ALL_LAGS)


def load_unified_returns(path: Path | None = None) -> pd.DataFrame:
    """Read the unified daily-returns parquet and normalise dtypes."""
    df = pd.read_parquet(path or RETURNS_UNIFIED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["infocode"] = df["infocode"].astype("Int64")
    return df[["date", "infocode", "ticker", "ret", "source"]]


def compute_features(
    returns: pd.DataFrame | None = None,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    id_col: str = "infocode",
) -> pd.DataFrame:
    """Compute the 31 lagged-return features for a date range.

    Parameters
    ----------
    returns : pd.DataFrame, optional
        Daily-return panel with at least ``[id_col, 'date', 'ret']``.  If
        omitted, loads from ``data/processed/returns_unified.parquet``.
    start, end : Timestamp-like, optional
        Restrict the OUTPUT date range.  Lookback rows before ``start`` are
        used internally to compute features for ``start`` itself; they are
        dropped from the output.  Default: full panel.
    id_col : str, default 'infocode'
        Column to group by.  ``'infocode'`` for the unified panel,
        ``'permno'`` for legacy CRSP usage.

    Returns
    -------
    pd.DataFrame
        Columns: ``[id_col, 'date', 'R1', 'R2', …, 'R240']``.  Rows where
        any feature is NaN (insufficient history) are dropped.
    """
    if returns is None:
        returns = load_unified_returns()

    # Need lookback room before the requested ``start`` so the first
    # requested row has all 31 features defined.  Use trading days, not
    # calendar days -- conservatively pad by 380 calendar days for the
    # 240-trading-day longest lookback.
    if start is not None:
        start = pd.Timestamp(start)
        lookback_floor = start - pd.Timedelta(days=int(MAX_LOOKBACK * 1.6))
    else:
        lookback_floor = None
    if end is not None:
        end = pd.Timestamp(end)

    df = returns.copy()
    df = df.sort_values([id_col, "date"]).reset_index(drop=True)

    # Optionally pre-filter to reduce work.  Always keep the full lookback.
    if lookback_floor is not None:
        df = df[df["date"] >= lookback_floor]
    if end is not None:
        df = df[df["date"] <= end]

    # Cumulative total-return price index per id
    df["price_idx"] = df.groupby(id_col)["ret"].transform(
        lambda r: (1.0 + r).cumprod()
    )

    # Each lag: R_{t,m} = P_t / P_{t-m} - 1
    for lag in ALL_LAGS:
        col = f"R{lag}"
        df[col] = df.groupby(id_col)["price_idx"].transform(
            lambda p, _lag=lag: p / p.shift(_lag) - 1.0
        )

    # Drop rows missing any feature (insufficient history)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Trim to the requested OUTPUT range
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]

    cols = [id_col, "date"] + FEATURE_COLS
    return df[cols].reset_index(drop=True)


def trade_dates_after(unified_returns: pd.DataFrame,
                      after: pd.Timestamp | str) -> list[pd.Timestamp]:
    """Sorted unique trading dates strictly after ``after``."""
    after = pd.Timestamp(after)
    dates = unified_returns.loc[unified_returns["date"] > after, "date"]
    return sorted(dates.dt.normalize().unique().tolist())


def feature_columns() -> list[str]:
    """The 31 feature names, in canonical order."""
    return list(FEATURE_COLS)
