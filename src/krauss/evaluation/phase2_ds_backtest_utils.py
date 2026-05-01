"""
Utilities for Datastream Phase 2 notebook backtests.

These helpers are designed for exploratory notebooks that compare score
families on the Datastream US-only extension sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from krauss.backtest.costs import apply_transaction_costs, compute_turnover
from krauss.backtest.portfolio import aggregate_portfolio_returns, build_daily_portfolios
from krauss.backtest.ranking import rank_and_select

ROOT = Path(__file__).resolve().parents[3]
DS_DIR = ROOT / "data" / "datastream"
K = 10
COST_BPS = 5.0
FAMILIES = ["dnn", "xgb", "rf", "ens1"]
FAMILY_LABELS = {"dnn": "DNN", "xgb": "XGB", "rf": "RF", "ens1": "ENS1"}


def load_phase2_ds_data():
    """Load Datastream Phase 2 predictions and returns."""
    pred = pd.read_parquet(DS_DIR / "predictions_phase2_ds.parquet")
    returns = pd.read_parquet(DS_DIR / "ds_daily_returns_usonly.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    returns["date"] = pd.to_datetime(returns["date"])
    # Normalize Datastream identifiers to the CRSP-style column name expected
    # by the shared backtest engine.
    if "permno" not in pred.columns and "infocode" in pred.columns:
        pred["permno"] = pred["infocode"]
    if "permno" not in returns.columns and "infocode" in returns.columns:
        returns["permno"] = returns["infocode"]
    return pred, returns


def add_zscore_scores(pred: pd.DataFrame, families: list[str] | None = None) -> pd.DataFrame:
    """Add 0.5*z(P) + 0.5*z(U) scores per family, cross-sectionally by date."""
    families = families or FAMILIES
    out = pred.copy()
    for family in families:
        p_col = f"p_{family}"
        u_col = f"u_{family}"
        p_z = out.groupby("date")[p_col].transform(_safe_zscore)
        u_z = out.groupby("date")[u_col].transform(_safe_zscore)
        out[f"score_z_{family}"] = 0.5 * p_z + 0.5 * u_z
    return out


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def run_score_backtest(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    score_col: str,
    k: int = K,
    cost_bps: float = COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a standard rank-based backtest and attach period IDs to the daily frame."""
    sel = rank_and_select(predictions, k=k, score_col=score_col)
    hold = build_daily_portfolios(sel, returns, k=k)
    daily = aggregate_portfolio_returns(hold)
    turn = compute_turnover(hold, k=k)
    daily = apply_transaction_costs(daily, turn, cost_bps)
    daily["date"] = pd.to_datetime(daily["date"])
    date_periods = predictions[["date", "period_id"]].drop_duplicates()
    daily = daily.merge(date_periods, on="date", how="left")
    return sel, hold, daily


def gated_rank_and_select(
    predictions: pd.DataFrame,
    p_col: str,
    u_col: str,
    k: int = K,
    threshold: float = 0.03,
) -> pd.DataFrame:
    """
    Rank by U but gate long/short candidate pools by P confidence.

    Long pool:  P > 0.5 + threshold, rank by U descending, take top k
    Short pool: P < 0.5 - threshold, rank by U ascending, take bottom k
    """
    df = predictions[["date", "permno", "period_id", p_col, u_col]].dropna().copy()
    records: list[dict] = []

    for d, day in df.groupby("date", sort=True):
        long_pool = day[day[p_col] > 0.5 + threshold].sort_values(u_col, ascending=False).head(k)
        short_pool = day[day[p_col] < 0.5 - threshold].sort_values(u_col, ascending=True).head(k)

        for rank, row in enumerate(long_pool.itertuples(index=False), start=1):
            records.append(
                {
                    "date": d,
                    "permno": row.permno,
                    "period_id": row.period_id,
                    "rank": rank,
                    "side": "long",
                    "score": getattr(row, u_col),
                }
            )
        for rank, row in enumerate(short_pool.itertuples(index=False), start=1):
            records.append(
                {
                    "date": d,
                    "permno": row.permno,
                    "period_id": row.period_id,
                    "rank": rank,
                    "side": "short",
                    "score": getattr(row, u_col),
                }
            )

    return pd.DataFrame(records)


def run_gated_backtest(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    p_col: str,
    u_col: str,
    k: int = K,
    threshold: float = 0.03,
    cost_bps: float = COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the gated-U backtest and attach period IDs."""
    sel = gated_rank_and_select(predictions, p_col=p_col, u_col=u_col, k=k, threshold=threshold)
    if sel.empty:
        hold = pd.DataFrame(columns=["date", "next_date", "port_ret", "long_ret", "short_ret", "n_long", "n_short"])
        daily = pd.DataFrame(columns=["date", "next_date", "port_ret", "long_ret", "short_ret", "n_long", "n_short", "turnover", "cost", "port_ret_net", "period_id"])
        return sel, hold, daily
    hold = build_daily_portfolios(sel, returns, k=k)
    daily = aggregate_portfolio_returns(hold)
    turn = compute_turnover(hold, k=k)
    daily = apply_transaction_costs(daily, turn, cost_bps)
    daily["date"] = pd.to_datetime(daily["date"])
    date_periods = predictions[["date", "period_id"]].drop_duplicates()
    daily = daily.merge(date_periods, on="date", how="left")
    return sel, hold, daily


def summary_stats(daily: pd.DataFrame) -> dict:
    """Compute headline daily/annualized stats for pre- and post-cost returns."""
    if daily.empty:
        return {
            "Daily Pre": np.nan,
            "Daily Post": np.nan,
            "Ann Pre": np.nan,
            "Ann Post": np.nan,
            "Sharpe Pre": np.nan,
            "Sharpe Post": np.nan,
            "Days": 0,
        }

    pre = daily["port_ret"]
    post = daily["port_ret_net"]
    return {
        "Daily Pre": pre.mean(),
        "Daily Post": post.mean(),
        "Ann Pre": pre.mean() * 252,
        "Ann Post": post.mean() * 252,
        "Sharpe Pre": sharpe(pre),
        "Sharpe Post": sharpe(post),
        "Days": int(len(daily)),
    }


def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe using mean/std * sqrt(252) with zero risk-free rate."""
    std = returns.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return returns.mean() / std * np.sqrt(252)


def per_period_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-period pre/post daily and annualized returns plus Sharpe."""
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "period_id",
                "start",
                "end",
                "days",
                "daily_pre",
                "daily_post",
                "ann_pre",
                "ann_post",
                "sharpe_pre",
                "sharpe_post",
            ]
        )

    rows = []
    for pid, grp in daily.groupby("period_id", sort=True):
        grp = grp.sort_values("date")
        pre = grp["port_ret"]
        post = grp["port_ret_net"]
        rows.append(
            {
                "period_id": int(pid),
                "start": grp["date"].min().date(),
                "end": grp["date"].max().date(),
                "days": int(len(grp)),
                "daily_pre": pre.mean(),
                "daily_post": post.mean(),
                "ann_pre": pre.mean() * 252,
                "ann_post": post.mean() * 252,
                "sharpe_pre": sharpe(pre),
                "sharpe_post": sharpe(post),
            }
        )
    return pd.DataFrame(rows)


def gated_activity_stats(daily: pd.DataFrame) -> dict:
    """Trading-day coverage and average long/short counts for gated strategies."""
    if daily.empty:
        return {
            "Trading Days": 0,
            "Mean Long/Day": np.nan,
            "Mean Short/Day": np.nan,
            "Mean Long/Active": np.nan,
            "Mean Short/Active": np.nan,
        }

    active = daily[(daily["n_long"] > 0) & (daily["n_short"] > 0)]
    return {
        "Trading Days": int(len(active)),
        "Mean Long/Day": daily["n_long"].mean(),
        "Mean Short/Day": daily["n_short"].mean(),
        "Mean Long/Active": active["n_long"].mean() if len(active) else np.nan,
        "Mean Short/Active": active["n_short"].mean() if len(active) else np.nan,
    }
