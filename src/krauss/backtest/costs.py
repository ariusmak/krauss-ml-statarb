"""
Transaction cost modeling.

Paper convention: 5 basis points per half-turn.

A half-turn is one side of a trade (either a buy or a sell).
A full round-trip (buy + sell) costs 10 bps.

Daily cost = turnover * cost_per_half_turn
where turnover is measured as the sum of absolute weight changes.
"""

import pandas as pd
import numpy as np


def compute_turnover(holdings: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Compute daily turnover from the holdings table.

    Turnover on day t = sum of |weight changes| from day t-1 to day t.
    On the first day, turnover = sum of |initial weights| (full portfolio build).

    Parameters
    ----------
    holdings : pd.DataFrame
        Output of portfolio.build_daily_portfolios().
        Columns include: date, permno, weight
    k : int
        Number of stocks per side.

    Returns
    -------
    pd.DataFrame
        date, turnover (sum of absolute weight changes)
    """
    # Build a date x permno weight matrix
    dates = sorted(holdings["date"].unique())

    turnover_records = []
    prev_weights = {}

    for d in dates:
        day_holdings = holdings[holdings["date"] == d]
        curr_weights = dict(zip(day_holdings["permno"], day_holdings["weight"]))

        # All permnos that appear in either day
        all_permnos = set(curr_weights.keys()) | set(prev_weights.keys())

        daily_turnover = sum(
            abs(curr_weights.get(p, 0.0) - prev_weights.get(p, 0.0))
            for p in all_permnos
        )

        turnover_records.append({"date": d, "turnover": daily_turnover})
        prev_weights = curr_weights

    return pd.DataFrame(turnover_records)


def apply_transaction_costs(
    daily_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_bps_per_half_turn: float = 5.0,
) -> pd.DataFrame:
    """
    Apply transaction costs to daily portfolio returns.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Output of portfolio.aggregate_portfolio_returns().
        Must have columns: date, port_ret
    turnover : pd.DataFrame
        Output of compute_turnover(). Columns: date, turnover
    cost_bps_per_half_turn : float
        Cost in basis points per half-turn (default 5).

    Returns
    -------
    pd.DataFrame
        Same as daily_returns with added columns:
        turnover, cost, port_ret_net
    """
    cost_frac = cost_bps_per_half_turn / 10_000

    df = daily_returns.merge(turnover, on="date", how="left")
    df["turnover"] = df["turnover"].fillna(0.0)
    df["cost"] = df["turnover"] * cost_frac
    df["port_ret_net"] = df["port_ret"] - df["cost"]

    return df
