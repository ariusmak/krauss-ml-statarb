"""
Portfolio construction: equal-weight, dollar-neutral, daily rebalance.

For each trading day:
    - Long k stocks, each weighted +1/k
    - Short k stocks, each weighted -1/k
    - Dollar neutral: sum of weights = 0
    - Portfolio return = mean(long returns) - mean(short returns)
"""

import pandas as pd
import numpy as np


def build_daily_portfolios(
    selections: pd.DataFrame,
    returns: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    Construct daily portfolio returns from ranked selections.

    Parameters
    ----------
    selections : pd.DataFrame
        Output of ranking.rank_and_select().
        Columns: date, permno, rank, side, score
        'date' is the signal/ranking date (day t).
    returns : pd.DataFrame
        Full return panel. Columns: permno, date, ret
    k : int
        Expected number of stocks on each side.

    Returns
    -------
    pd.DataFrame
        One row per stock-day holding:
        date         : signal date (day t)
        next_date    : return date (day t+1)
        permno       : int
        side         : 'long' or 'short'
        weight       : +1/k or -1/k
        next_day_ret : realized return on t+1
        contrib      : weight * next_day_ret
    """
    # Get next-day returns: for signal date t, we need return on t+1
    ret = returns[["permno", "date", "ret"]].sort_values(["permno", "date"])
    ret["next_date"] = ret.groupby("permno")["date"].shift(-1)
    ret["next_day_ret"] = ret.groupby("permno")["ret"].shift(-1)
    ret = ret.dropna(subset=["next_day_ret"])
    ret["next_date"] = ret["next_date"].astype("datetime64[ns]")

    # Merge selections with next-day returns
    holdings = selections.merge(
        ret[["permno", "date", "next_date", "next_day_ret"]],
        on=["date", "permno"],
        how="inner",
    )

    # Equal weights
    holdings["weight"] = np.where(
        holdings["side"] == "long", 1.0 / k, -1.0 / k
    )
    holdings["contrib"] = holdings["weight"] * holdings["next_day_ret"]

    holdings = holdings.sort_values(["date", "side", "permno"])
    holdings = holdings.reset_index(drop=True)

    return holdings


def aggregate_portfolio_returns(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stock-level holdings to daily portfolio returns.

    Parameters
    ----------
    holdings : pd.DataFrame
        Output of build_daily_portfolios().

    Returns
    -------
    pd.DataFrame
        date         : signal date
        next_date    : return date
        port_ret     : total portfolio return
        long_ret     : long leg return (mean of long stock returns)
        short_ret    : short leg return (mean of short stock returns)
        n_long       : count of long positions
        n_short      : count of short positions
    """
    long = holdings[holdings["side"] == "long"]
    short = holdings[holdings["side"] == "short"]

    long_agg = long.groupby("date").agg(
        long_ret=("next_day_ret", "mean"),
        n_long=("permno", "count"),
        next_date=("next_date", "first"),
    )

    short_agg = short.groupby("date").agg(
        short_ret=("next_day_ret", "mean"),
        n_short=("permno", "count"),
    )

    daily = long_agg.join(short_agg, how="inner")
    # Portfolio return = long_ret - short_ret (dollar neutral L/S)
    daily["port_ret"] = daily["long_ret"] - daily["short_ret"]
    daily = daily.reset_index()

    return daily[["date", "next_date", "port_ret", "long_ret", "short_ret",
                   "n_long", "n_short"]]
