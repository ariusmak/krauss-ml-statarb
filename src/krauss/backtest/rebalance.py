"""
Rebalance logic and turnover diagnostics.

The paper uses daily rebalancing: each day, the portfolio is rebuilt
from scratch based on new rankings. This module provides utilities
to analyze turnover patterns and position changes.
"""

import pandas as pd
import numpy as np


def compute_position_changes(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Track which stocks enter and exit the portfolio each day.

    Parameters
    ----------
    holdings : pd.DataFrame
        Output of portfolio.build_daily_portfolios().
        Columns: date, permno, side

    Returns
    -------
    pd.DataFrame
        date, n_new_long, n_new_short, n_exit_long, n_exit_short,
        n_stay_long, n_stay_short, n_side_switch
    """
    dates = sorted(holdings["date"].unique())
    records = []
    prev_long = set()
    prev_short = set()

    for d in dates:
        day = holdings[holdings["date"] == d]
        curr_long = set(day.loc[day["side"] == "long", "permno"])
        curr_short = set(day.loc[day["side"] == "short", "permno"])

        new_long = curr_long - prev_long - prev_short
        new_short = curr_short - prev_short - prev_long
        exit_long = prev_long - curr_long - curr_short
        exit_short = prev_short - curr_short - curr_long
        stay_long = curr_long & prev_long
        stay_short = curr_short & prev_short
        # Stocks that switched sides
        long_to_short = curr_short & prev_long
        short_to_long = curr_long & prev_short

        records.append({
            "date": d,
            "n_new_long": len(new_long),
            "n_new_short": len(new_short),
            "n_exit_long": len(exit_long),
            "n_exit_short": len(exit_short),
            "n_stay_long": len(stay_long),
            "n_stay_short": len(stay_short),
            "n_side_switch": len(long_to_short) + len(short_to_long),
        })

        prev_long = curr_long
        prev_short = curr_short

    return pd.DataFrame(records)
