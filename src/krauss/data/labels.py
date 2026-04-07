"""
Label generation for Phase 1 and Phase 2.

Phase 1 target (binary classification):
    y_binary = 1 if stock's next-day return > next-day cross-sectional median
               0 otherwise

Phase 2 target (excess return):
    u_excess = next_day_return - next_day_cross_sectional_median_return

Both targets are computed relative to the cross-sectional median of the
eligible universe on each day. The eligible universe is determined by the
no-lookahead monthly membership rule.

Important alignment:
    Features at date t use information through t.
    Labels at date t use the return realized on t+1.
    The cross-sectional median is computed on t+1 across eligible stocks.
"""

import pandas as pd
import numpy as np


def compute_labels(
    returns: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute classification and regression labels for eligible stocks.

    Parameters
    ----------
    returns : pd.DataFrame
        Full return panel. Columns: permno, date, ret
    eligible : pd.DataFrame
        Daily eligibility table. Columns: date, permno
        One row per eligible stock-day.

    Returns
    -------
    pd.DataFrame
        date             : the feature date (day t)
        permno           : int
        next_day_date    : the date of the realized return (day t+1)
        next_day_ret     : stock's return on t+1
        next_day_median  : cross-sectional median return on t+1
        u_excess         : next_day_ret - next_day_median
        y_binary         : 1 if u_excess > 0, else 0
    """
    # Get next-day return for each (permno, date) pair
    ret = returns[["permno", "date", "ret"]].copy()
    ret = ret.sort_values(["permno", "date"]).reset_index(drop=True)

    # Shift return back by one day within each stock to get next-day return
    ret["next_day_ret"] = ret.groupby("permno")["ret"].shift(-1)
    ret["next_day_date"] = ret.groupby("permno")["date"].shift(-1)

    # Drop rows with no next-day return
    ret = ret.dropna(subset=["next_day_ret"])
    ret["next_day_date"] = ret["next_day_date"].astype("datetime64[ns]")

    # Restrict to eligible stock-days
    # 'date' here is the feature date t; the stock must be eligible on t
    labels = eligible.merge(ret, on=["date", "permno"], how="inner")

    # Compute cross-sectional median on next_day_date across eligible stocks
    # A stock's next-day must also be eligible for it to count in the median
    eligible_next = eligible.rename(columns={"date": "next_day_date"})
    labels = labels.merge(
        eligible_next, on=["next_day_date", "permno"], how="inner"
    )

    median_by_day = (
        labels.groupby("next_day_date")["next_day_ret"]
        .median()
        .rename("next_day_median")
    )
    labels = labels.merge(median_by_day, on="next_day_date", how="left")

    # Excess return and binary label
    labels["u_excess"] = labels["next_day_ret"] - labels["next_day_median"]
    labels["y_binary"] = (labels["u_excess"] > 0).astype(int)

    labels = labels[
        ["date", "permno", "next_day_date", "next_day_ret",
         "next_day_median", "u_excess", "y_binary"]
    ].sort_values(["date", "permno"]).reset_index(drop=True)

    return labels
