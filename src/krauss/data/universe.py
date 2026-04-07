"""
S&P 500 universe construction with strict no-lookahead monthly membership.

Rule:
    The month-end S&P 500 constituent list for month M determines the
    eligible trading universe for month M+1.

    For any trading day t in month M+1, a stock is eligible if and only if:
        1. It was an S&P 500 member as of the last calendar day of month M
           (per crsp.dsp500list spell data).
        2. It has sufficient data availability (checked downstream).

    The universe updates month-by-month. It is NOT frozen for an entire
    study period.

Example:
    Jan 31 membership -> eligible universe for all of February
    Feb 28 membership -> eligible universe for all of March
"""

import pandas as pd
import numpy as np


def build_membership_matrix(
    sp500_raw: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build the monthly membership panel from raw S&P 500 spell data.

    For each month-end in [start_date, end_date], check which PERMNOs are
    S&P 500 members on that date. Mark them as eligible for the following
    month (effective_month = month-end + 1 month).

    Parameters
    ----------
    sp500_raw : pd.DataFrame
        From wrds_extract.fetch_sp500_membership().
        Columns: permno, start, ending
    start_date : str
        Earliest month-end to evaluate (e.g. '1989-01-01').
    end_date : str
        Latest month-end to evaluate (e.g. '2015-12-31').

    Returns
    -------
    pd.DataFrame
        permno         : int
        month_end_date : datetime  — the month-end when membership was checked
        effective_month: Period[M] — the month this eligibility applies to
        is_member      : bool      — always True (table only contains members)
    """
    # Generate all month-end dates in range
    month_ends = pd.date_range(start=start_date, end=end_date, freq="ME")

    records = []
    for me_date in month_ends:
        # A stock is a member on me_date if any spell covers that date:
        #   spell.start <= me_date AND (spell.ending >= me_date OR ending is NaT)
        mask = (sp500_raw["start"] <= me_date) & (
            (sp500_raw["ending"] >= me_date) | sp500_raw["ending"].isna()
        )
        members = sp500_raw.loc[mask, "permno"].unique()

        # This membership snapshot governs the NEXT month
        effective = (me_date + pd.offsets.MonthBegin(1)).to_period("M")

        for p in members:
            records.append(
                {
                    "permno": int(p),
                    "month_end_date": me_date,
                    "effective_month": effective,
                    "is_member": True,
                }
            )

    membership = pd.DataFrame(records)
    return membership


def get_eligible_universe(
    membership: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> np.ndarray:
    """
    Return the set of eligible PERMNOs for a given trading date.

    Maps trade_date to its calendar month, then looks up which PERMNOs
    have is_member=True for that effective_month.

    Parameters
    ----------
    membership : pd.DataFrame
        Output of build_membership_matrix().
    trade_date : pd.Timestamp
        The trading date to query.

    Returns
    -------
    np.ndarray of int
        Eligible PERMNOs for that date.
    """
    month = trade_date.to_period("M")
    mask = membership["effective_month"] == month
    return membership.loc[mask, "permno"].values


def build_daily_eligibility(
    membership: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """
    Expand the monthly membership panel to a daily eligibility table.

    For each trading date, maps to its effective_month and joins to the
    membership panel to produce one row per eligible (date, permno) pair.

    Parameters
    ----------
    membership : pd.DataFrame
        Output of build_membership_matrix().
    trading_dates : pd.Series of datetime
        All trading dates from the return data.

    Returns
    -------
    pd.DataFrame
        date   : datetime
        permno : int
        One row per eligible stock-day.
    """
    dates_df = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    dates_df["effective_month"] = dates_df["date"].dt.to_period("M")

    eligible = dates_df.merge(
        membership[["effective_month", "permno"]],
        on="effective_month",
        how="inner",
    )
    eligible = (
        eligible[["date", "permno"]]
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )
    return eligible
