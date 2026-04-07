"""
Daily returns construction with delisting-return adjustment.

The paper uses Datastream total-return indices. Our WRDS analogue is the
CRSP daily stock file `ret` field, which already includes dividends
(holding-period return). We adjust for delistings to avoid survivorship bias.

Delisting adjustment follows the standard CRSP convention:
    adjusted_ret = (1 + ret) * (1 + dlret) - 1
on the delisting date. If ret is missing on that date, we use dlret alone.
"""

import pandas as pd
import numpy as np


def adjust_for_delistings(
    daily: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge delisting returns into the daily return series.

    Parameters
    ----------
    daily : pd.DataFrame
        Output of wrds_extract.fetch_daily_stock_data().
        Must have columns: permno, date, ret
    delist : pd.DataFrame
        Output of wrds_extract.fetch_delisting_returns().
        Columns: permno, dlstdt, dlret, dlstcd

    Returns
    -------
    pd.DataFrame
        Same as daily, with ret adjusted on delisting dates.
    """
    df = daily.copy()

    # Merge delisting info onto matching permno + date
    delist_merge = delist[["permno", "dlstdt", "dlret"]].rename(
        columns={"dlstdt": "date"}
    )
    df = df.merge(delist_merge, on=["permno", "date"], how="left")

    # Where we have a delisting return, adjust
    has_dlret = df["dlret"].notna()
    has_ret = df["ret"].notna()

    # Case 1: both ret and dlret exist
    both = has_dlret & has_ret
    df.loc[both, "ret"] = (1 + df.loc[both, "ret"]) * (1 + df.loc[both, "dlret"]) - 1

    # Case 2: dlret exists but ret is missing — use dlret as the return
    dlret_only = has_dlret & ~has_ret
    df.loc[dlret_only, "ret"] = df.loc[dlret_only, "dlret"]

    df = df.drop(columns=["dlret"])
    return df


def build_return_panel(
    daily: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a clean daily return panel with delisting adjustment.

    Parameters
    ----------
    daily : pd.DataFrame
        Raw CRSP daily data.
    delist : pd.DataFrame
        Raw CRSP delisting data.

    Returns
    -------
    pd.DataFrame
        permno, date, ret (delisting-adjusted)
        Sorted by permno, date. Missing returns dropped.
    """
    df = adjust_for_delistings(daily, delist)
    df = df[["permno", "date", "ret"]].dropna(subset=["ret"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    return df
