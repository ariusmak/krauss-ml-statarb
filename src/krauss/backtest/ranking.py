"""
Cross-sectional ranking and top-k / bottom-k selection.

For each trading day:
    1. Take model probability outputs (p_hat) for all eligible stocks.
    2. Rank descending.
    3. Select top k (long candidates) and bottom k (short candidates).
"""

import pandas as pd
import numpy as np


def rank_and_select(
    predictions: pd.DataFrame,
    k: int,
    score_col: str = "p_hat",
) -> pd.DataFrame:
    """
    Rank stocks cross-sectionally and select top-k / bottom-k.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: date, permno, {score_col}
    k : int
        Number of stocks on each side (long and short).
    score_col : str
        Column to rank by (descending). Higher = more likely to outperform.

    Returns
    -------
    pd.DataFrame
        date, permno, rank, side ('long' or 'short'), score
    """
    df = predictions[["date", "permno", score_col]].copy()
    df = df.dropna(subset=[score_col])

    # Rank within each day (1 = highest score)
    df["rank"] = df.groupby("date")[score_col].rank(
        method="first", ascending=False
    ).astype(int)

    # Count stocks per day to determine bottom-k threshold
    day_counts = df.groupby("date")["rank"].transform("max")

    # Top k = long, bottom k = short
    long_mask = df["rank"] <= k
    short_mask = df["rank"] > (day_counts - k)

    selections = df[long_mask | short_mask].copy()
    selections["side"] = np.where(
        selections["rank"] <= k, "long", "short"
    )
    selections = selections.rename(columns={score_col: "score"})
    selections = selections.sort_values(["date", "side", "rank"])
    selections = selections.reset_index(drop=True)

    return selections
