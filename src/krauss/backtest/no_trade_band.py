"""
Cost-aware no-trade band with sequential two-pointer swap logic.

Algorithm (per day, per side):
    1. Top-k candidates chosen by score (higher = long, lower = short).
    2. Overlap between incumbents and candidates is kept unconditionally.
    3. Non-overlap incumbents ("expiring") and non-overlap candidates ("new")
       are sorted by Û:
           long side:  expiring ascending  (weakest first),
                       new descending      (strongest first).
           short side: expiring descending (highest-Û first, most replaceable),
                       new ascending       (most-negative-Û first).
    4. Two-pointer sweep:
           long:  swap if Û(new_j) >= Û(inc_i) + threshold  → advance both
                  else                                       → advance i only
           short: swap if Û(new_j) <= Û(inc_i) - threshold  → advance both
                  else                                       → advance i only
    5. Incumbents with NaN Û (e.g., dropped from universe) are force-evicted
       and their slots are filled by unused candidates when available.

Because incumbents may survive even when they fall out of today's top-k,
actual held positions are NOT just the k-highest-scoring stocks, so the
function tracks day-by-day holdings explicitly rather than reselecting.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def backtest_with_band(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    k: int,
    long_score_col: str,
    short_score_col: str,
    u_col: str,
    half_turn_bps: float = 5.0,
    swap_threshold_bps: float = 10.0,
) -> Dict[str, pd.DataFrame]:
    """
    Walk-forward backtest with a cost-aware no-trade band.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: date, permno, long_score_col, short_score_col, u_col.
    returns : pd.DataFrame
        Must have columns: permno, date, ret.
    k : int
        Number of positions per side when the portfolio is full.
    long_score_col : str
        Score used to select long candidates (top-k). NaN = ineligible.
    short_score_col : str
        Score used to select short candidates (bottom-k). NaN = ineligible.
    u_col : str
        Û column used for the incumbent-vs-challenger swap test.
    half_turn_bps : float
        Transaction cost per half-turn (default 5 bps, per the paper).
    swap_threshold_bps : float
        Minimum Û improvement required to swap (default 10 bps, round-trip cost).

    Returns
    -------
    dict with keys 'holdings', 'daily', 'turnover'.
    """
    threshold = swap_threshold_bps / 1e4

    cols = ["date", "permno", long_score_col, u_col]
    if short_score_col != long_score_col:
        cols.append(short_score_col)
    preds = predictions[cols].copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values(["date", "permno"]).reset_index(drop=True)

    holdings_records: List[dict] = []
    long_prev: List[int] = []
    short_prev: List[int] = []

    for d, day in preds.groupby("date", sort=True):
        u_map = dict(zip(day["permno"].values, day[u_col].values))

        ls = day[long_score_col].values
        perm = day["permno"].values
        mask_l = ~np.isnan(ls)
        if mask_l.any():
            order_l = np.argsort(-ls[mask_l], kind="stable")[:k]
            long_cands = perm[mask_l][order_l].tolist()
        else:
            long_cands = []

        ss = day[short_score_col].values
        mask_s = ~np.isnan(ss)
        if mask_s.any():
            order_s = np.argsort(ss[mask_s], kind="stable")[:k]
            short_cands = perm[mask_s][order_s].tolist()
        else:
            short_cands = []

        new_long = _apply_band(long_prev, long_cands, u_map, threshold, side="long")
        new_short = _apply_band(short_prev, short_cands, u_map, threshold, side="short")

        for p in new_long:
            holdings_records.append({"date": d, "permno": p, "side": "long"})
        for p in new_short:
            holdings_records.append({"date": d, "permno": p, "side": "short"})

        long_prev = new_long
        short_prev = new_short

    holdings = pd.DataFrame(holdings_records)

    # Merge next-day returns
    ret = returns[["permno", "date", "ret"]].copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret = ret.sort_values(["permno", "date"])
    ret["next_date"] = ret.groupby("permno")["date"].shift(-1)
    ret["next_day_ret"] = ret.groupby("permno")["ret"].shift(-1)

    holdings = holdings.merge(
        ret[["permno", "date", "next_date", "next_day_ret"]],
        on=["permno", "date"],
        how="left",
    )

    # Dynamic equal weights within each side (portfolio size may vary)
    side_counts = (
        holdings.groupby(["date", "side"]).size().rename("n_side").reset_index()
    )
    holdings = holdings.merge(side_counts, on=["date", "side"], how="left")
    holdings["weight"] = np.where(
        holdings["side"] == "long",
        1.0 / holdings["n_side"],
        -1.0 / holdings["n_side"],
    )
    holdings["contrib"] = holdings["weight"] * holdings["next_day_ret"]

    # Daily aggregation
    long_agg = (
        holdings[holdings["side"] == "long"]
        .groupby("date")
        .agg(
            long_ret=("next_day_ret", "mean"),
            n_long=("permno", "count"),
            next_date=("next_date", "first"),
        )
    )
    short_agg = (
        holdings[holdings["side"] == "short"]
        .groupby("date")
        .agg(short_ret=("next_day_ret", "mean"), n_short=("permno", "count"))
    )
    daily = long_agg.join(short_agg, how="outer")
    daily["long_ret"] = daily["long_ret"].fillna(0.0)
    daily["short_ret"] = daily["short_ret"].fillna(0.0)
    daily["n_long"] = daily["n_long"].fillna(0).astype(int)
    daily["n_short"] = daily["n_short"].fillna(0).astype(int)
    daily["port_ret"] = daily["long_ret"] - daily["short_ret"]
    daily = daily.reset_index()

    turnover = _compute_turnover(holdings)

    cost_frac = half_turn_bps / 1e4
    daily = daily.merge(turnover, on="date", how="left")
    daily["turnover"] = daily["turnover"].fillna(0.0)
    daily["cost"] = daily["turnover"] * cost_frac
    daily["port_ret_net"] = daily["port_ret"] - daily["cost"]

    # Last signal day has no next-day return; drop if entirely NaN
    daily = daily.dropna(subset=["port_ret"]).reset_index(drop=True)

    return {"holdings": holdings, "daily": daily, "turnover": turnover}


def _apply_band(
    incumbents: List[int],
    candidates: List[int],
    u_map: dict,
    threshold: float,
    side: str,
) -> List[int]:
    """Two-pointer swap decision for one side."""
    if not incumbents:
        return list(candidates)

    inc_set = set(incumbents)
    cand_set = set(candidates)
    overlap = inc_set & cand_set
    expiring = [p for p in incumbents if p not in overlap]
    new_cands = [p for p in candidates if p not in overlap]

    def u_of(p):
        v = u_map.get(p, np.nan)
        return v if pd.notna(v) else np.nan

    # Force-evict incumbents that no longer have a valid Û (e.g., dropped from universe)
    expiring_valid = [p for p in expiring if pd.notna(u_of(p))]
    num_forced_evicts = len(expiring) - len(expiring_valid)

    if side == "long":
        expiring_sorted = sorted(expiring_valid, key=u_of)  # ascending
        new_sorted = sorted(new_cands, key=u_of, reverse=True)  # descending
    else:  # short
        expiring_sorted = sorted(expiring_valid, key=u_of, reverse=True)
        new_sorted = sorted(new_cands, key=u_of)

    kept: List[int] = []
    added: List[int] = []
    i = j = 0

    while i < len(expiring_sorted) and j < len(new_sorted):
        u_inc = u_of(expiring_sorted[i])
        u_new = u_of(new_sorted[j])

        if side == "long":
            swap = u_new >= u_inc + threshold
        else:
            swap = u_new <= u_inc - threshold

        if swap:
            added.append(new_sorted[j])
            i += 1
            j += 1
        else:
            kept.append(expiring_sorted[i])
            i += 1

    while i < len(expiring_sorted):
        kept.append(expiring_sorted[i])
        i += 1

    # Fill force-evicted slots from any unused candidates
    for _ in range(num_forced_evicts):
        if j < len(new_sorted):
            added.append(new_sorted[j])
            j += 1
        else:
            break

    return list(overlap) + kept + added


def _compute_turnover(holdings: pd.DataFrame) -> pd.DataFrame:
    """Sum of |weight changes| per day (supports variable per-day weights)."""
    dates = sorted(holdings["date"].unique())
    records = []
    prev_w: dict = {}
    for d in dates:
        day = holdings.loc[holdings["date"] == d]
        curr_w = dict(zip(day["permno"], day["weight"]))
        all_p = set(curr_w.keys()) | set(prev_w.keys())
        turn = sum(abs(curr_w.get(p, 0.0) - prev_w.get(p, 0.0)) for p in all_p)
        records.append({"date": d, "turnover": turn})
        prev_w = curr_w
    return pd.DataFrame(records)
