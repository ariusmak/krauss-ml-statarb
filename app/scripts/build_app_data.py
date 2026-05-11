"""
Build derived data artifacts for the Streamlit research-demo app.

Reads pre-computed predictions and return panels from the repo's data/
directory (and from the cost-modeling branch via data/_staging/ if present),
then writes a small set of parquet and JSON files under ``app/data/`` that
the Streamlit app consumes directly.

The app must read only these pre-computed files; no live backtests run at
page-render time.

Usage
-----
    python scripts/build_app_data.py

Inputs (main branch)
--------------------
    data/processed/predictions_phase1.parquet
    data/processed/predictions_phase2.parquet
    data/processed/daily_returns.parquet
    data/raw/sp500_index_daily.parquet

Inputs (cost-modeling branch, staged to data/_staging/)
-------------------------------------------------------
    data/_staging/predictions_phase2_ds.parquet
    data/_staging/ds_daily_returns_usonly.parquet
    data/_staging/ds_universe_daily_usonly.parquet
    data/_staging/ds_sp500_membership.parquet

Outputs (app/data/)
-------------------
    equity_curves.parquet    — daily equity curve per (model, scheme, cost_regime, era)
    summary_table.parquet    — one row per (model, scheme, era, cost_regime)
    daily_holdings.parquet   — top-10 longs / bottom-10 shorts per (date, scheme)
                               for the four demo schemes on ENS1
    regime_labels.parquet    — date, VIX, VIX_5d, regime, extension_era flag
    disagreement_panel.parquet — per-date P_hat / U_hat for every stock (for Page 4)
    pipeline_metadata.json   — period ranges, model hyperparameters, data notes

Phase 1 of the app delivery needs only equity_curves, summary_table,
regime_labels, and pipeline_metadata — the rest are built opportunistically
when the inputs are available.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
STAGING = ROOT / "data" / "_staging"
APP_DATA = ROOT / "app" / "data"


MODEL_COLS_P1 = {
    "RF": "p_rf",
    "XGB": "p_xgb",
    "DNN": "p_dnn",
    "ENS1": "p_ens1",
    "ENS2": "p_ens2",
    "ENS3": "p_ens3",
}
MODEL_COLS_P2_DS = {
    "RF": "p_rf",
    "XGB": "p_xgb",
    "DNN": "p_dnn",
    "ENS1": "p_ens1",
}

COST_BPS_PER_HALF_TURN = 5.0
K = 10


def _log(msg: str) -> None:
    print(f"[build_app_data] {msg}", flush=True)


# Columns eligible for category-encoding on write — only applied when the
# caller opts in (large parquets where the storage win is material).  Small
# files keep these as object/string to avoid the categorical-arithmetic
# footgun (Bug A).
_CATEGORY_COLS = {
    "model", "scheme", "era", "cost_regime", "side", "id_type",
}


def _compact(df: pd.DataFrame, *, categorize: bool = False) -> pd.DataFrame:
    """Downcast numerics in place; optionally category-encode label columns.

    Numeric downcasting (float64 → float32, Int64/int64 → int32 for IDs and
    ranks) always runs and is safe.  Categorical encoding for the four
    label columns is **off by default** because downstream string-concat
    on those columns trips ``TypeError: Object with dtype category cannot
    perform the numpy op add``.  Pass ``categorize=True`` only for large
    parquets where the on-disk saving is material — currently just
    ``disagreement_panel.parquet``.
    """
    for c in df.columns:
        if categorize and c in _CATEGORY_COLS and df[c].dtype == "object":
            df[c] = df[c].astype("category")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif str(df[c].dtype) == "Int64":  # pandas nullable
            # Stock IDs and ranks fit comfortably in int32.
            df[c] = df[c].astype("int32")
        elif df[c].dtype == "int64" and c in {"rank", "stock_id"}:
            df[c] = df[c].astype("int32")
    return df


def _write_parquet(df: pd.DataFrame, path: Path,
                    *, categorize: bool = False) -> None:
    _compact(df, categorize=categorize).to_parquet(
        path, index=False, compression="zstd", compression_level=9
    )


# -----------------------------------------------------------------------------
# Backtest primitives (self-contained — no dependency on src/krauss.backtest
# so the script runs even if those modules are empty).
# -----------------------------------------------------------------------------

def rank_topk_bottomk(pred: pd.DataFrame, score_col: str, k: int,
                      id_col: str = "permno") -> pd.DataFrame:
    """Rank each date descending by score and tag long/short sides."""
    df = pred[["date", id_col, score_col]].dropna(subset=[score_col]).copy()
    df["rank"] = df.groupby("date")[score_col].rank(method="first",
                                                      ascending=False).astype(int)
    day_max = df.groupby("date")["rank"].transform("max")
    long_mask = df["rank"] <= k
    short_mask = df["rank"] > (day_max - k)
    sel = df[long_mask | short_mask].copy()
    sel["side"] = np.where(sel["rank"] <= k, "long", "short")
    return sel[["date", id_col, "side", "rank", score_col]].rename(
        columns={score_col: "score"}
    )


def join_next_day_returns(sel: pd.DataFrame, returns: pd.DataFrame,
                          id_col: str = "permno") -> pd.DataFrame:
    ret = returns[[id_col, "date", "ret"]].sort_values([id_col, "date"])
    ret["next_day_ret"] = ret.groupby(id_col)["ret"].shift(-1)
    ret = ret.dropna(subset=["next_day_ret"])[[id_col, "date", "next_day_ret"]]
    out = sel.merge(ret, on=["date", id_col], how="inner")
    return out


def daily_portfolio_returns(holdings: pd.DataFrame, k: int) -> pd.DataFrame:
    """Equal-weight dollar-neutral: port_ret = mean(long) - mean(short)."""
    grp = holdings.groupby(["date", "side"])["next_day_ret"].mean().unstack("side")
    grp = grp.rename(columns={"long": "long_ret", "short": "short_ret"})
    grp["port_ret"] = grp["long_ret"] - grp["short_ret"]
    return grp.reset_index()


def daily_turnover(holdings: pd.DataFrame, k: int,
                   id_col: str = "permno") -> pd.DataFrame:
    """Turnover as sum of absolute weight changes across consecutive dates."""
    h = holdings.assign(weight=np.where(holdings["side"] == "long",
                                         1.0 / k, -1.0 / k))
    pivot = h.pivot_table(index="date", columns=id_col, values="weight",
                           aggfunc="sum", fill_value=0.0)
    pivot = pivot.sort_index()
    diffs = pivot.diff()
    diffs.iloc[0] = pivot.iloc[0]
    turnover = diffs.abs().sum(axis=1)
    return turnover.reset_index(name="turnover")


def apply_costs(daily: pd.DataFrame, turnover: pd.DataFrame,
                bps: float = COST_BPS_PER_HALF_TURN) -> pd.DataFrame:
    cost_frac = bps / 10_000.0
    df = daily.merge(turnover, on="date", how="left")
    df["turnover"] = df["turnover"].fillna(0.0)
    df["cost"] = df["turnover"] * cost_frac
    df["port_ret_net"] = df["port_ret"] - df["cost"]
    return df


# -----------------------------------------------------------------------------
# Scoring schemes.  Each returns a DataFrame with columns
# (date, id, score) ready to feed into rank_topk_bottomk.
# -----------------------------------------------------------------------------

def score_p_only(pred: pd.DataFrame, model: str, id_col: str) -> tuple[pd.DataFrame, str]:
    col = f"p_{model.lower()}"
    return pred[["date", id_col, col]], col


def score_u_only(pred: pd.DataFrame, model: str, id_col: str) -> tuple[pd.DataFrame, str]:
    col = f"u_{model.lower()}"
    return pred[["date", id_col, col]], col


def _cross_sectional_z(series: pd.Series, groupby: pd.Series) -> pd.Series:
    """Per-day z-score; safe against zero std."""
    mu = series.groupby(groupby).transform("mean")
    sd = series.groupby(groupby).transform("std").replace(0, np.nan)
    z = (series - mu) / sd
    return z.fillna(0.0)


def score_z_composite(pred: pd.DataFrame, model: str, id_col: str) -> tuple[pd.DataFrame, str]:
    p_col = f"p_{model.lower()}"
    u_col = f"u_{model.lower()}"
    df = pred[["date", id_col, p_col, u_col]].copy()
    df["_zp"] = _cross_sectional_z(df[p_col], df["date"])
    df["_zu"] = _cross_sectional_z(df[u_col], df["date"])
    df["score_zcomp"] = 0.5 * df["_zp"] + 0.5 * df["_zu"]
    return df[["date", id_col, "score_zcomp"]], "score_zcomp"


def score_product_composite(pred: pd.DataFrame, model: str, id_col: str) -> tuple[pd.DataFrame, str]:
    p_col = f"p_{model.lower()}"
    u_col = f"u_{model.lower()}"
    df = pred[["date", id_col, p_col, u_col]].copy()
    df["score_prod"] = (2.0 * df[p_col] - 1.0) * df[u_col]
    return df[["date", id_col, "score_prod"]], "score_prod"


def _selections_p_gate(pred: pd.DataFrame, model: str, id_col: str,
                        gate: float, k: int) -> pd.DataFrame:
    """Asymmetric P-gate + asymmetric U-ranked selection.

    Matches the canonical ``gated_rank_and_select`` in
    ``src/krauss/evaluation/phase2_ds_backtest_utils.py``:

    * Longs:  keep P > 0.5 + gate, sort by U descending, take top k.
    * Shorts: keep P < 0.5 - gate, sort by U ascending, take bottom k.
    """
    p_col = f"p_{model.lower()}"
    u_col = f"u_{model.lower()}"
    df = pred[["date", id_col, p_col, u_col]].dropna().copy()

    longs = df[df[p_col] > 0.5 + gate]
    shorts = df[df[p_col] < 0.5 - gate]

    # Daily top-k longs (largest u), bottom-k shorts (smallest u).
    longs = longs.sort_values(["date", u_col], ascending=[True, False])
    longs["rank"] = longs.groupby("date").cumcount() + 1
    longs = longs[longs["rank"] <= k].copy()
    longs["side"] = "long"

    shorts = shorts.sort_values(["date", u_col], ascending=[True, True])
    shorts["rank"] = shorts.groupby("date").cumcount() + 1
    shorts = shorts[shorts["rank"] <= k].copy()
    shorts["side"] = "short"

    sel = pd.concat([longs, shorts], ignore_index=True)
    sel["score"] = sel[u_col]
    return sel[["date", id_col, "side", "rank", "score"]]


# -----------------------------------------------------------------------------
# Backtest driver
# -----------------------------------------------------------------------------

def run_backtest(pred: pd.DataFrame, returns: pd.DataFrame,
                 selections: pd.DataFrame, k: int, id_col: str) -> pd.DataFrame:
    """Given a selections DataFrame (date, id, side, rank[, score]), compute
    daily portfolio returns, turnover, and post-cost returns.
    """
    holdings = join_next_day_returns(selections, returns, id_col=id_col)
    if holdings.empty:
        return pd.DataFrame(columns=["date", "port_ret", "port_ret_net",
                                      "turnover", "long_ret", "short_ret"])
    daily = daily_portfolio_returns(holdings, k)
    turn = daily_turnover(holdings, k, id_col=id_col)
    daily = apply_costs(daily, turn)
    return daily[["date", "long_ret", "short_ret", "port_ret",
                  "turnover", "cost", "port_ret_net"]]


def _sel_from_score(pred: pd.DataFrame, score_df: pd.DataFrame, score_col: str,
                     k: int, id_col: str) -> pd.DataFrame:
    sel = rank_topk_bottomk(score_df, score_col, k, id_col=id_col)
    return sel


# -----------------------------------------------------------------------------
# Schemes list — each scorer takes (pred, model, id_col, k) and returns a
# selections DataFrame with columns (date, id_col, side, rank, score).
# -----------------------------------------------------------------------------

def _selector_for(scheme_key: str):
    def _p_only(pred, m, idc, k):
        s, c = score_p_only(pred, m, idc)
        return _sel_from_score(pred, s, c, k, idc)
    def _u_only(pred, m, idc, k):
        s, c = score_u_only(pred, m, idc)
        return _sel_from_score(pred, s, c, k, idc)
    def _zcomp(pred, m, idc, k):
        s, c = score_z_composite(pred, m, idc)
        return _sel_from_score(pred, s, c, k, idc)
    def _prod(pred, m, idc, k):
        s, c = score_product_composite(pred, m, idc)
        return _sel_from_score(pred, s, c, k, idc)
    def _gate(gate):
        return lambda pred, m, idc, k: _selections_p_gate(pred, m, idc, gate, k)

    return {
        "P-only":       _p_only,
        "U-only":       _u_only,
        "Z-comp":       _zcomp,
        "Product":      _prod,
        "P-gate(0.03)": _gate(0.03),
        "P-gate(0.05)": _gate(0.05),
    }[scheme_key]


SCHEME_KEYS = ["P-only", "U-only", "Z-comp", "Product",
               "P-gate(0.03)", "P-gate(0.05)"]


def trading_day_stats(daily: pd.DataFrame, ret_col: str) -> dict:
    r = daily[ret_col].dropna()
    if r.empty:
        return dict(daily_mean=np.nan, ann_return=np.nan, sharpe=np.nan,
                    trading_days=0)
    mean = r.mean()
    std = r.std(ddof=0)
    sharpe = np.sqrt(252) * mean / std if std > 0 else np.nan
    return dict(daily_mean=float(mean),
                ann_return=float(252 * mean),
                sharpe=float(sharpe),
                trading_days=int(r.shape[0]))


# -----------------------------------------------------------------------------
# Era builders
# -----------------------------------------------------------------------------

def build_era_crsp(pred_p1: pd.DataFrame, pred_p2: pd.DataFrame,
                    returns_crsp: pd.DataFrame) -> pd.DataFrame:
    """Backtest all (model, scheme) pairs on the 1992-2015 CRSP era.

    - The P-only baseline is run on the 6 Phase-1 models from pred_p1.
    - The other five schemes are run on the 4 Phase-2 models with both p_ and
      u_ heads (RF, XGB, DNN, ENS1) using pred_p2.
    """
    rows = []
    era = "1992-2015 (CRSP)"

    # P-only across all 6 models (Phase 1 predictions).
    for model, col in MODEL_COLS_P1.items():
        _log(f"CRSP  · {model} · P-only")
        scorer = _selector_for("P-only")
        sel = scorer(pred_p1, model, "permno", K)
        daily = run_backtest(pred_p1, returns_crsp, sel, K, id_col="permno")
        if daily.empty:
            continue
        rows.append(_attach_meta(daily, model, "P-only", era))

    # Phase 2 schemes on the 4 models with u_ predictions.
    p2_models = ["RF", "XGB", "DNN", "ENS1"]
    for model in p2_models:
        for scheme_key in [s for s in SCHEME_KEYS if s != "P-only"]:
            _log(f"CRSP  · {model} · {scheme_key}")
            scorer = _selector_for(scheme_key)
            sel = scorer(pred_p2, model, "permno", K)
            daily = run_backtest(pred_p2, returns_crsp, sel, K, id_col="permno")
            if daily.empty:
                continue
            rows.append(_attach_meta(daily, model, scheme_key, era))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_era_extension(pred_p2_ds: pd.DataFrame,
                        returns_ds: pd.DataFrame) -> pd.DataFrame:
    """Backtest on the 2015-2025 Datastream extension.

    The extension predictions only include rf/xgb/dnn/ens1.  We run all six
    schemes on these four models.
    """
    rows = []
    era = "2015-2025 (extension)"

    for model in MODEL_COLS_P2_DS.keys():
        for scheme_key in SCHEME_KEYS:
            _log(f"EXT   · {model} · {scheme_key}")
            scorer = _selector_for(scheme_key)
            sel = scorer(pred_p2_ds, model, "infocode", K)
            daily = run_backtest(pred_p2_ds, returns_ds, sel, K, id_col="infocode")
            if daily.empty:
                continue
            rows.append(_attach_meta(daily, model, scheme_key, era))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _attach_meta(daily: pd.DataFrame, model: str, scheme: str, era: str) -> pd.DataFrame:
    """Tag a daily returns DataFrame with (model, scheme, era) and attach
    cumulative curves for both cost regimes.
    """
    df = daily.sort_values("date").copy()
    df["model"] = model
    df["scheme"] = scheme
    df["era"] = era
    # cost regimes are encoded as stacked rows:
    gross = df.rename(columns={"port_ret": "ret"}).copy()
    gross["cost_regime"] = "no_cost"
    gross["ret"] = df["port_ret"]
    net = df.rename(columns={"port_ret_net": "ret"}).copy()
    net["cost_regime"] = "5bps_half_turn"
    net["ret"] = df["port_ret_net"]
    stacked = pd.concat([
        gross[["date", "model", "scheme", "era", "cost_regime", "ret"]],
        net[["date", "model", "scheme", "era", "cost_regime", "ret"]],
    ], ignore_index=True)
    # cumulative compounded
    stacked = stacked.sort_values(["model", "scheme", "era", "cost_regime", "date"])
    stacked["cum_ret"] = stacked.groupby(
        ["model", "scheme", "era", "cost_regime"]
    )["ret"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod() - 1.0)
    # daily turnover is repeated on both cost rows
    stacked = stacked.merge(
        df[["date", "turnover"]], on="date", how="left"
    )
    return stacked


# -----------------------------------------------------------------------------
# Summary and regime builders
# -----------------------------------------------------------------------------

def densify_equity_curves(equity: pd.DataFrame) -> pd.DataFrame:
    """Reindex every (model, scheme, era, cost_regime) onto the dense union
    of trading dates *for that era*, filling no-trade days with ret = 0.

    Gated schemes only have rows on days where the gate produced a portfolio.
    For Sharpe / drawdown / cumulative-curve calculations to be honest over
    user-selected windows, every scheme needs to share the same per-era date
    axis — a no-trade day is genuinely a 0 % return, not a missing day.
    """
    group_cols = ["era", "model", "scheme", "cost_regime"]
    df = equity.copy()
    df["ret"] = df["ret"].fillna(0.0)
    if "turnover" in df.columns:
        df["turnover"] = df["turnover"].fillna(0.0)

    # Per-era trading-day axis: union of every date that any scheme produced a
    # row on within that era.  Use the non-gated P-only series as the spine if
    # available because it trades every day.
    era_axis = {}
    for era, grp in df.groupby("era", observed=True):
        era_axis[era] = (
            grp["date"].drop_duplicates().sort_values().reset_index(drop=True)
        )

    dense_rows = []
    for keys, grp in df.groupby(group_cols, observed=True, sort=False):
        era, model, scheme, cost_regime = keys
        spine = pd.DataFrame({"date": era_axis[era]})
        full = spine.merge(
            grp[["date", "ret", "turnover"]] if "turnover" in grp.columns
            else grp[["date", "ret"]],
            on="date", how="left",
        )
        full["ret"] = full["ret"].fillna(0.0)
        if "turnover" in full.columns:
            full["turnover"] = full["turnover"].fillna(0.0)
        full["era"] = era
        full["model"] = model
        full["scheme"] = scheme
        full["cost_regime"] = cost_regime
        dense_rows.append(full)

    out = pd.concat(dense_rows, ignore_index=True)
    out = out.sort_values(group_cols + ["date"]).reset_index(drop=True)
    out["cum_ret"] = out.groupby(group_cols, observed=True)["ret"].transform(
        lambda r: (1.0 + r).cumprod() - 1.0
    )
    # Additive cumulative P&L for the more-honest "$ won per $ traded" view.
    out["cum_pnl"] = out.groupby(group_cols, observed=True)["ret"].cumsum()
    return out


def _matched_days_per_era(equity: pd.DataFrame) -> dict:
    """The set of dates P-gate(0.05) ENS1 actually trades, per era.

    Must be called on the **pre-densification** equity frame: in that frame
    P-gate(0.05) only has rows on days where the gate produced a portfolio,
    so ``ret.notna()`` cleanly identifies the 2,703 / 792 anchor days.

    After densification (added in Phase 5b) every (model, scheme, era,
    cost_regime) has rows on every era trading day with ``ret = 0`` filled in
    on no-trade days, so ``ret.notna()`` is uninformative and must not be
    used.  The build script computes this dict once *before* densification
    and threads the result through ``build_summary``.
    """
    # Anchor on the 5bps cost regime: ``ret = gross - cost`` is never exactly
    # zero on a real trade day (cost > 0 whenever turnover > 0), so this
    # detection works against both pre- and post-densification frames.  The
    # no_cost regime occasionally lands at ret = 0.0 by chance when the
    # long and short legs realise identical returns, which produces an
    # off-by-one anchor count in era 2.
    mask = (
        (equity["scheme"] == "P-gate(0.05)")
        & (equity["model"] == "ENS1")
        & (equity["cost_regime"] == "5bps_half_turn")
        & (equity["ret"].notna())
        & (equity["ret"] != 0)
    )
    sub = equity[mask]
    return {era: set(grp["date"].unique())
            for era, grp in sub.groupby("era", observed=True)}


def build_summary(equity: pd.DataFrame,
                   matched_by_era: dict | None = None) -> pd.DataFrame:
    """Aggregate per-(model, scheme, era, cost_regime) summary metrics.

    Pass ``matched_by_era`` precomputed from the pre-densification equity
    frame so the matched-days anchor stays correct after densification.  If
    omitted, falls back to computing it from ``equity`` (only correct for
    sparse / non-densified inputs).
    """
    if matched_by_era is None:
        matched_by_era = _matched_days_per_era(equity)
    rows = []
    group_cols = ["model", "scheme", "era", "cost_regime"]
    for keys, grp in equity.groupby(group_cols, observed=True):
        model, scheme, era, cost_regime = keys
        r = grp["ret"].dropna()
        if r.empty:
            continue
        mean = r.mean()
        std = r.std(ddof=0)
        sharpe = np.sqrt(252) * mean / std if std > 0 else np.nan

        # Return restricted to days where P-gate(0.05) was active.
        matched = matched_by_era.get(era, set())
        if matched:
            mr = grp[grp["date"].isin(matched)]["ret"].dropna()
            matched_mean = float(mr.mean()) if len(mr) else np.nan
            matched_days = int(len(mr))
            matched_sharpe = (float(np.sqrt(252) * mr.mean() / mr.std(ddof=0))
                               if len(mr) > 1 and mr.std(ddof=0) > 0 else np.nan)
        else:
            matched_mean = np.nan
            matched_days = 0
            matched_sharpe = np.nan

        tail = r.iloc[-252:] if len(r) >= 252 else r

        # ---- Phase 5b risk metrics --------------------------------------
        ann_ret = float(252 * mean)
        ann_vol = float(np.sqrt(252) * std) if std > 0 else np.nan

        # Sortino: annualised mean / annualised downside-deviation.  Downside
        # deviation = std of negative-only daily returns (ddof=0).
        downside = r[r < 0]
        if len(downside) >= 2 and downside.std(ddof=0) > 0:
            dd_std = float(downside.std(ddof=0))
            sortino = float(np.sqrt(252) * mean / dd_std)
        else:
            sortino = np.nan

        # Max drawdown on the densified compounded curve.  cum_ret already
        # exists in `grp`; running max gives peak; current minus peak gives
        # drawdown; minimum across the series is the worst.
        cum_curve = (1.0 + r).cumprod()
        running_max = cum_curve.cummax()
        drawdown = cum_curve / running_max - 1.0
        max_dd = float(drawdown.min()) if len(drawdown) else np.nan

        calmar = (ann_ret / abs(max_dd)) if max_dd not in (0, np.nan) and max_dd < 0 else np.nan

        hit_rate = float((r > 0).mean()) if len(r) else np.nan
        worst_day = float(r.min()) if len(r) else np.nan
        best_day = float(r.max()) if len(r) else np.nan

        rows.append(dict(
            model=model, scheme=scheme, era=era, cost_regime=cost_regime,
            daily_return=float(mean),
            full_sample_return=float(mean),          # alias: clearer name
            matched_days_return=matched_mean,         # mean ret on P-gate(0.05) days
            matched_days_count=matched_days,
            matched_days_sharpe=matched_sharpe,
            ann_return=ann_ret,
            ann_vol=ann_vol,
            volatility=ann_vol,                      # alias for the simulator
            sharpe=float(sharpe),
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            hit_rate=hit_rate,
            worst_day=worst_day,
            best_day=best_day,
            trading_days=int(len(r)),
            avg_turnover=float(grp["turnover"].mean()),
            cum_return=float(grp["cum_ret"].iloc[-1]) if not grp.empty else np.nan,
            total_pnl=float(r.sum()),                # additive cum P&L
            trailing_1y_return=float(252 * tail.mean()),
        ))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Real VIX regime labels (vendored from
# src/krauss/regimes/vix_regimes.py on the vix-regime-analysis branch)
# -----------------------------------------------------------------------------

VIX_LOW_THRESHOLD = 20.0
VIX_HIGH_THRESHOLD = 30.0
VIX_SMOOTHING_WINDOW = 5


def build_regime_labels(vix: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Label each trading day with a VIX regime using the vix_regimes logic.

    Thresholds: <20 low_vol, 20-30 mid_vol, >30 high_vol.
    Smoothing: 5-day trailing mean, strictly backward-looking (shift(1)).
    """
    df = (
        vix[["date", "vix"]].sort_values("date").reset_index(drop=True).copy()
    )
    df["date"] = pd.to_datetime(df["date"])

    # Trailing mean strictly excluding day t.
    df["vix_smooth"] = (
        df["vix"].shift(1)
        .rolling(window=VIX_SMOOTHING_WINDOW,
                  min_periods=VIX_SMOOTHING_WINDOW)
        .mean()
    )

    regime = pd.Series(pd.NA, index=df.index, dtype="object")
    mask_low = df["vix_smooth"] < VIX_LOW_THRESHOLD
    mask_high = df["vix_smooth"] > VIX_HIGH_THRESHOLD
    mask_mid = (~mask_low) & (~mask_high) & df["vix_smooth"].notna()
    regime.loc[mask_low] = "low_vol"
    regime.loc[mask_mid] = "mid_vol"
    regime.loc[mask_high] = "high_vol"
    df["regime"] = regime

    df["extension_era"] = df["date"] >= pd.Timestamp("2015-10-16")
    return df[["date", "vix", "vix_smooth", "regime", "extension_era"]]


# -----------------------------------------------------------------------------
# Daily holdings for Page 8 (trading demo)
# -----------------------------------------------------------------------------

def build_daily_holdings(pred_p2_crsp: pd.DataFrame,
                          pred_p2_ds: pd.DataFrame,
                          ret_crsp: pd.DataFrame,
                          ret_ds: pd.DataFrame,
                          permno_ticker: pd.DataFrame | None) -> pd.DataFrame:
    """For each of the four demo schemes on ENS1, list the 10 longs and 10
    shorts on every trading day together with realized next-day returns.
    """
    demo_schemes = ["P-only", "Z-comp", "P-gate(0.03)", "P-gate(0.05)"]
    rows = []
    for scheme_key in demo_schemes:
        for era, pred, ret, idc in [
            ("1992-2015 (CRSP)",    pred_p2_crsp, ret_crsp, "permno"),
            ("2015-2025 (extension)", pred_p2_ds, ret_ds,  "infocode"),
        ]:
            _log(f"holdings · {era} · {scheme_key}")
            scorer = _selector_for(scheme_key)
            sel = scorer(pred, "ENS1", idc, K)
            h = join_next_day_returns(sel, ret, id_col=idc)
            if h.empty:
                continue
            # attach P_hat and U_hat (ENS1) where possible
            p_ens1 = pred[["date", idc, "p_ens1"]].rename(columns={"p_ens1": "p_hat"})
            u_ens1 = pred[["date", idc, "u_ens1"]].rename(columns={"u_ens1": "u_hat"})
            h = h.merge(p_ens1, on=["date", idc], how="left")
            h = h.merge(u_ens1, on=["date", idc], how="left")
            h["scheme"] = scheme_key
            h["era"] = era
            h = h.rename(columns={idc: "stock_id"})
            h["id_type"] = idc
            rows.append(h[["date", "scheme", "era", "side", "rank",
                           "stock_id", "id_type", "p_hat", "u_hat",
                           "next_day_ret"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if permno_ticker is not None and not permno_ticker.empty:
        out = out.merge(
            permno_ticker.rename(columns={"permno": "stock_id"}),
            on="stock_id", how="left"
        )
    return out


# -----------------------------------------------------------------------------
# Authoritative tables vendored from branch-specific notebooks.
# These are copied verbatim from cell outputs; see each function's docstring
# for the source notebook and commit.
# -----------------------------------------------------------------------------

def build_cost_band_table() -> pd.DataFrame:
    """16-row no-trade band × model matrix from
    ``notebooks/cost_band_test.ipynb`` on the cost-modeling branch.

    Parameters: k=10, 5 bps per half-turn, 10 bps no-trade band threshold.
    All values are post-cost daily returns (%), annualised return (%) and
    net Sharpe — copied verbatim from the notebook's cell outputs.
    """
    rows = [
        # (model, scheme, baseline_dret_pct, band_dret_pct,
        #  baseline_annret_pct, band_annret_pct, baseline_sharpe, band_sharpe)
        ("DNN",  "P-gate(0.03)",  0.0298, -0.0073,   7.52,  -1.85,  0.13, -0.05),
        ("DNN",  "P-gate(0.05)",  0.0603,  0.0059,  15.21,   1.50,  0.23,  0.03),
        ("DNN",  "U-only",        0.0822, -0.0037,  20.71,  -0.93,  0.52, -0.03),
        ("DNN",  "Z-comp",        0.1218,  0.0039,  30.70,   0.98,  0.79,  0.03),
        ("ENS1", "P-gate(0.03)",  0.3300,  0.2754,  83.16,  69.40,  1.61,  1.45),
        ("ENS1", "P-gate(0.05)",  0.4506,  0.2808, 113.55,  70.75,  1.59,  1.02),
        ("ENS1", "U-only",        0.2567,  0.2561,  64.68,  64.54,  1.72,  1.71),
        ("ENS1", "Z-comp",        0.3058,  0.2900,  77.06,  73.08,  1.97,  1.85),
        ("RF",   "P-gate(0.03)",  0.2633,  0.2504,  66.35,  63.11,  1.68,  1.61),
        ("RF",   "P-gate(0.05)",  0.2844,  0.2501,  71.66,  63.01,  1.44,  1.52),
        ("RF",   "U-only",        0.2199,  0.2174,  55.41,  54.79,  1.51,  1.50),
        ("RF",   "Z-comp",        0.2862,  0.2487,  72.12,  62.68,  1.92,  1.65),
        ("XGB",  "P-gate(0.03)",  0.2073,  0.2256,  52.25,  56.86,  1.65,  1.74),
        ("XGB",  "P-gate(0.05)",  0.2542,  0.2552,  64.05,  64.30,  1.37,  1.72),
        ("XGB",  "U-only",        0.1829,  0.2078,  46.09,  52.37,  1.47,  1.68),
        ("XGB",  "Z-comp",        0.2242,  0.2428,  56.51,  61.20,  1.80,  1.92),
    ]
    df = pd.DataFrame(rows, columns=[
        "model", "scheme",
        "baseline_daily_return_pct", "band10_daily_return_pct",
        "baseline_ann_return_pct", "band10_ann_return_pct",
        "baseline_sharpe", "band10_sharpe",
    ])
    df["sharpe_delta"] = df["band10_sharpe"] - df["baseline_sharpe"]
    df["daily_return_delta_pct"] = (
        df["band10_daily_return_pct"] - df["baseline_daily_return_pct"]
    )
    df["ann_return_delta_pct"] = (
        df["band10_ann_return_pct"] - df["baseline_ann_return_pct"]
    )
    df["source"] = "notebooks/cost_band_test.ipynb (cost-modeling branch)"
    return df


def build_regime_k_sensitivity_table() -> pd.DataFrame:
    """Z-comp ENS1 × VIX regime × k grid from
    ``notebooks/regime_analysis.ipynb`` on the vix-regime-analysis branch.

    daily_return_bps = post-cost daily return in basis points (1 bp = 0.01 %).
    sharpe = annualised Sharpe net of 5 bps/half-turn costs.
    """
    rows = [
        # (regime, k, daily_return_bps, sharpe)
        ("all",       10,  30.58, 1.92),
        ("all",       50,  13.01, 1.57),
        ("all",      100,   7.19, 1.12),
        ("all",      150,   4.66, 0.84),
        ("all",      200,   3.28, 0.66),
        ("low_vol",   10,  22.52, 2.42),
        ("low_vol",   50,   9.60, 2.09),
        ("low_vol",  100,   4.69, 1.31),
        ("low_vol",  150,   2.35, 0.66),
        ("low_vol",  200,   1.46, 0.34),
        ("mid_vol",   10,  43.32, 2.26),
        ("mid_vol",   50,  18.52, 1.89),
        ("mid_vol",  100,  10.55, 1.42),
        ("mid_vol",  150,   7.79, 1.28),
        ("mid_vol",  200,   5.38, 1.01),
        ("high_vol",  10,  39.07, 1.31),
        ("high_vol",  50,  16.02, 0.99),
        ("high_vol", 100,  11.58, 0.94),
        ("high_vol", 150,   8.65, 0.84),
        ("high_vol", 200,   7.70, 0.89),
    ]
    df = pd.DataFrame(rows, columns=[
        "regime", "k", "daily_return_bps", "sharpe",
    ])
    df["source"] = "notebooks/regime_analysis.ipynb (vix-regime-analysis branch)"
    return df


def build_spy_benchmark(start: str = "1992-01-01",
                          end: str | None = None) -> pd.DataFrame:
    """Download SPY daily adjusted close and convert to a daily-return series.

    Uses yfinance with auto_adjust=True so the returns already include
    dividends (total-return series).  Cached to app/data/spy_benchmark.parquet
    by the main driver.
    """
    import yfinance as yf

    tkr = yf.Ticker("SPY")
    hist = tkr.history(start=start, end=end or pd.Timestamp.today().isoformat(),
                        auto_adjust=True)
    df = pd.DataFrame({
        "date": hist.index.tz_localize(None).normalize(),
        "close": hist["Close"].values,
    })
    df["ret"] = df["close"].pct_change()
    df["cum_ret"] = (1.0 + df["ret"].fillna(0.0)).cumprod() - 1.0
    return df


def build_regime_leg_decomp_table() -> pd.DataFrame:
    """ENS1 Z-comp Sharpe under three 'can we rescue high-vol?' rules from
    ``notebooks/regime_leg_decomp.ipynb`` on the vix-regime-analysis branch.
    """
    rows = [
        ("A. Baseline (all days)",                1.925,
          "Full 1992-2015 sample, k=10, post-cost"),
        ("B. Baseline ex Sep-Nov 2008 (GFC)",     2.180,
          "Drops 66 days around the Lehman window"),
        ("C. Cash-on-high-vol days",              2.070,
          "Sit out when VIX 5-day mean > 30"),
        ("D. Long leg only, high-vol days",      -0.050,
          "High-vol short-leg is the source of the drag"),
        ("E. Regime-conditional k (best fit)",    1.910,
          "Grid-searched k by regime — no lift over A"),
    ]
    df = pd.DataFrame(rows, columns=["rule", "sharpe", "description"])
    df["source"] = "notebooks/regime_leg_decomp.ipynb (vix-regime-analysis branch)"
    return df


# -----------------------------------------------------------------------------
# Disagreement panel for Page 4
# -----------------------------------------------------------------------------

def build_disagreement_panel(pred_p2: pd.DataFrame,
                              pred_p2_ds: pd.DataFrame | None = None
                              ) -> pd.DataFrame:
    """For every (date, stock) record p_ens1 and u_ens1, stacked across both
    eras so the Page-4 rolling disagreement plot covers 1992-2025.
    """
    crsp = pred_p2[["date", "permno", "p_ens1", "u_ens1"]].copy()
    crsp = crsp.rename(columns={"permno": "stock_id"})
    crsp["id_type"] = "permno"
    crsp["era"] = "1992-2015 (CRSP)"

    frames = [crsp]
    if pred_p2_ds is not None and not pred_p2_ds.empty:
        ext = pred_p2_ds[["date", "infocode", "p_ens1", "u_ens1"]].copy()
        ext = ext.rename(columns={"infocode": "stock_id"})
        ext["id_type"] = "infocode"
        ext["era"] = "2015-2025 (extension)"
        frames.append(ext)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["p_ens1", "u_ens1"])
    return out


# -----------------------------------------------------------------------------
# Pipeline metadata
# -----------------------------------------------------------------------------

def build_pipeline_metadata(pred_p1: pd.DataFrame,
                             pred_p2_ds: pd.DataFrame | None) -> dict:
    crsp_start = pd.Timestamp(pred_p1["date"].min()).strftime("%Y-%m-%d")
    crsp_end = pd.Timestamp(pred_p1["date"].max()).strftime("%Y-%m-%d")
    ext_start = ext_end = None
    if pred_p2_ds is not None and not pred_p2_ds.empty:
        ext_start = pd.Timestamp(pred_p2_ds["date"].min()).strftime("%Y-%m-%d")
        ext_end = pd.Timestamp(pred_p2_ds["date"].max()).strftime("%Y-%m-%d")

    return {
        "eras": {
            "crsp": {"start": crsp_start, "end": crsp_end,
                      "universe": "S&P 500 constituents (CRSP, no lookahead)"},
            "extension": {"start": ext_start, "end": ext_end,
                            "universe": "S&P 500 constituents via Datastream (US only)"},
        },
        "models": {
            "RF": {
                "type": "Random Forest classifier (scikit-learn)",
                "source": "src/krauss/models/rf_phase1.py:29-49",
                "hyperparameters": {
                    "n_estimators": 1000, "max_depth": 20,
                    "max_features": "floor(sqrt(31)) = 5",
                    "bootstrap": True, "max_samples": 0.632,
                    "criterion": "entropy", "random_state": 1,
                },
                "target": "y_binary",
            },
            "XGB": {
                "type": "XGBoost classifier",
                "source": "src/krauss/models/xgb_phase1.py:32-51",
                "hyperparameters": {
                    "n_estimators": 100, "max_depth": 3,
                    "learning_rate": 0.1,
                    "colsample_bynode": "15/31 = 0.484",
                    "min_child_weight": 10,
                    "reg_lambda": 0, "reg_alpha": 0, "gamma": 1e-5,
                    "max_bin": 20, "objective": "binary:logistic",
                    "eval_metric": "logloss",
                },
                "target": "y_binary",
            },
            "DNN": {
                "type": "Feed-forward NN with maxout activations (PyTorch)",
                "source": "src/krauss/models/dnn_phase1.py:63-304",
                "hyperparameters": {
                    "architecture": "31 -> Maxout(31) -> Maxout(10) -> Maxout(5) -> Linear(2)",
                    "maxout_channels": 2,
                    "dropout_input": 0.1, "dropout_hidden": 0.5,
                    "optimizer": "Adadelta(lr=1.0, rho=0.99, eps=1e-8)",
                    "loss": "CrossEntropyLoss",
                    "epochs": 400, "batch_size": 1024,
                    "l1_lambda": 1e-5,
                    "early_stop_window": 5, "early_stop_patience": 5,
                },
                "target": "y_binary",
            },
            "MT-DNN": {
                "type": "Multi-task DNN with shared trunk + cls and reg heads",
                "source": "src/krauss/models/dnn_multitask.py:55-275",
                "hyperparameters": {
                    "trunk": "31 -> Maxout(31) -> Maxout(10) -> Maxout(5)",
                    "cls_head": "Linear(5, 1) + sigmoid",
                    "reg_head": "Linear(5, 1)",
                    "loss": "0.5 * BCE + 0.5 * Huber",
                    "other": "same schedule as Phase-1 DNN",
                },
                "target": "y_binary and u_excess",
                "trained_on": "CRSP Phase 2 only — not trained on the Datastream extension",
            },
            "RF cls+reg pair": {
                "type": "RandomForestClassifier + RandomForestRegressor",
                "source": "src/krauss/models/rf_extension.py:22-49",
                "hyperparameters": {
                    "classifier": "identical to RF Phase 1",
                    "regressor": "same tree params, criterion='squared_error'",
                },
                "target": "y_binary (cls) and u_excess (reg)",
            },
            "XGB cls+reg pair": {
                "type": "XGBClassifier + XGBRegressor",
                "source": "src/krauss/models/xgb_extension.py:22-58",
                "hyperparameters": {
                    "classifier": "identical to XGB Phase 1",
                    "regressor": "same params, objective='reg:pseudohubererror'",
                },
                "target": "y_binary (cls) and u_excess (reg)",
            },
            "ENS1": {
                "type": "Equal-weight mean of DNN, XGB, RF predictions",
                "source": "src/krauss/models/ensembles_phase1.py:14-24 (Phase 1); "
                            "src/krauss/models/ensembles_phase2.py:16-32 (Phase 2)",
                "formula": "(p_dnn + p_xgb + p_rf) / 3  — Phase-2 ENS1 also "
                            "averages u_dnn/u_xgb/u_rf the same way",
            },
            "ENS2": {
                "type": "Gini-weighted mean of DNN, XGB, RF (Phase 1 only)",
                "source": "src/krauss/models/ensembles_phase1.py:37-70",
                "formula": "weights proportional to training-period Gini = 2*AUC - 1; "
                            "falls back to ENS1 if all Gini <= 0",
            },
            "ENS3": {
                "type": "Rank-weighted mean of DNN, XGB, RF (Phase 1 only)",
                "source": "src/krauss/models/ensembles_phase1.py:73-101",
                "formula": "w_i = (1/R_i) / sum_j(1/R_j), R_i = Gini rank "
                            "(1 best, 3 worst) — Aiolfi & Timmermann 2006",
            },
        },
        "scoring_schemes": {
            "P-only":        "rank by P_hat (direction prediction only)",
            "U-only":        "rank by U_hat (excess-return magnitude only)",
            "Z-comp":        "rank by 0.5 * z(P_hat) + 0.5 * z(U_hat) per day",
            "Product":       "rank by (2*P_hat - 1) * U_hat (signed)",
            "P-gate(0.03)":  "longs: P>0.53, top k by U desc; shorts: P<0.47, bottom k by U asc",
            "P-gate(0.05)":  "longs: P>0.55, top k by U desc; shorts: P<0.45, bottom k by U asc",
        },
        "backtest": {
            "k_long": K, "k_short": K,
            "rebalance": "daily, close-to-close",
            "weighting": "equal weight per leg, dollar neutral",
            "cost_bps_per_half_turn": COST_BPS_PER_HALF_TURN,
            "cost_regimes": {
                "no_cost":         "pre-cost gross returns",
                "5bps_half_turn":  "5 bps per half-turn (paper convention)",
            },
        },
        "data_sources": {
            "crsp": "CRSP daily stock file via WRDS",
            "datastream": "Refinitiv Datastream (US-only extension)",
            "sp500_index": "WRDS S&P 500 daily index",
        },
        "app_build_timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _maybe_read(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        _log(f"SKIP (missing): {path}")
        return None
    return pd.read_parquet(path, **kwargs)


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Load inputs --------------------------------------------------------
    _log("Loading CRSP predictions + returns...")
    pred_p1 = pd.read_parquet(PROCESSED / "predictions_phase1.parquet")
    pred_p2 = pd.read_parquet(PROCESSED / "predictions_phase2.parquet")
    returns_crsp = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    returns_crsp["ret"] = returns_crsp["ret"].astype("float64")

    vix_path = RAW / "vix_daily.parquet"
    if not vix_path.exists():
        raise FileNotFoundError(
            f"Missing {vix_path}. Run `python scripts/fetch_vix.py` first."
        )
    vix = pd.read_parquet(vix_path)

    # Extension era (optional)
    pred_p2_ds = _maybe_read(STAGING / "predictions_phase2_ds.parquet")
    returns_ds = _maybe_read(STAGING / "ds_daily_returns_usonly.parquet")
    if returns_ds is not None:
        returns_ds["ret"] = returns_ds["ret"].astype("float64")

    # ---- Equity curves ------------------------------------------------------
    _log("Building CRSP equity curves...")
    eq_crsp = build_era_crsp(pred_p1, pred_p2, returns_crsp)

    if pred_p2_ds is not None and returns_ds is not None:
        _log("Building 2015-2025 extension equity curves...")
        eq_ext = build_era_extension(pred_p2_ds, returns_ds)
    else:
        eq_ext = pd.DataFrame()

    equity_sparse = pd.concat([eq_crsp, eq_ext], ignore_index=True)

    # Compute the matched-days anchor BEFORE densification — see the
    # docstring on ``_matched_days_per_era``.  Pre-densification, the anchor
    # only has rows on the ~2,703 / 792 days the gate actually produced a
    # portfolio, so ``ret.notna()`` correctly identifies trade days.
    matched_by_era = _matched_days_per_era(equity_sparse)
    for era, days in matched_by_era.items():
        _log(f"  matched-days anchor · {era}: {len(days):,} dates")

    _log(f"Densifying equity curves over per-era trading-day axes…")
    equity = densify_equity_curves(equity_sparse)
    eq_path = APP_DATA / "equity_curves.parquet"
    _write_parquet(equity, eq_path)
    _log(f"Wrote {eq_path}  ({len(equity):,} rows, "
         f"{equity['date'].nunique():,} distinct dates)")

    # ---- Summary table ------------------------------------------------------
    # Use the **sparse** frame for summary metrics so gated schemes' daily_
    # return is the mean over actually-traded days, not the calendar-day mean
    # diluted by densified zero rows.  The dense frame is for the simulator
    # and chart layers, which want a continuous time series.
    summary = build_summary(equity_sparse, matched_by_era=matched_by_era)
    sum_path = APP_DATA / "summary_table.parquet"
    _write_parquet(summary, sum_path)
    _log(f"Wrote {sum_path}  ({len(summary):,} rows)")

    # ---- Regime labels ------------------------------------------------------
    regimes = build_regime_labels(vix, equity)
    rg_path = APP_DATA / "regime_labels.parquet"
    _write_parquet(regimes, rg_path)
    _log(f"Wrote {rg_path}  ({len(regimes):,} rows)")

    # ---- Daily holdings (Phase 4; built if data present) --------------------
    if pred_p2_ds is not None and returns_ds is not None:
        holdings = build_daily_holdings(pred_p2, pred_p2_ds, returns_crsp,
                                          returns_ds, permno_ticker=None)
        if not holdings.empty:
            hold_path = APP_DATA / "daily_holdings.parquet"
            _write_parquet(holdings, hold_path)
            _log(f"Wrote {hold_path}  ({len(holdings):,} rows)")

    # ---- SPY benchmark (Phase 4b) -------------------------------------------
    spy_path = APP_DATA / "spy_benchmark.parquet"
    try:
        spy = build_spy_benchmark()
        _write_parquet(spy, spy_path)
        _log(f"Wrote {spy_path}  ({len(spy):,} rows, "
             f"{spy['date'].min().date()} to {spy['date'].max().date()})")
    except Exception as exc:  # noqa: BLE001 — offline / no yfinance
        _log(f"SPY fetch failed ({exc}); skipping. App still runs if "
             f"spy_benchmark.parquet already exists.")

    # ---- Authoritative vendored tables (Phase 3) ---------------------------
    cost_bands = build_cost_band_table()
    cb_path = APP_DATA / "cost_bands.parquet"
    _write_parquet(cost_bands, cb_path)
    _log(f"Wrote {cb_path}  ({len(cost_bands):,} rows)")

    reg_ks = build_regime_k_sensitivity_table()
    rk_path = APP_DATA / "regime_k_sensitivity.parquet"
    _write_parquet(reg_ks, rk_path)
    _log(f"Wrote {rk_path}  ({len(reg_ks):,} rows)")

    reg_leg = build_regime_leg_decomp_table()
    rl_path = APP_DATA / "regime_leg_decomp.parquet"
    _write_parquet(reg_leg, rl_path)
    _log(f"Wrote {rl_path}  ({len(reg_leg):,} rows)")

    # ---- Disagreement panel (Phase 2) ---------------------------------------
    disagree = build_disagreement_panel(pred_p2, pred_p2_ds)
    dis_path = APP_DATA / "disagreement_panel.parquet"
    # Disagreement panel is the only file where categorical encoding gives
    # material storage savings (4.1M rows).  Opt in here only.
    _write_parquet(disagree, dis_path, categorize=True)
    _log(f"Wrote {dis_path}  ({len(disagree):,} rows)")

    # ---- Metadata -----------------------------------------------------------
    meta = build_pipeline_metadata(pred_p1, pred_p2_ds)
    meta_path = APP_DATA / "pipeline_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    _log(f"Wrote {meta_path}")

    _log(f"Done in {time.time() - t0:0.1f}s.")


if __name__ == "__main__":
    main()
