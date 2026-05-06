"""High-level simulator API for the web app.

This is the single contract the frontend should use.  Two entry points:

    run_backtest(...)   — fixed start/end window, returns BacktestResult
    run_simulation(...) — fixed start, realised P&L through the latest signal
                          date with next-day returns, plus ``next_positions``
                          populated for tomorrow

The implementation glues together
    - predictions_unified.parquet     (predictions panel)
    - returns_unified.parquet         (next-day returns for backtest P&L)
    - krauss.backtest.{ranking, portfolio, costs, no_trade_band}
    - krauss.backtest.annualization

Identifier convention: every dataframe returned from this API is keyed on
``infocode`` (canonical) and carries a ``ticker`` column for display.

Valid (era, scheme) combinations:
    1992-2015 (phase1_h2o):   only "P-only" -- no Û available
    2015-today (phase2_ds*):  all six schemes

Trying to run an extension scheme (U-only / Z-comp / Product / P-gate*) on
a window that includes pre-2015-10-16 dates raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from krauss.backtest.annualization import (
    annualised_vol,
    cagr,
    calmar_ratio,
    sharpe_annual,
    sortino_annual,
)
from krauss.backtest.costs import apply_transaction_costs, compute_turnover
from krauss.backtest.no_trade_band import backtest_with_band
from krauss.backtest.portfolio import (
    aggregate_portfolio_returns,
    build_daily_portfolios,
)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PREDS_PATH = PROCESSED / "predictions_unified.parquet"
DEFAULT_RETURNS_PATH = PROCESSED / "returns_unified.parquet"

FAMILIES: tuple[str, ...] = ("DNN", "XGB", "RF", "ENS1", "ENS2", "ENS3")
SCHEMES: tuple[str, ...] = (
    "P-only", "U-only", "Z-comp", "Product",
    "P-gate(0.03)", "P-gate(0.05)",
)
PHASE2_ERA_START = pd.Timestamp("2015-10-16")

# Default cost convention (paper-aligned)
DEFAULT_HALF_TURN_BPS = 5.0
DEFAULT_BAND_THRESHOLD_BPS = 10.0
DEFAULT_RF_ANNUAL = 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Output of run_backtest / run_simulation.

    Attributes
    ----------
    daily : pd.DataFrame
        One row per trade date.  Columns:
            date, port_ret, port_ret_net, turnover, cost,
            long_ret, short_ret, n_long, n_short
    equity_curve : pd.Series
        Compounded post-cost wealth (= (1 + port_ret_net).cumprod()),
        indexed by date.  Starts at 1.0.
    holdings : pd.DataFrame
        One row per stock-day position held.  Columns:
            date, infocode, ticker, side ('long' / 'short'), weight,
            next_day_ret, contrib
    risk_metrics : dict
        Headline risk + return numbers computed on ``daily['port_ret_net']``::
            cagr, ann_vol, sharpe, sortino, max_drawdown, calmar,
            hit_rate, total_return, mean_daily, n_days,
            avg_turnover, avg_n_long, avg_n_short
    next_positions : pd.DataFrame | None
        Only populated by ``run_simulation`` (live mode).  Shows the
        recommended top-k longs and bottom-k shorts from the latest
        available prediction date.  Columns:
            date, infocode, ticker, side, score, p, u
    params : dict
        Echo of the parameters used (family, scheme, k, dates, costs, band).
    """
    daily: pd.DataFrame
    equity_curve: pd.Series
    holdings: pd.DataFrame
    risk_metrics: dict
    next_positions: Optional[pd.DataFrame] = None
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
_FAMILY_TO_TAG = {"DNN": "dnn", "XGB": "xgb", "RF": "rf", "ENS1": "ens1",
                  "ENS2": "ens2", "ENS3": "ens3"}


def _validate(family: str, scheme: str, start: pd.Timestamp, end: pd.Timestamp,
              k: int, no_trade_band: bool) -> None:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if start > end:
        raise ValueError(
            f"start ({start.date()}) must be on or before end ({end.date()})"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scheme != "P-only" and start < PHASE2_ERA_START:
        raise ValueError(
            f"Scheme {scheme!r} requires Û (only available 2015-10-16+); "
            f"start={start.date()} is before the Phase 2 era. "
            f"Use 'P-only' for pre-2015-10-16 windows or move start "
            f"to {PHASE2_ERA_START.date()}+."
        )
    if family in ("ENS2", "ENS3") and (
        scheme != "P-only" or end > pd.Timestamp("2015-10-15")
    ):
        # ENS2/ENS3 only exist in the Phase 1 era and only support P-only.
        raise ValueError(
            f"family {family!r} only has predictions in the 1992-2015 "
            f"Phase 1 H2O era and only with the 'P-only' scheme."
        )
    if no_trade_band and scheme not in ("U-only", "Z-comp",
                                         "P-gate(0.03)", "P-gate(0.05)"):
        raise ValueError(
            f"no_trade_band=True only supports schemes that produce a Û-based "
            f"score: U-only / Z-comp / P-gate(0.03) / P-gate(0.05).  "
            f"Got scheme={scheme!r}."
        )


def valid_scheme_for_era(start: pd.Timestamp, end: pd.Timestamp,
                         scheme: str) -> bool:
    """Convenience for the frontend: would (start, end, scheme) validate?"""
    if scheme not in SCHEMES:
        return False
    try:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
    except (TypeError, ValueError):
        return False
    if pd.isna(start) or pd.isna(end) or start > end:
        return False
    if scheme != "P-only" and start < PHASE2_ERA_START:
        return False
    return True


def _load_predictions(path: Path | None) -> pd.DataFrame:
    p = path or DEFAULT_PREDS_PATH
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df["infocode"] = df["infocode"].astype("Int64")
    return df


def _load_returns(path: Path | None) -> pd.DataFrame:
    p = path or DEFAULT_RETURNS_PATH
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df["infocode"] = df["infocode"].astype("Int64")
    # The shared backtest engine expects column ``permno``; we feed it
    # infocode renamed to permno for compatibility.
    return df[["date", "infocode", "ret"]].rename(columns={"infocode": "permno"})


def _materialise_score_columns(preds: pd.DataFrame, family: str, scheme: str
                               ) -> pd.DataFrame:
    """Add ``score_long``, ``score_short`` columns to ``preds`` per scheme.

    Returns a copy of ``preds`` with new columns.  Used by both rank-based
    backtests (top-k by score_long, bottom-k by score_short) and the
    no-trade band wrapper.
    """
    tag = _FAMILY_TO_TAG[family]
    p_col = f"p_{tag}"
    u_col = f"u_{tag}"
    out = preds.copy()

    if scheme == "P-only":
        out["score_long"] = out[p_col]
        out["score_short"] = out[p_col]
    elif scheme == "U-only":
        out["score_long"] = out[u_col]
        out["score_short"] = out[u_col]
    elif scheme == "Z-comp":
        # 0.5 * z(P) + 0.5 * z(U) per day, cross-sectionally.
        grp_p = out.groupby("date")[p_col]
        grp_u = out.groupby("date")[u_col]
        z_p = (out[p_col] - grp_p.transform("mean")) / grp_p.transform("std")
        z_u = (out[u_col] - grp_u.transform("mean")) / grp_u.transform("std")
        zc = 0.5 * z_p + 0.5 * z_u
        out["score_long"] = zc
        out["score_short"] = zc
    elif scheme == "Product":
        prod = (2 * out[p_col] - 1) * out[u_col]
        out["score_long"] = prod
        out["score_short"] = prod
    elif scheme.startswith("P-gate("):
        thresh = float(scheme[len("P-gate("):-1])
        out["score_long"] = out[u_col].where(out[p_col] > 0.5 + thresh)
        out["score_short"] = out[u_col].where(out[p_col] < 0.5 - thresh)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")
    return out


def _select_long_short(preds: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select top-k longs and bottom-k shorts from materialised score columns."""
    records: list[pd.DataFrame] = []
    cols = ["date", "permno", "score_long", "score_short"]
    df = preds[cols].copy()

    for _, day in df.groupby("date", sort=True):
        longs = (
            day.dropna(subset=["score_long"])
            .sort_values(["score_long", "permno"], ascending=[False, True])
            .head(k)
            .copy()
        )
        if len(longs):
            longs["rank"] = range(1, len(longs) + 1)
            longs["side"] = "long"
            longs["score"] = longs["score_long"]
            records.append(longs[["date", "permno", "rank", "side", "score"]])

        shorts = (
            day.dropna(subset=["score_short"])
            .sort_values(["score_short", "permno"], ascending=[True, True])
            .head(k)
            .copy()
        )
        if len(shorts):
            shorts["rank"] = range(1, len(shorts) + 1)
            shorts["side"] = "short"
            shorts["score"] = shorts["score_short"]
            records.append(shorts[["date", "permno", "rank", "side", "score"]])

    if not records:
        return pd.DataFrame(columns=["date", "permno", "rank", "side", "score"])
    return pd.concat(records, ignore_index=True)


def _empty_daily() -> pd.DataFrame:
    """Daily-result frame with the public schema but no rows."""
    return pd.DataFrame(
        columns=[
            "date", "next_date", "port_ret", "long_ret", "short_ret",
            "n_long", "n_short", "turnover", "cost", "port_ret_net",
        ]
    )


def _empty_holdings() -> pd.DataFrame:
    """Holdings frame with the public schema but no rows."""
    return pd.DataFrame(
        columns=[
            "date", "infocode", "ticker", "side", "weight",
            "next_day_ret", "contrib",
        ]
    )


def _risk_metrics(daily: pd.DataFrame) -> dict:
    """Compute headline risk + return numbers from a daily-returns frame."""
    r = daily["port_ret_net"]
    if len(r) == 0:
        return {
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "hit_rate": float("nan"),
            "total_return": float("nan"),
            "mean_daily": float("nan"),
            "n_days": 0,
            "avg_turnover": float("nan"),
            "avg_n_long": float("nan"),
            "avg_n_short": float("nan"),
        }
    if len(r) < 2:
        return {k: float("nan") for k in (
            "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "calmar", "hit_rate", "total_return", "mean_daily", "n_days",
            "avg_turnover", "avg_n_long", "avg_n_short",
        )}
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "cagr": cagr(r),
        "ann_vol": annualised_vol(r),
        "sharpe": sharpe_annual(r, rf_annual=DEFAULT_RF_ANNUAL),
        "sortino": sortino_annual(r, rf_annual=DEFAULT_RF_ANNUAL),
        "max_drawdown": dd,
        "calmar": calmar_ratio(r),
        "hit_rate": float((r > 0).mean()),
        "total_return": float(eq.iloc[-1] - 1.0),
        "mean_daily": float(r.mean()),
        "n_days": int(len(r)),
        "avg_turnover": float(daily["turnover"].mean())
            if "turnover" in daily.columns else float("nan"),
        "avg_n_long": float(daily["n_long"].mean())
            if "n_long" in daily.columns else float("nan"),
        "avg_n_short": float(daily["n_short"].mean())
            if "n_short" in daily.columns else float("nan"),
    }


def _attach_tickers(holdings: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Add a ticker column to a holdings frame using the predictions panel."""
    tmap = (preds[["infocode", "ticker"]]
            .dropna(subset=["ticker"])
            .drop_duplicates(subset="infocode", keep="last"))
    if "permno" in holdings.columns:
        holdings = holdings.rename(columns={"permno": "infocode"})
    return holdings.merge(tmap, on="infocode", how="left")


def _latest_realized_signal_date(returns: pd.DataFrame) -> pd.Timestamp | None:
    """Latest signal date with at least one later return date available."""
    dates = pd.Series(pd.to_datetime(returns["date"].dropna().unique()))
    dates = dates.sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return None
    return pd.Timestamp(dates.iloc[-2])


def _build_next_positions(
    preds: pd.DataFrame,
    *,
    family: str,
    scheme: str,
    k: int,
    last_pred_date: pd.Timestamp,
    start: pd.Timestamp,
) -> pd.DataFrame | None:
    """Build the latest available long/short recommendation table."""
    next_day_preds = preds[preds["date"] == last_pred_date].copy()
    if len(next_day_preds) == 0 or last_pred_date < start:
        return None

    scored = _materialise_score_columns(next_day_preds, family, scheme)
    tag = _FAMILY_TO_TAG[family]
    p_col = f"p_{tag}"
    u_col = f"u_{tag}"

    long_pool = scored.dropna(subset=["score_long"]).copy()
    long_pool = long_pool.nlargest(k, "score_long")
    long_pool["side"] = "long"
    long_pool["score"] = long_pool["score_long"]

    short_pool = scored.dropna(subset=["score_short"]).copy()
    short_pool = short_pool.nsmallest(k, "score_short")
    short_pool["side"] = "short"
    short_pool["score"] = short_pool["score_short"]

    next_positions = pd.concat([long_pool, short_pool], ignore_index=True)
    next_positions = next_positions.rename(columns={p_col: "p", u_col: "u"})
    cols = ["date", "infocode", "ticker", "side", "score"]
    if "p" in next_positions.columns:
        cols.append("p")
    if "u" in next_positions.columns:
        cols.append("u")
    return (
        next_positions[cols].reset_index(drop=True)
        if all(c in next_positions.columns for c in cols)
        else next_positions.reset_index(drop=True)
    )


def _build_band_next_positions(
    preds: pd.DataFrame,
    rets: pd.DataFrame,
    *,
    family: str,
    scheme: str,
    k: int,
    last_pred_date: pd.Timestamp,
    start: pd.Timestamp,
    band_threshold_bps: float,
    half_turn_cost_bps: float,
) -> pd.DataFrame | None:
    """Build latest positions after applying the path-dependent band rule."""
    if last_pred_date < start:
        return None

    window = preds[(preds["date"] >= start) & (preds["date"] <= last_pred_date)]
    if window.empty:
        return None

    scored = _materialise_score_columns(window, family, scheme)
    tag = _FAMILY_TO_TAG[family]
    u_col = f"u_{tag}"
    p_col = f"p_{tag}"

    out = backtest_with_band(
        predictions=scored.rename(columns={"infocode": "permno"}),
        returns=rets,
        k=k,
        long_score_col="score_long",
        short_score_col="score_short",
        u_col=u_col,
        half_turn_bps=half_turn_cost_bps,
        swap_threshold_bps=band_threshold_bps,
    )
    latest = out["holdings"].copy()
    latest["date"] = pd.to_datetime(latest["date"])
    latest = latest[latest["date"] == last_pred_date].copy()
    if latest.empty:
        return None

    day_scores = scored[scored["date"] == last_pred_date][
        ["date", "infocode", "ticker", p_col, u_col, "score_long", "score_short"]
    ].copy()
    latest = latest.rename(columns={"permno": "infocode"})
    latest = latest.merge(day_scores, on=["date", "infocode"], how="left")
    latest["score"] = latest["score_long"].where(
        latest["side"] == "long", latest["score_short"]
    )
    latest = latest.rename(columns={p_col: "p", u_col: "u"})
    longs = latest[latest["side"] == "long"].sort_values(
        ["score", "infocode"], ascending=[False, True]
    )
    shorts = latest[latest["side"] == "short"].sort_values(
        ["score", "infocode"], ascending=[True, True]
    )
    latest = pd.concat([longs, shorts], ignore_index=True)
    return latest[["date", "infocode", "ticker", "side", "score", "p", "u"]]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def run_backtest(
    *,
    family: Literal["DNN", "XGB", "RF", "ENS1", "ENS2", "ENS3"],
    scheme: Literal["P-only", "U-only", "Z-comp", "Product",
                    "P-gate(0.03)", "P-gate(0.05)"],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    k: int = 10,
    no_trade_band: bool = False,
    band_threshold_bps: float = DEFAULT_BAND_THRESHOLD_BPS,
    half_turn_cost_bps: float = DEFAULT_HALF_TURN_BPS,
    predictions_path: Path | None = None,
    returns_path: Path | None = None,
) -> BacktestResult:
    """Run a fixed-window backtest of one (family, scheme) configuration.

    All parameters are keyword-only.  See ``BacktestResult`` for the shape
    of the returned object.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    _validate(family, scheme, start, end, k, no_trade_band)

    preds = _load_predictions(predictions_path)
    rets = _load_returns(returns_path)

    # Restrict to the requested window
    preds = preds[(preds["date"] >= start) & (preds["date"] <= end)].copy()
    if len(preds) == 0:
        raise ValueError(
            f"No predictions found in window {start.date()} -> {end.date()}"
        )

    preds = _materialise_score_columns(preds, family, scheme)
    # The shared engine joins predictions to returns on ``permno``.
    preds_for_engine = preds.rename(columns={"infocode": "permno"})

    if no_trade_band:
        # u_col is the per-family Û.
        u_col = f"u_{_FAMILY_TO_TAG[family]}"
        out = backtest_with_band(
            predictions=preds_for_engine,
            returns=rets,
            k=k,
            long_score_col="score_long",
            short_score_col="score_short",
            u_col=u_col,
            half_turn_bps=half_turn_cost_bps,
            swap_threshold_bps=band_threshold_bps,
        )
        daily = out["daily"]
        holdings = out["holdings"]
    else:
        sel = _select_long_short(preds_for_engine, k=k)
        if len(sel) == 0:
            raise RuntimeError(
                f"No selections produced -- check (family={family}, scheme={scheme}, "
                f"window={start.date()}->{end.date()})"
            )
        holdings = build_daily_portfolios(sel, rets, k=k)
        daily = aggregate_portfolio_returns(holdings)
        daily["date"] = pd.to_datetime(daily["date"])
        turn = compute_turnover(holdings, k=k)
        daily = apply_transaction_costs(daily, turn, half_turn_cost_bps)

    daily = daily.sort_values("date").reset_index(drop=True)
    holdings = _attach_tickers(holdings, preds)

    eq = (1.0 + daily["port_ret_net"]).cumprod()
    eq.index = daily["date"]
    eq.name = "equity"

    metrics = _risk_metrics(daily)

    params = {
        "family": family, "scheme": scheme, "k": k,
        "start": str(start.date()), "end": str(end.date()),
        "no_trade_band": no_trade_band,
        "band_threshold_bps": band_threshold_bps if no_trade_band else None,
        "half_turn_cost_bps": half_turn_cost_bps,
    }
    return BacktestResult(
        daily=daily, equity_curve=eq, holdings=holdings,
        risk_metrics=metrics, next_positions=None, params=params,
    )


def run_simulation(
    *,
    family: Literal["DNN", "XGB", "RF", "ENS1", "ENS2", "ENS3"],
    scheme: Literal["P-only", "U-only", "Z-comp", "Product",
                    "P-gate(0.03)", "P-gate(0.05)"],
    start: pd.Timestamp | str,
    k: int = 10,
    no_trade_band: bool = False,
    band_threshold_bps: float = DEFAULT_BAND_THRESHOLD_BPS,
    half_turn_cost_bps: float = DEFAULT_HALF_TURN_BPS,
    predictions_path: Path | None = None,
    returns_path: Path | None = None,
) -> BacktestResult:
    """Live simulation: realised backtest plus next-session positions.

    Equivalent to ``run_backtest`` from ``start`` through the latest signal
    date with next-day returns available, but additionally populates
    ``next_positions`` from the latest prediction date.
    """
    start = pd.Timestamp(start)

    preds = _load_predictions(predictions_path)
    rets_raw = _load_returns(returns_path)

    last_return_date = pd.Timestamp(rets_raw["date"].max())
    last_pred_date = pd.Timestamp(preds["date"].max())
    if start > last_pred_date:
        raise ValueError(
            f"start ({start.date()}) is after the last prediction date "
            f"({last_pred_date.date()})"
        )
    _validate(family, scheme, start, last_pred_date, k, no_trade_band)

    last_realized_signal_date = _latest_realized_signal_date(rets_raw)
    if last_realized_signal_date is None:
        backtest_end = None
    else:
        backtest_end = min(last_realized_signal_date, last_pred_date)

    if backtest_end is not None and start <= backtest_end:
        result = run_backtest(
            family=family, scheme=scheme, start=start, end=backtest_end, k=k,
            no_trade_band=no_trade_band, band_threshold_bps=band_threshold_bps,
            half_turn_cost_bps=half_turn_cost_bps,
            predictions_path=predictions_path, returns_path=returns_path,
        )
    else:
        daily = _empty_daily()
        equity = pd.Series(dtype="float64", name="equity")
        params = {
            "family": family, "scheme": scheme, "k": k,
            "start": str(start.date()), "end": None,
            "no_trade_band": no_trade_band,
            "band_threshold_bps": band_threshold_bps if no_trade_band else None,
            "half_turn_cost_bps": half_turn_cost_bps,
        }
        result = BacktestResult(
            daily=daily,
            equity_curve=equity,
            holdings=_empty_holdings(),
            risk_metrics=_risk_metrics(daily),
            next_positions=None,
            params=params,
        )

    # Next-day positions: top-k longs / bottom-k shorts on the LATEST
    # available prediction date.  These are the picks for the next session.
    if no_trade_band:
        result.next_positions = _build_band_next_positions(
            preds,
            rets_raw,
            family=family,
            scheme=scheme,
            k=k,
            last_pred_date=last_pred_date,
            start=start,
            band_threshold_bps=band_threshold_bps,
            half_turn_cost_bps=half_turn_cost_bps,
        )
    else:
        result.next_positions = _build_next_positions(
            preds,
            family=family,
            scheme=scheme,
            k=k,
            last_pred_date=last_pred_date,
            start=start,
        )

    result.params["mode"] = "simulation"
    result.params["last_return_date"] = str(last_return_date.date())
    result.params["last_pred_date"] = str(last_pred_date.date())
    result.params["last_realized_signal_date"] = (
        str(backtest_end.date()) if backtest_end is not None else None
    )
    return result
