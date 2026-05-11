"""Page 7 — Simulator.

Replays pre-computed daily returns into a what-would-have-happened equity
curve for a user-selected (start, end, scheme, model, cost regime, capital)
window.  No live backtests — it re-aggregates ``equity_curves.parquet`` into
the metrics the user actually wants to see when comparing across windows.

The unmistakable disclaimer at the bottom of the page makes the historical-
only nature of the simulation clear: nothing here is a live prediction or a
recommendation to trade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Simulator", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_equity_curves, load_spy_benchmark,
)

if not data_build_is_complete():
    missing_build_warning()

equity = load_equity_curves()
spy = load_spy_benchmark()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) defines "
    "Sharpe, Sortino, max drawdown and the rest of the vocabulary used "
    "below."
)

st.title("Simulator")
st.caption(
    "Pick a date window and a strategy variant. The page replays the "
    "pre-computed daily returns over your window and reports the equity "
    "curve, drawdown, and a 12-cell risk-metric panel — everything is "
    "historical."
)

# ---------------------------------------------------------------------------
# 1.  User inputs
# ---------------------------------------------------------------------------

eq_min = pd.Timestamp(equity["date"].min()).date()
eq_max = pd.Timestamp(equity["date"].max()).date()

DEFAULT_START = pd.Timestamp("2010-01-01").date()
DEFAULT_END = pd.Timestamp("2015-10-15").date()

with st.container(border=True):
    st.markdown("**Inputs**")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input(
        "Start date",
        value=max(min(DEFAULT_START, eq_max), eq_min),
        min_value=eq_min, max_value=eq_max,
    )
    end = c2.date_input(
        "End date",
        value=max(min(DEFAULT_END, eq_max), eq_min),
        min_value=eq_min, max_value=eq_max,
    )
    capital = c3.number_input(
        "Initial capital ($)",
        min_value=1_000.0, max_value=10_000_000.0,
        value=10_000.0, step=1_000.0, format="%.2f",
    )

    SCHEME_CHOICES = ["P-only", "U-only", "Z-comp", "Product",
                      "P-gate(0.03)", "P-gate(0.05)"]
    MODEL_CHOICES = ["ENS1", "RF", "XGB", "DNN"]
    COST_REGIME_LABELS = {
        "Post-cost (5 bps/half-turn)": "5bps_half_turn",
        "Pre-cost (no costs)":          "no_cost",
    }

    c4, c5, c6 = st.columns(3)
    scheme = c4.selectbox("Scoring scheme", SCHEME_CHOICES, index=0)
    model = c5.selectbox("Model", MODEL_CHOICES, index=0)
    cost_label = c6.selectbox("Cost regime", list(COST_REGIME_LABELS),
                                 index=0)
    cost_regime = COST_REGIME_LABELS[cost_label]

    st.caption(
        "For the no-trade-band experiment, see the "
        "[Cost-aware execution](Cost-aware_execution) page — band-effect "
        "Sharpe deltas are tabulated there from the live notebook outputs."
    )

# ---------------------------------------------------------------------------
# 2.  Edge cases
# ---------------------------------------------------------------------------

if pd.Timestamp(end) < pd.Timestamp(start):
    st.error("End date must be on or after the start date.")
    st.stop()

start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

# ---------------------------------------------------------------------------
# 3.  Filter the strategy series
# ---------------------------------------------------------------------------

strat = equity.query(
    "model == @model and scheme == @scheme and cost_regime == @cost_regime"
).copy()
strat = strat[(strat["date"] >= start_ts) & (strat["date"] <= end_ts)]
strat = strat.sort_values("date").reset_index(drop=True)

if strat.empty:
    st.warning(
        "The selected window has no rows in the equity curves file. "
        "This usually means the window falls outside the era boundary "
        "for the chosen scheme — try widening the date range."
    )
    st.stop()

if len(strat) < 20:
    st.warning(
        f"The selected window has only {len(strat)} trading days. Risk "
        "metrics computed on fewer than ~20 days are noisy — interpret "
        "the numbers below as illustrative, not statistical."
    )

# ---------------------------------------------------------------------------
# 4.  Compute strategy + SPY equity curves
# ---------------------------------------------------------------------------

ret = strat["ret"].fillna(0.0).reset_index(drop=True)
strat["nav"] = capital * (1.0 + ret).cumprod()
strat_cum_ret = float(strat["nav"].iloc[-1] / capital - 1.0)

# SPY restricted to the same window, rebased to the same starting NAV.
spy_in = pd.DataFrame()
spy_cum_ret = np.nan
if spy is not None and not spy.empty:
    spy_in = spy[(spy["date"] >= start_ts) & (spy["date"] <= end_ts)].copy()
    spy_in = spy_in.sort_values("date").reset_index(drop=True)
    if not spy_in.empty:
        spy_in["ret"] = spy_in["ret"].fillna(0.0)
        spy_in["nav"] = capital * (1.0 + spy_in["ret"]).cumprod()
        spy_cum_ret = float(spy_in["nav"].iloc[-1] / capital - 1.0)

# ---------------------------------------------------------------------------
# 5.  A — Equity curve
# ---------------------------------------------------------------------------

st.subheader("A. Equity curve")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=strat["date"], y=strat["nav"], mode="lines",
    name=f"Strategy · {model} · {scheme} · {cost_label}",
    line=dict(color="#2e7d32", width=2),
    hovertemplate="%{x|%Y-%m-%d}<br>NAV = $%{y:,.2f}<extra></extra>",
))
if not spy_in.empty:
    fig.add_trace(go.Scatter(
        x=spy_in["date"], y=spy_in["nav"], mode="lines",
        name="SPY total return — for context only",
        line=dict(color="#888", width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>SPY NAV = $%{y:,.2f}<extra></extra>",
    ))
fig.update_layout(
    height=450,
    yaxis_title="Portfolio value ($)",
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.18),
    margin=dict(l=60, r=20, t=30, b=50),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 6.  B — Drawdown panel
# ---------------------------------------------------------------------------

st.subheader("B. Drawdown")

cum_curve = (1.0 + ret).cumprod()
running_max = cum_curve.cummax()
drawdown = cum_curve / running_max - 1.0
strat["drawdown"] = drawdown

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=strat["date"], y=strat["drawdown"] * 100,
    fill="tozeroy",
    mode="lines",
    line=dict(color="#c62828", width=1),
    fillcolor="rgba(198, 40, 40, 0.22)",
    name="Drawdown",
    hovertemplate="%{x|%Y-%m-%d}<br>drawdown = %{y:.2f}%<extra></extra>",
))
fig_dd.update_layout(
    height=240,
    yaxis_title="Drawdown (%)",
    xaxis_title=None,
    showlegend=False,
    hovermode="x",
    margin=dict(l=60, r=20, t=20, b=40),
)
fig_dd.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# 7.  C — Risk metrics grid (3 × 4)
# ---------------------------------------------------------------------------

st.subheader("C. Risk metrics")


def _annualised_mean(r: pd.Series) -> float:
    return float(252 * r.mean())


def _annualised_std(r: pd.Series) -> float:
    s = r.std(ddof=0)
    return float(np.sqrt(252) * s) if s > 0 else 0.0


def _sharpe(r: pd.Series) -> float:
    s = r.std(ddof=0)
    return float(np.sqrt(252) * r.mean() / s) if s > 0 else float("nan")


def _sortino(r: pd.Series) -> float:
    downside = r[r < 0]
    if len(downside) < 2:
        return float("nan")
    s = downside.std(ddof=0)
    return float(np.sqrt(252) * r.mean() / s) if s > 0 else float("nan")


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    cum = (1.0 + r).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _calmar(r: pd.Series) -> float:
    mdd = _max_drawdown(r)
    if mdd is np.nan or mdd == 0:
        return float("nan")
    return float(_annualised_mean(r) / abs(mdd))


def _alpha_vs(spy_ret: pd.Series, strat_ret: pd.Series) -> float:
    if len(spy_ret) < 2 or len(strat_ret) < 2:
        return float("nan")
    return _annualised_mean(strat_ret) - _annualised_mean(spy_ret)


def _correlation_with_spy(strat_dates: pd.Series, strat_ret: pd.Series,
                           spy_df: pd.DataFrame) -> float:
    if spy_df.empty:
        return float("nan")
    paired = (
        pd.DataFrame({"date": strat_dates.values, "strat": strat_ret.values})
        .merge(spy_df[["date", "ret"]], on="date", how="inner")
        .dropna()
    )
    if len(paired) < 2:
        return float("nan")
    return float(paired["strat"].corr(paired["ret"]))


sharpe = _sharpe(ret)
sortino = _sortino(ret)
max_dd = _max_drawdown(ret)
calmar = _calmar(ret)
ann_ret = _annualised_mean(ret)
ann_vol = _annualised_std(ret)
hit_rate = float((ret > 0).mean()) if len(ret) else float("nan")
worst = float(ret.min()) if len(ret) else float("nan")
best = float(ret.max()) if len(ret) else float("nan")

spy_returns = spy_in["ret"] if not spy_in.empty else pd.Series(dtype=float)
alpha_vs_spy = _alpha_vs(spy_returns, ret) if not spy_in.empty else float("nan")
corr_with_spy = _correlation_with_spy(strat["date"], ret, spy_in)


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _pct(v: float, digits: int = 2) -> str:
    if v is None or np.isnan(v):
        return "n/a"
    return f"{v * 100:+.{digits}f}%"


def _num(v: float, digits: int = 2) -> str:
    if v is None or np.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def metric(col, label: str, value: str, helptext: str = "") -> None:
    with col:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                f"<div style='font-size:22px;font-weight:700;color:#142B4F;"
                "margin-top:-4px;line-height:1.1'>" + value + "</div>",
                unsafe_allow_html=True,
            )
            if helptext:
                st.caption(helptext)


# Row 1
r1 = st.columns(3)
metric(r1[0], "Cumulative return",
        f"{_money(strat['nav'].iloc[-1] - capital)}  ({_pct(strat_cum_ret)})",
        f"Final NAV {_money(strat['nav'].iloc[-1])} from {_money(capital)} starting capital.")
metric(r1[1], "Annualised return", _pct(ann_ret),
        "Mean daily return × 252.")
metric(r1[2], "Annualised volatility", _pct(ann_vol),
        "Std-dev of daily return × √252.")

# Row 2
r2 = st.columns(3)
metric(r2[0], "Sharpe ratio", _num(sharpe),
        "Annualised return ÷ annualised volatility.")
metric(r2[1], "Sortino ratio", _num(sortino),
        "Annualised return ÷ annualised downside deviation.")
metric(r2[2], "Max drawdown", _pct(max_dd),
        "Worst peak-to-trough on the equity curve over this window.")

# Row 3
r3 = st.columns(3)
metric(r3[0], "Calmar ratio", _num(calmar),
        "Annualised return ÷ |max drawdown|.")
metric(r3[1], "Hit rate", _pct(hit_rate, digits=1),
        "Fraction of days with positive net return.")
metric(r3[2], "Best / worst day",
        f"{_pct(best)} / {_pct(worst)}",
        "Single-day extremes inside the window.")

# Row 4
r4 = st.columns(3)
metric(r4[0], "Trading days in window", f"{len(ret):,}",
        "Calendar days with a daily return row in the parquet.")
metric(r4[1], "Excess return vs SPY",
        _pct(alpha_vs_spy) if not np.isnan(alpha_vs_spy) else "n/a",
        "Strategy ann. return − SPY ann. return. "
        "Not regression-adjusted alpha.")
metric(r4[2], "Correlation with SPY",
        _num(corr_with_spy) if not np.isnan(corr_with_spy) else "n/a",
        "Pearson correlation of daily strategy and daily SPY returns.")

# ---------------------------------------------------------------------------
# 8.  D — SPY comparison callout
# ---------------------------------------------------------------------------

st.subheader("D. Strategy vs SPY")

if not spy_in.empty:
    diff = strat_cum_ret - spy_cum_ret
    with st.container(border=True):
        st.markdown(
            f"Your strategy returned **{_pct(strat_cum_ret)}** over this "
            f"window. SPY returned **{_pct(spy_cum_ret)}** over the same "
            f"window. Difference: **{_pct(diff)}**."
        )
        st.markdown(
            "*The strategy is dollar-neutral — it earns its return from the "
            "spread between its long and short legs, not from broad market "
            "direction. SPY is plotted as context, not as the formal "
            "benchmark a market-neutral strategy is judged against.*"
        )
else:
    st.info("SPY data not available; comparison skipped.")

# ---------------------------------------------------------------------------
# 9.  E — Disclaimer banner
# ---------------------------------------------------------------------------

st.warning(
    "⚠︎ **Historical simulation, not a live trading dashboard.** "
    "This page replays pre-computed daily returns from frozen model "
    "predictions trained on data through 2025-09-24. It shows what would "
    "have happened if you had deployed this strategy historically — not "
    "what would happen if you deployed it today. The strategy's "
    "**baseline (non-gated) schemes have materially decayed in the "
    "post-2015 era**; see the [Conclusion](Conclusion) page."
)
