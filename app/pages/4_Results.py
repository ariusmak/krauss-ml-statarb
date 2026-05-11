"""Page 4 — Extension results.

Adapted from the original Overview page.  Adds a KPI strip with the four
headline numbers from the deck, keeps the full 1992-2025 equity curve with
SPY overlay for honest context, and points the reader to the per-cell deep-
dive on the Results matrix appendix page.
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

st.set_page_config(page_title="Results", page_icon=":bar_chart:", layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_equity_curves, load_summary_table, load_pipeline_metadata,
    load_spy_benchmark,
)

if not data_build_is_complete():
    missing_build_warning()

equity = load_equity_curves()
summary = load_summary_table()
meta = load_pipeline_metadata()
spy = load_spy_benchmark()

st.info(
    ":bulb: **New to this?** Read the [Background primer](Background) first — "
    "and the [Paper overview](Paper_overview) and [Methodology](Methodology) "
    "pages frame what the numbers below mean."
)

st.title("Extension results")
st.caption(
    "Headline numbers from the full 1992-2025 walk-forward, then the equity "
    "curve with SPY overlaid for context. Per-cell drill-down lives on the "
    "[Results matrix](Results_matrix) appendix page."
)


# --- KPI strip --------------------------------------------------------------

def _sharpe(model: str, scheme: str, era: str,
             cost: str = "5bps_half_turn") -> float | None:
    row = summary.query(
        "model == @model and scheme == @scheme and era == @era "
        "and cost_regime == @cost"
    )
    return None if row.empty else float(row["sharpe"].iloc[0])


era1 = "1992-2015 (CRSP)"
era2 = "2015-2025 (extension)"

# Era 1 best scheme = P-gate(0.05) ENS1 (matched-days = full sample for the
# anchor scheme — see build_app_data._matched_days_per_era).
era1_best = _sharpe("ENS1", "P-gate(0.05)", era1)
era1_zcomp = _sharpe("ENS1", "Z-comp", era1)

# Era 2 "least-bad" — pick the highest Sharpe across our six schemes.
era2_pool = summary.query(
    "model == 'ENS1' and era == @era2 and cost_regime == '5bps_half_turn'"
)
if era2_pool.empty:
    era2_best, era2_best_scheme = None, "n/a"
else:
    best_row = era2_pool.sort_values("sharpe", ascending=False).iloc[0]
    era2_best = float(best_row["sharpe"])
    era2_best_scheme = str(best_row["scheme"])
era2_baseline = _sharpe("ENS1", "P-only", era2)


def _kpi(col, label: str, value: str, helptext: str,
          color: str = "#142B4F") -> None:
    with col:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                f"<div style='font-size:30px;font-weight:700;color:{color};"
                "margin-top:-6px'>" + value + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(helptext)


k1, k2, k3, k4 = st.columns(4)
_kpi(k1, "Era 1 best — P-gate(0.05) on ENS1",
      f"Sharpe {era1_best:0.2f}" if era1_best is not None else "n/a",
      "Post-cost. Trades 2,703 of 5,750 days — selectivity, not better "
      "prediction. Matched-days return = full-sample return.",
      color="#2e7d32")
_kpi(k2, "Era 1 best non-gated — Z-comp on ENS1",
      f"Sharpe {era1_zcomp:0.2f}" if era1_zcomp is not None else "n/a",
      "0.5·z(P̂) + 0.5·z(Û). The cleanest win that doesn't lean on day-"
      "selection.",
      color="#1f77b4")
era2_best_days = int(best_row["trading_days"]) if not era2_pool.empty else None
if era2_best_days:
    days_text = (
        "Trades 792 of 2,500 extension days (32%). A null test "
        "(10,000 random 792-day subsamples of P-only) places this "
        "Sharpe at the 99.8th percentile; signal, not lucky "
        "subsample selection."
    )
else:
    days_text = "Post-cost on the 2015-2025 extension."
_kpi(k3, f"Era 2 best — {era2_best_scheme} on ENS1",
      f"Sharpe {era2_best:0.2f}" if era2_best is not None else "n/a",
      days_text,
      color="#142B4F")
_kpi(k4, "Era 2 baseline — P-only on ENS1",
      f"Sharpe {era2_baseline:0.2f}" if era2_baseline is not None else "n/a",
      "Post-cost. The paper's headline scheme on the out-of-sample window "
      "— this is what alpha decay looks like.",
      color="#c62828")

# --- Full 1992-2025 equity curve -------------------------------------------

st.subheader("ENS1 P-only equity curve, 1992-2025, with SPY for context")

ens1 = equity.query("model == 'ENS1' and scheme == 'P-only'").copy()
ens1 = ens1.sort_values(["cost_regime", "date"])
ens1["ret"] = ens1["ret"].fillna(0.0)
ens1["stitched_cum"] = ens1.groupby("cost_regime", observed=True)["ret"].transform(
    lambda s: (1.0 + s).cumprod()
)

fig = go.Figure()
ens1_colors = {"no_cost": "#1f77b4", "5bps_half_turn": "#2e7d32"}
ens1_labels = {
    "no_cost": "ENS1 P-only · pre-cost",
    "5bps_half_turn": "ENS1 P-only · post-cost (5 bps/half-turn)",
}
for cr, grp in ens1.groupby("cost_regime", observed=True):
    fig.add_trace(go.Scatter(
        x=grp["date"], y=grp["stitched_cum"],
        mode="lines",
        name=ens1_labels[cr],
        line=dict(color=ens1_colors[cr], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>growth = %{y:.2f}x<extra></extra>",
    ))

if spy is not None and not spy.empty:
    start = ens1["date"].min()
    spy_in = spy[spy["date"] >= start].copy()
    spy_in["ret"] = spy_in["ret"].fillna(0.0)
    spy_in["cum_ret"] = (1.0 + spy_in["ret"]).cumprod()
    fig.add_trace(go.Scatter(
        x=spy_in["date"], y=spy_in["cum_ret"],
        mode="lines",
        name="SPY total return (long market exposure — for context only)",
        line=dict(color="#999", width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>SPY growth = %{y:.2f}x<extra></extra>",
    ))

fig.add_shape(
    type="line", x0="2015-10-16", x1="2015-10-16",
    y0=0, y1=1, yref="paper",
    line=dict(color="#666", width=1, dash="dash"),
)
fig.add_annotation(
    x="2015-10-16", y=1.0, yref="paper", yanchor="bottom",
    text="extension era →", showarrow=False,
    font=dict(size=11, color="#444"),
)

fig.update_layout(
    height=500,
    xaxis_title=None,
    yaxis_title="Cumulative return (log scale)",
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=50, r=20, t=30, b=60),
    hovermode="x unified",
)
fig.update_yaxes(type="log")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "ENS1 is the equal-weighted mean of RF, XGB and DNN direction "
    "predictions. **Y-axis is log-scale 'growth of \\$1'** — necessary "
    "because pre-cost cumulative returns over 33 years reach the "
    "billions, and a linear axis hides the post-cost line entirely. "
    "Before costs it climbs steeply across the CRSP era. After "
    "5 bps/half-turn, growth slows from 2008 onwards and reverses across "
    "the extension era (visible as the line declining from ~2015 onward). "
    "**SPY is plotted as context, not as a direct benchmark**: this "
    "strategy is dollar-neutral, so the honest comparator is cash plus "
    "alpha, not a 100 %-long index."
)

# --- Top-three-findings card row -------------------------------------------

st.subheader("Three findings worth carrying forward")

col1, col2, col3 = st.columns(3)
sharpe_ens1_precost = _sharpe("ENS1", "P-only", era1, cost="no_cost")
sharpe_ens1_postcost = _sharpe("ENS1", "P-only", era1)
sharpe_prod_postcost = _sharpe("ENS1", "Product", era1)

with col1:
    st.metric(
        "Baseline works, but costs matter",
        f"Sharpe {sharpe_ens1_postcost:0.2f}",
        delta=(f"{sharpe_ens1_postcost - sharpe_ens1_precost:+0.2f} vs pre-cost"
                if sharpe_ens1_postcost and sharpe_ens1_precost else None),
    )
    st.caption("ENS1 P-only, 1992-2015, k=10, post-cost.")

zcomp_row = summary.query(
    "model == 'ENS1' and scheme == 'Z-comp' and era == @era1 "
    "and cost_regime == '5bps_half_turn'"
)
pony_row = summary.query(
    "model == 'ENS1' and scheme == 'P-only' and era == @era1 "
    "and cost_regime == '5bps_half_turn'"
)
daily_zcomp = float(zcomp_row['daily_return'].iloc[0]) if not zcomp_row.empty else None
daily_pony = float(pony_row['daily_return'].iloc[0]) if not pony_row.empty else None
with col2:
    st.metric(
        "Z-score composite lifts daily return",
        f"{daily_zcomp*100:.2f}%" if daily_zcomp is not None else "n/a",
        delta=(f"{(daily_zcomp - daily_pony)*100:+.2f}pp vs P-only"
                if daily_zcomp is not None and daily_pony is not None else None),
    )
    st.caption(
        "Z-comp averages cross-sectional z-scores of P̂ and Û. "
        "Daily return is higher than P-only's; Sharpe is slightly "
        "lower (1.97 vs 2.18) because Z-comp runs at higher vol."
    )

with col3:
    st.metric(
        "Product composite destroys returns",
        f"Sharpe {sharpe_prod_postcost:0.2f}" if sharpe_prod_postcost is not None else "n/a",
        delta=(f"{sharpe_prod_postcost - sharpe_ens1_postcost:+0.2f} vs P-only"
                if sharpe_prod_postcost is not None and sharpe_ens1_postcost else None),
    )
    st.caption("Disagreement on sign + multiplication = sign-flips.")

st.divider()
st.markdown(
    "→ Filter every (era, scheme, model, cost regime) we ran on the "
    "**[Results matrix](Results_matrix)** appendix page. "
    "Why the baseline P-only scheme collapsed post-2015 — and why the "
    "gated variants extract residual signal — is unpacked on "
    "**[Regimes](Regimes)** and **[Conclusion](Conclusion)**."
)
