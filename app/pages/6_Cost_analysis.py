"""Page 6 — Cost analysis.

Turnover × Sharpe scatter with a Pareto-frontier overlay, plus the 16-row
no-trade band × model matrix from ``notebooks/cost_band_test.ipynb`` on the
cost-modeling branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Cost analysis", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_summary_table, load_cost_bands,
)
from lib.charts import MODEL_ORDER, SCHEME_ORDER  # noqa: E402

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
bands = load_cost_bands()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains "
    "basis points, half-turn costs, and Sharpe — start there if cost "
    "terminology is unfamiliar."
)

st.title("Cost analysis")
st.caption(
    "Turnover is the first-order cost driver — under 5 bps/half-turn a "
    "portfolio that turns over 200 % of NAV per day pays 10 bps in daily "
    "frictions. The scatter below plots every (model, scheme, era, cost "
    "regime) we ran; the 16-row table at the bottom extends the analysis to "
    "a no-trade band that suppresses small daily weight changes."
)

# --- Turnover vs Sharpe scatter --------------------------------------------
st.subheader("Turnover vs post-cost Sharpe")

scatter_era = st.radio(
    "Era", options=sorted(summary["era"].unique().tolist()),
    horizontal=True,
)
scatter_cost = st.radio(
    "Cost regime", options=sorted(summary["cost_regime"].unique().tolist()),
    horizontal=True,
    index=sorted(summary["cost_regime"].unique().tolist()).index("5bps_half_turn"),
)

scatter_sub = summary.query(
    "era == @scatter_era and cost_regime == @scatter_cost"
).copy()

# --- Pareto frontier: best Sharpe at or below each turnover level ----------
front = scatter_sub.sort_values("avg_turnover").copy()
front["cummax_sharpe"] = front["sharpe"].cummax()
frontier_mask = front["sharpe"] == front["cummax_sharpe"]
frontier = front[frontier_mask].sort_values("avg_turnover")

scatter_sub["label"] = (
    scatter_sub["model"].astype(str) + " · " + scatter_sub["scheme"].astype(str)
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=scatter_sub["avg_turnover"],
    y=scatter_sub["sharpe"],
    mode="markers+text",
    text=scatter_sub["label"],
    textposition="top center",
    textfont=dict(size=10),
    marker=dict(
        size=9,
        color=scatter_sub["sharpe"],
        colorscale="Viridis",
        colorbar=dict(title="Sharpe"),
        line=dict(width=0),
    ),
    hovertemplate=(
        "%{text}<br>Turnover=%{x:.3f}<br>"
        "Sharpe=%{y:.2f}<extra></extra>"
    ),
    showlegend=False,
))
fig.add_trace(go.Scatter(
    x=frontier["avg_turnover"],
    y=frontier["sharpe"],
    mode="lines",
    line=dict(color="#c62828", dash="dash", width=2),
    name="Pareto frontier",
    hoverinfo="skip",
))
fig.update_layout(
    height=520,
    xaxis_title="Average daily turnover (fraction of NAV)",
    yaxis_title=f"Post-cost Sharpe ({scatter_cost})",
    margin=dict(l=50, r=20, t=30, b=50),
    legend=dict(orientation="h", y=-0.17),
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "The dashed red line is the Sharpe frontier — the best Sharpe achievable "
    "at or below each turnover level. Points above the frontier are strictly "
    "dominant on the turnover/Sharpe trade-off; points below pay more cost "
    "for less return."
)

# --- 16-row cost-band matrix ------------------------------------------------
st.divider()
st.subheader("No-trade band (10 bps) vs baseline")

if bands is None or bands.empty:
    st.warning("Cost-band results not yet in app/data/.")
    st.stop()

st.caption(
    "A *no-trade band* suppresses any daily weight change smaller than the "
    "threshold — the idea being that small day-to-day signal flips rarely "
    "carry enough alpha to pay the round-trip cost. The table below is the "
    "16 (model × scheme) cell result from the notebook "
    "`notebooks/cost_band_test.ipynb` on the cost-modeling branch. 'Baseline' "
    "columns are post-cost with no band; 'Band 10 bps' applies a 10-bps "
    "threshold. Positive Δ favours the band."
)

display = bands.rename(columns={
    "baseline_daily_return_pct": "Baseline daily ret %",
    "band10_daily_return_pct":   "Band 10bps daily ret %",
    "daily_return_delta_pct":    "Δ daily %",
    "baseline_ann_return_pct":   "Baseline ann ret %",
    "band10_ann_return_pct":     "Band 10bps ann ret %",
    "ann_return_delta_pct":      "Δ ann %",
    "baseline_sharpe":           "Baseline Sharpe",
    "band10_sharpe":             "Band 10bps Sharpe",
    "sharpe_delta":              "Δ Sharpe",
})

# Order for readability.
display["model"] = pd.Categorical(display["model"], categories=MODEL_ORDER,
                                   ordered=True)
display["scheme"] = pd.Categorical(display["scheme"], categories=SCHEME_ORDER,
                                    ordered=True)
display = display.sort_values(["model", "scheme"]).reset_index(drop=True)

st.dataframe(
    display[[
        "model", "scheme",
        "Baseline daily ret %", "Band 10bps daily ret %", "Δ daily %",
        "Baseline Sharpe", "Band 10bps Sharpe", "Δ Sharpe",
    ]],
    width="stretch", hide_index=True,
    column_config={
        "Baseline daily ret %": st.column_config.NumberColumn(format="%.4f"),
        "Band 10bps daily ret %": st.column_config.NumberColumn(format="%.4f"),
        "Δ daily %": st.column_config.NumberColumn(format="%+.4f"),
        "Baseline Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Band 10bps Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Δ Sharpe": st.column_config.NumberColumn(format="%+.2f"),
    },
)

# --- XGB interpretation -----------------------------------------------------
st.info(
    "**Why XGB is the only model that benefits from the band.** XGB's "
    "baseline turnover is the highest of the four families, and the shallow "
    "gradient-boosted trees are sensitive to small daily changes in feature "
    "values — a stock's rank on the XGB score flips more often than on RF or "
    "DNN even when the underlying prediction barely moves. That churn is "
    "costly but not alpha-producing. The 10-bps band filters out exactly "
    "those small flips, which is why every XGB scheme gets a positive Sharpe "
    "delta while ENS1 and DNN lose Sharpe under the same band."
)

# --- Sharpe-delta bar chart -------------------------------------------------
st.subheader("Sharpe delta by (model, scheme)")

bar = display.copy()
bar["key"] = bar["model"].astype(str) + " · " + bar["scheme"].astype(str)
bar_colors = bar["Δ Sharpe"].apply(
    lambda x: "#2e7d32" if x > 0 else "#c62828"
)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=bar["key"], y=bar["Δ Sharpe"],
    marker_color=bar_colors,
    text=[f"{v:+.2f}" for v in bar["Δ Sharpe"]],
    textposition="outside",
    showlegend=False,
))
fig_bar.add_hline(y=0, line_color="#555", line_width=1)
fig_bar.update_layout(
    height=400,
    yaxis_title="Band 10 bps Sharpe − baseline Sharpe",
    xaxis_title=None,
    xaxis_tickangle=-45,
    margin=dict(l=50, r=20, t=20, b=120),
)
st.plotly_chart(fig_bar, width="stretch")

st.caption(
    "Source: `notebooks/cost_band_test.ipynb` on the cost-modeling branch. "
    "All values post-cost at 5 bps per half-turn."
)
