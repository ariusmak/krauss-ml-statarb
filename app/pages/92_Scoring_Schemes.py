"""Page 4 — Scoring schemes explained.

Six configured scoring schemes, plus the interactive directional-disagreement
scatter widget and a rolling disagreement-rate chart that overlays the CRSP
and 2015-live extension eras.
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

st.set_page_config(page_title="Scoring schemes", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_disagreement_panel,
    load_summary_table,
    missing_build_warning,
)

if not data_build_is_complete():
    missing_build_warning()

dis = load_disagreement_panel()
summary = load_summary_table()

st.caption(
    ":books: **Appendix** — scoring-rule detail supporting Methodology & "
    "Extensions and Results & Risk Diagnostics."
)

st.title("Scoring schemes explained")
st.caption(
    "Six configured ways to combine direction (P̂) and magnitude (Û) into "
    "a single per-stock score. P-gate appears as a family because the gate "
    "threshold behaves non-linearly — 0.03 gives a meaningful subset while "
    "0.05 is much more selective."
)

# --- Prose description + quick-stat cards -----------------------------------
st.subheader("The scheme families at a glance")

cards = [
    ("P-only",
     "Rank each stock by P̂ (paper baseline). No magnitude information. "
     "Robust pre-cost but leaves the U head unused."),
    ("U-only",
     "Rank by signed Û. Uses magnitude only, ignoring direction confidence. "
     "Slightly underperforms P-only in the CRSP era."),
    ("Z-comp",
     "0.5·z(P̂) + 0.5·z(Û), z-scored per-day. Raises ENS1 daily return in "
     "the CRSP era, but does not beat P-only on Sharpe."),
    ("Product",
     "(2P̂ − 1)·Û. Multiplies a centred direction signal by magnitude. "
     "Destroys returns because the two heads disagree on ~half of stocks."),
    ("P-gate(c)",
     "Longs keep P̂ > 0.5+c then rank by Û desc; shorts keep P̂ < 0.5−c "
     "then rank by Û asc. Higher c → fewer active days but cleaner "
     "conviction."),
]

cols = st.columns(5)
for (name, desc), col in zip(cards, cols):
    with col:
        row = summary.query(
            "model == 'ENS1' and scheme == @name and era == '1992-2015 (CRSP)'"
            " and cost_regime == '5bps_half_turn'"
        ) if name != "P-gate(c)" else summary.query(
            "model == 'ENS1' and scheme == 'P-gate(0.03)' "
            "and era == '1992-2015 (CRSP)' and cost_regime == '5bps_half_turn'"
        )
        sharpe = float(row["sharpe"].iloc[0]) if not row.empty else float("nan")
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(desc)
            st.metric("Sharpe (post-cost, ENS1)",
                      f"{sharpe:0.2f}" if not np.isnan(sharpe) else "n/a")

# --- Interactive scatter ----------------------------------------------------
st.divider()
st.subheader("Directional-disagreement scatter")

if dis is None or dis.empty:
    st.warning("Disagreement panel not available in app/data/.")
    st.stop()

min_date = dis["date"].min()
max_date = dis["date"].max()

selected_date = st.slider(
    "Pick a date",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=pd.Timestamp("2008-10-13").date() if pd.Timestamp("2008-10-13") >= min_date
            and pd.Timestamp("2008-10-13") <= max_date else min_date.date(),
    format="YYYY-MM-DD",
)
selected_ts = pd.Timestamp(selected_date)

day = dis[dis["date"] == selected_ts].copy()
if day.empty:
    # Snap to nearest available date.
    nearest_idx = (dis["date"] - selected_ts).abs().idxmin()
    selected_ts = dis.loc[nearest_idx, "date"]
    day = dis[dis["date"] == selected_ts].copy()
    st.info(f"No data for the selected date; showing nearest: {selected_ts.date()}")

day["x"] = day["p_ens1"] - 0.5
day["y"] = day["u_ens1"]
day["quadrant"] = np.where(
    (day["x"] >= 0) & (day["y"] >= 0), "Q1 (agree +)",
    np.where(
        (day["x"] < 0) & (day["y"] < 0), "Q3 (agree −)",
        np.where((day["x"] >= 0) & (day["y"] < 0), "Q2 (disagree)",
                  "Q4 (disagree)")
    ),
)
agree_color = "#2e7d32"
disagree_color = "#c62828"
day["color"] = np.where(day["quadrant"].str.startswith("Q2")
                         | day["quadrant"].str.startswith("Q4"),
                         disagree_color, agree_color)

q1 = int(((day["x"] >= 0) & (day["y"] >= 0)).sum())
q2 = int(((day["x"] >= 0) & (day["y"] < 0)).sum())
q3 = int(((day["x"] < 0) & (day["y"] < 0)).sum())
q4 = int(((day["x"] < 0) & (day["y"] >= 0)).sum())
n_total = q1 + q2 + q3 + q4
n_agree = q1 + q3
n_disagree = q2 + q4

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=day["x"], y=day["y"],
    mode="markers",
    marker=dict(size=7, color=day["color"], opacity=0.7,
                  line=dict(width=0)),
    text=day["stock_id"],
    hovertemplate="stock %{text}<br>P̂−0.5=%{x:.3f}"
                    "<br>Û=%{y:.4f}<extra></extra>",
    showlegend=False,
))
fig.add_hline(y=0, line_dash="dash", line_color="#555")
fig.add_vline(x=0, line_dash="dash", line_color="#555")
fig.update_layout(
    height=520,
    xaxis_title="P̂ − 0.5  (direction confidence, ENS1)",
    yaxis_title="Û  (predicted excess return, ENS1)",
    margin=dict(l=50, r=20, t=30, b=50),
)
st.plotly_chart(fig, use_container_width=True)

# 2x2 quadrant count table
qtable = pd.DataFrame({
    "Û ≥ 0": [f"Q1 · agree · {q1}", f"Q4 · disagree · {q4}"],
    "Û < 0": [f"Q2 · disagree · {q2}", f"Q3 · agree · {q3}"],
}, index=["P̂ ≥ 0.5", "P̂ < 0.5"])
col_a, col_b = st.columns([3, 2])
with col_a:
    st.markdown(f"**{selected_ts.date()}** — {n_total} eligible stocks")
    st.dataframe(qtable, use_container_width=True)
with col_b:
    st.metric(
        "Disagreement rate",
        f"{(n_disagree / n_total * 100) if n_total else 0:0.1f}%",
    )
    st.caption(
        f"{n_disagree} of {n_total} stocks sit in quadrants Q2 or Q4. "
        "A rate near 50 % means the two heads are effectively independent."
    )

# --- Rolling disagreement rate ---------------------------------------------
st.divider()
st.subheader("Rolling 252-day disagreement rate, CRSP vs extension")

daily_rate = (
    dis
    .assign(agree=lambda d: (np.sign(d["p_ens1"] - 0.5)
                               == np.sign(d["u_ens1"])))
    .groupby(["era", "date"], observed=True)["agree"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "agree_n", "count": "total"})
    .reset_index()
)
daily_rate["disagree_rate"] = 1 - daily_rate["agree_n"] / daily_rate["total"]
daily_rate = daily_rate.sort_values(["era", "date"])
daily_rate["rolling_252"] = (
    daily_rate.groupby("era", observed=True)["disagree_rate"]
    .transform(lambda s: s.rolling(252, min_periods=60).mean())
)

era_colors = {
    "1992-2015 (CRSP)":    "#1f77b4",
    "2015-2025 (extension)": "#d62728",
}
fig2 = go.Figure()
for era, grp in daily_rate.groupby("era", observed=True):
    fig2.add_trace(go.Scatter(
        x=grp["date"], y=grp["rolling_252"],
        name=era, mode="lines",
        line=dict(color=era_colors.get(era, "#555"), width=2),
    ))
fig2.add_hline(y=0.5, line_dash="dot", line_color="#999",
                annotation_text="50% = heads independent",
                annotation_position="bottom right")
fig2.add_shape(
    type="line", x0="2015-10-16", x1="2015-10-16",
    y0=0, y1=1, yref="paper",
    line=dict(color="#666", width=1, dash="dash"),
)
fig2.add_annotation(
    x="2015-10-16", y=1.0, yref="paper", yanchor="bottom",
    text="extension era begins", showarrow=False,
    font=dict(size=11, color="#444"),
)
fig2.update_layout(
    height=380,
    yaxis=dict(title="Fraction of stocks where sign(P̂−0.5) ≠ sign(Û)",
                range=[0.35, 0.65], tickformat=".0%"),
    xaxis_title=None,
    legend=dict(orientation="h", y=-0.18),
    margin=dict(l=50, r=20, t=20, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

# Quick numerical comparison of the two eras
crsp_avg = daily_rate.query("era == '1992-2015 (CRSP)'")["disagree_rate"].mean()
ext_avg = daily_rate.query("era == '2015-2025 (extension)'")["disagree_rate"].mean()
col_a, col_b, col_c = st.columns(3)
col_a.metric("Avg disagreement — CRSP era",
              f"{crsp_avg * 100:.1f}%")
col_b.metric("Avg disagreement — extension era",
              f"{ext_avg * 100:.1f}%",
              delta=f"{(ext_avg - crsp_avg) * 100:+.1f} pp")
col_c.metric("Distance from 50% in extension",
              f"{abs(ext_avg - 0.5) * 100:.1f} pp")

st.warning(
    "**Interpretation — the post-2015 drop in disagreement is diagnostic of Û "
    "collapse, not improved prediction.** A genuinely aligned informative U "
    "head would lower disagreement *and* lift returns. In the extension era "
    "we see the first without a broad unconditional recovery: P-only, U-only, "
    "Z-comp, and Product are weak or negative, while the P-gates stay positive "
    "only by trading selectively. The mechanical explanation is that Û has shrunk "
    "towards zero in magnitude, so its sign is pinned almost entirely by "
    "numerical noise around the cross-sectional median — which happens to "
    "co-move with the sign of P̂ − 0.5. The histogram below makes this "
    "visible."
)

# --- Magnitude histogram: Û in era 1 vs era 2 ------------------------------
st.subheader("Û magnitude distribution — era 1 vs era 2")

# Subsample for a clean histogram — 250k points per era keeps the density
# shape exact while bounding memory at render time.
def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)

crsp_u = _sample(dis.query("era == '1992-2015 (CRSP)'"), 250_000)["u_ens1"]
ext_u = _sample(dis.query("era == '2015-2025 (extension)'"), 250_000)["u_ens1"]

crsp_std = float(crsp_u.std())
ext_std = float(ext_u.std())
crsp_iqr = float(crsp_u.quantile(0.75) - crsp_u.quantile(0.25))
ext_iqr = float(ext_u.quantile(0.75) - ext_u.quantile(0.25))

# Common bins for an honest side-by-side comparison.
bin_limit = float(max(crsp_u.abs().quantile(0.995), ext_u.abs().quantile(0.995)))
bins = np.linspace(-bin_limit, bin_limit, 120)

fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=crsp_u, name="1992-2015 (CRSP)",
    xbins=dict(start=-bin_limit, end=bin_limit,
                size=(2 * bin_limit) / 120),
    histnorm="probability density",
    marker_color="#1f77b4", opacity=0.55,
))
fig3.add_trace(go.Histogram(
    x=ext_u, name="2015-2025 (extension)",
    xbins=dict(start=-bin_limit, end=bin_limit,
                size=(2 * bin_limit) / 120),
    histnorm="probability density",
    marker_color="#d62728", opacity=0.55,
))
fig3.update_layout(
    height=380, barmode="overlay",
    xaxis_title="Û (ENS1 predicted excess return)",
    yaxis_title="Density",
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=50, r=20, t=20, b=40),
)
st.plotly_chart(fig3, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Û std — CRSP era", f"{crsp_std:.5f}")
col_b.metric("Û std — extension",
              f"{ext_std:.5f}",
              delta=f"{(ext_std - crsp_std):+.5f}")
col_c.metric("Compression ratio (CRSP / ext)",
              f"{crsp_std / ext_std:0.2f}×" if ext_std else "n/a")

st.caption(
    f"Û in the extension era has roughly {crsp_std / max(ext_std, 1e-12):0.1f}× "
    "less standard deviation than in the CRSP era. With magnitudes that small, "
    "the sign of Û is effectively pinned to the sign of P̂ − 0.5 by rounding "
    "noise, which is why the disagreement rate collapses without any return "
    "benefit — the heads aren't agreeing on a signal, there just isn't a "
    "signal in Û any more."
)

# --- Why the product composite fails ---------------------------------------
st.divider()
st.subheader("Why the product composite destroys returns")

st.markdown(
    """
The product composite ranks by `(2P̂ − 1) · Û`. When both signals
agree — Q1 or Q3 in the scatter above — the product is positive and
pointing the right way. When they disagree — Q2 or Q4 — the product
flips sign. With disagreement rates sitting in the mid 40s across the
CRSP sample, the product composite spends a large fraction of its score
distribution with a sign that points against one of the two signals,
and that is enough to flip the post-cost Sharpe from +2 to slightly
negative.

Contrast this with the Z-score composite, which adds the two signals
rather than multiplying them. Disagreement just attenuates the combined
score toward zero rather than flipping its sign, so Q2/Q4 stocks drop
out of the tails instead of contaminating them.
    """
)
