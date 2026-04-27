"""Page 7 — What didn't work.

Honest account of the VIX regime analysis, which also produced a null result:
regime-conditional k sizing delivers no Sharpe lift over the fixed-(10,10)
baseline.  The two earlier regime frameworks (MA/drawdown, leading-indicator
cross-validation) are mentioned but not shown here because their outputs are
not part of the current data pipeline.
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

st.set_page_config(page_title="What didn't work", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_regime_k_sensitivity, load_regime_leg_decomp, load_summary_table,
)

if not data_build_is_complete():
    missing_build_warning()

regime_k = load_regime_k_sensitivity()
regime_leg = load_regime_leg_decomp()
summary = load_summary_table()

st.title("What didn't work")
st.caption(
    "Three regime frameworks. This page presents the one that's fully "
    "materialised in the current app data; the other two produced null "
    "results but aren't shown here because their outputs aren't part of "
    "the repo's current data pipeline."
)

# --- Top-level note ---------------------------------------------------------
build_log_path = Path(__file__).resolve().parent.parent.parent / "docs" / "build_log.md"
if build_log_path.exists():
    build_log_link = (
        "[docs/build_log.md](https://github.com/ariusmak/krauss-ml-statarb/"
        "blob/main/docs/build_log.md)"
    )
else:
    build_log_link = "`docs/build_log.md` (on the main branch)"

st.markdown(
    f"""
### About this page

Three regime-conditioning experiments were run across the course of the
project:

1. **MA / drawdown regimes** — split the sample into up-trend / down-trend /
   drawdown buckets using a trailing moving-average rule on the S&P 500.
2. **Leading-indicator CV regimes** — cross-validate a regime classifier
   on macro leading indicators (yield-curve slope, credit spreads, employment
   momentum) before assigning per-regime k.
3. **VIX regimes** — shown below. Fixed thresholds <20 / 20-30 / >30 on a
   5-day smoothed VIX.

The first two are null results documented in {build_log_link}; their
outputs are not part of the current pipeline, so this page does not try to
reconstruct them. The VIX analysis is fully materialised — every number
below comes from `notebooks/regime_analysis.ipynb` and
`notebooks/regime_leg_decomp.ipynb` on the `vix-regime-analysis` branch.
"""
)

# --- VIX regime analysis ---------------------------------------------------
st.header("VIX regime analysis")

st.markdown(
    """
The question: does the optimal portfolio size k depend on the VIX
volatility regime? The paper hints at this in Table 4 by using VIX > 30
as a crisis indicator in a factor regression, so it's a natural extension
to ask whether we should size the long-short book differently in low /
mid / high vol days.

**Headline result: no.** The fixed k=10 allocation wins in every regime
at every scoring scheme we tried. The regime-conditional grid search
either matches or underperforms the fixed baseline. The Sharpe drop in
high-vol days is almost entirely attributable to a handful of days in
September-November 2008 — outside that window the short leg does what
it is supposed to do.
"""
)

# --- Regime × k grid -------------------------------------------------------
st.subheader("Z-comp ENS1 Sharpe by regime × k")

if regime_k is not None and not regime_k.empty:
    pivot = regime_k.pivot(index="regime", columns="k", values="sharpe")
    # Force row order low -> mid -> high with all at top.
    row_order = [r for r in ["all", "low_vol", "mid_vol", "high_vol"]
                  if r in pivot.index]
    pivot = pivot.loc[row_order]

    fig_k = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=1.0,
        text=[[f"{v:.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="Sharpe"),
    ))
    fig_k.update_layout(
        height=340,
        xaxis_title="k (each side)",
        yaxis_title="Regime",
        margin=dict(l=80, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_k, width="stretch")

    # Also show the raw-return grid alongside.
    with st.expander("Raw post-cost daily return (bps)"):
        bps_pivot = regime_k.pivot(index="regime", columns="k",
                                     values="daily_return_bps")
        bps_pivot = bps_pivot.loc[row_order]
        st.dataframe(bps_pivot.round(2), width="stretch")
else:
    st.info("regime_k_sensitivity.parquet not present in app/data/.")

st.markdown(
    """
Two things stand out.

1. **k=10 dominates at every regime row.** Every row is a monotonic slide
   from the first column to the last — the best Sharpe is always at the
   smallest (most concentrated) k. The regime structure does not flip that.
2. **High-vol Sharpe is lower than low/mid but not catastrophically so.**
   At k=10 the ratio is 1.31 / 2.42 ≈ 0.54, meaningful but not enough to
   justify sitting out those days given that they're only ~10% of the
   sample.
"""
)

# --- Leg decomposition / rescue attempts ------------------------------------
st.subheader("Rescue attempts on the high-vol regime")

if regime_leg is not None and not regime_leg.empty:
    st.dataframe(
        regime_leg.drop(columns=["source"]),
        width="stretch", hide_index=True,
        column_config={
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.3f"),
        },
    )
else:
    st.info("regime_leg_decomp.parquet not present in app/data/.")

st.markdown(
    """
**Reading.** Rule B (dropping Sep-Nov 2008) lifts Sharpe from 1.92 to
2.18 — a one-off GFC artefact that accounts for most of the high-vol
weakness. Rule C (sit out whenever VIX 5-day mean > 30) delivers a
modest +0.15 lift but only by giving up the ~9% of the sample that
happens to contain the biggest short-leg pay-days in other crises.
Rule D isolates where the high-vol drag actually sits: in the short
leg. And Rule E is the headline null — a full grid search over
regime-conditional k ends up essentially at the same Sharpe as the
fixed baseline.
"""
)

# --- Tie-in to the app's summary table for at-a-glance decay ---------------
st.subheader("Cross-check: post-2015 decay across schemes")

xcheck = summary.query(
    "model == 'ENS1' and era == '2015-2025 (extension)' "
    "and cost_regime == '5bps_half_turn'"
)[["scheme", "sharpe", "daily_return", "trading_days"]].copy()
xcheck = xcheck.rename(columns={
    "sharpe": "Sharpe (extension, post-cost)",
    "daily_return": "Daily ret (%)",
    "trading_days": "Trading days",
})
xcheck["Daily ret (%)"] = xcheck["Daily ret (%)"] * 100
st.dataframe(
    xcheck, width="stretch", hide_index=True,
    column_config={
        "Sharpe (extension, post-cost)": st.column_config.NumberColumn(format="%.2f"),
        "Daily ret (%)": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    "No scoring scheme, and no regime overlay we tried, restores positive "
    "post-cost Sharpe in the 2015-2025 extension. That's the empirical "
    "content behind the 'what didn't work' label on this page — not just "
    "one failed regime idea, but the full family."
)

# --- Summary ---------------------------------------------------------------
st.divider()
st.info(
    "**Overall verdict.** VIX regime conditioning is one of three null "
    "results on the regime question. The alpha decay after 2008-ish is "
    "not concentrated in a particular vol regime; it appears regime-"
    "universal. The other two regime frameworks (MA/drawdown, leading-"
    "indicator CV) produced the same conclusion through different routes "
    "— see the build log for details."
)
