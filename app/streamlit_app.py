"""Executive summary for the Krauss ML stat-arb research system."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="Krauss stat-arb — executive summary",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_model_returns,
    load_model_returns_metadata,
    load_summary_table,
    missing_build_warning,
    model_returns_health,
    runtime_mode,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
model_returns = load_model_returns()
metadata = load_model_returns_metadata() or {}
health_ok, health_errors = model_returns_health()
mode = runtime_mode()


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _summary_value(
    *,
    model: str,
    scheme: str,
    era: str,
    cost_regime: str,
    column: str,
) -> float | None:
    row = summary.query(
        "model == @model and scheme == @scheme and era == @era "
        "and cost_regime == @cost_regime"
    )
    return None if row.empty else float(row[column].iloc[0])


def _kpi(col, label: str, value: str, caption: str, color: str = "#142B4F"):
    with col:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                "<div style='font-size:30px;font-weight:750;color:"
                f"{color};line-height:1.1;margin-top:-4px'>{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(caption)


paper_ens1_daily = 0.0045
ours_ens1_precost = _summary_value(
    model="ENS1",
    scheme="P-only",
    era="1992-2015 (CRSP)",
    cost_regime="no_cost",
    column="full_sample_return",
)
reproduction_ratio = (
    ours_ens1_precost / paper_ens1_daily
    if ours_ens1_precost is not None
    else None
)

era1_zcomp_sharpe = _summary_value(
    model="ENS1",
    scheme="Z-comp",
    era="1992-2015 (CRSP)",
    cost_regime="5bps_half_turn",
    column="sharpe",
)
era2_baseline_sharpe = _summary_value(
    model="ENS1",
    scheme="P-only",
    era="2015-2025 (extension)",
    cost_regime="5bps_half_turn",
    column="sharpe",
)

signal_max = metadata.get("signal_date_max", "n/a")
realized_max = metadata.get("realized_return_date_max", "n/a")

st.title("Krauss ML stat-arb — research system")
st.caption(
    "A reproduction and extension of Krauss, Do & Huck (2017), delivered as "
    "a working trading-system dashboard: data pipeline, prediction panel, "
    "model-return artifact, transaction-cost backtests, risk diagnostics, "
    "and an interactive simulator."
)
if mode["can_refresh"]:
    st.caption("Runtime mode: repo-backed live refresh is available.")
else:
    st.caption(
        "Runtime mode: frozen standalone snapshot. "
        "Repo-backed refresh is unavailable."
    )

st.markdown(
    """
**Thesis.** The original paper reproduces cleanly, magnitude-aware scoring
adds useful diagnostics, and the unconditional edge weakens sharply in the
2015-live extension after costs. The gated variants are more selective, but
the most important result is not a bigger Sharpe ratio; it is a reproducible
system that shows where and why the strategy stops working.
"""
)

k1, k2, k3, k4 = st.columns(4)
_kpi(
    k1,
    "Reproduction parity",
    _fmt_pct(reproduction_ratio, 0),
    "ENS1 pre-cost daily return reproduced vs Krauss et al. Table 2.",
    "#2e7d32",
)
_kpi(
    k2,
    "Magnitude composite",
    f"Sharpe {_fmt_num(era1_zcomp_sharpe)}",
    "ENS1 Z-comp, 1992-2015, post-cost. Uses P-hat and U-hat additively.",
    "#1f77b4",
)
_kpi(
    k3,
    "Post-2015 baseline",
    f"Sharpe {_fmt_num(era2_baseline_sharpe)}",
    "ENS1 P-only after costs in the extension era: alpha decay.",
    "#c62828",
)
_kpi(
    k4,
    "Live artifact status",
    f"Through {realized_max}",
    f"Signals through {signal_max}; model_returns health: "
    f"{'pass' if health_ok else 'needs attention'}.",
    "#2e7d32" if health_ok else "#c62828",
)

if not health_ok:
    with st.expander("Model-return health issues", expanded=True):
        for err in health_errors:
            st.write(f"- {err}")

st.subheader("Strategy equity snapshot")
plot = model_returns.query(
    "model == 'ENS1' and scheme == 'P-only' and cost_regime == 'baseline_5bps'"
).copy()
plot = plot.sort_values("next_date")
plot["nav"] = (1.0 + plot["ret"].fillna(0.0)).cumprod()

fig = go.Figure()
previous_tail = None
for era, grp in plot.groupby("era", observed=True):
    trace = grp
    if previous_tail is not None:
        trace = pd.concat([previous_tail, grp], ignore_index=True)
    fig.add_trace(
        go.Scatter(
            x=trace["next_date"],
            y=trace["nav"],
            mode="lines",
            name=f"ENS1 P-only · {era}",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>wealth = %{y:.2f}<extra></extra>"
            ),
        )
    )
    previous_tail = grp.tail(1)
fig.add_shape(
    type="line",
    x0="2015-10-16",
    x1="2015-10-16",
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="#666", width=1, dash="dash"),
)
fig.add_annotation(
    x="2015-10-16",
    y=1.0,
    yref="paper",
    yanchor="bottom",
    text="extension era",
    showarrow=False,
)
fig.update_layout(
    height=420,
    yaxis_title="Compounded wealth, post-cost",
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.18),
    margin=dict(l=55, r=20, t=35, b=55),
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader("How to read the app")
    st.markdown(
        """
The app is organized as a reproducible research system rather than a slide
deck dumped into Streamlit.

1. **Paper & Reproduction** summarizes Krauss, Do & Huck (2017), then checks
   whether our P-only reproduction matches the paper's headline daily returns
   and equity curves.
2. **System Architecture** shows the data contract: Datastream S&P 500
   membership, lagged-return features, model outputs, portfolio construction,
   transaction costs, and the validated app-facing parquet artifacts.
3. **Methodology & Extensions** explains the new dual-output target:
   `P_hat` for direction and `U_hat` for magnitude, plus the six scoring
   schemes that turn those heads into long/short ranks.
4. **Results & Risk Diagnostics** is the main evidence page. It compares ENS1
   across every scoring scheme, shows transaction-cost drag, tests the
   no-trade band, and stress-tests drawdowns and volatility regimes.
5. **Simulator** is the interactive deliverable. Use **Historical Backtest**
   to choose a date range, model, scoring rule, cost regime, and capital, then
   inspect NAV, drawdown, and risk metrics. Use **Live Simulation** to replay
   the latest realized live window and see the newest recommended long/short
   positions for the next trading session.
6. **Conclusion** separates what worked from what failed: reproduction holds,
   gated scoring is selectively useful, the no-trade band can suppress churn,
   and the post-2015 edge mostly decays after costs.
"""
    )

    st.markdown(
        """
**Important simulator feature.** The **No-trade band 10 bps** cost regime is
available only for scoring rules that use `U_hat`. Instead of replacing the
portfolio every day, it keeps incumbent positions unless a candidate improves
predicted excess return by at least one round-trip cost. That is why the
simulator is useful for grading: it lets you test both the research signal and
the execution rule from one page.
"""
    )

with right:
    st.subheader("Where to inspect")
    checks = [
        ("Paper summary", "Paper & Reproduction",
         "Original question, data, features, models, and cost convention."),
        ("Reproduction proof", "Paper & Reproduction",
         "Parity table, ENS1 metrics, and reproduction equity curves."),
        ("Pipeline integrity", "System Architecture",
         "No-lookahead universe, feature build, model panel, and app artifacts."),
        ("Extension logic", "Methodology & Extensions",
         "P_hat/U_hat target, scoring rules, P-gates, and no-trade band rule."),
        ("Main results", "Results & Risk Diagnostics",
         "ENS1 by scoring scheme, transaction costs, regimes, and risk."),
        ("Interactive backtests", "Simulator",
         "Historical replay with NAV, drawdown, SPY context, and risk metrics."),
        ("Live workflow", "Simulator",
         "Latest realized window, source health, and next-session positions."),
        ("Research takeaways", "Conclusion",
         "What reproduced, what extended, what failed, and next work."),
    ]
    st.dataframe(
        pd.DataFrame(checks, columns=["Requirement", "Page", "What to look for"]),
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "Every figure is historical. The app automatically starts a background "
    "incremental refresh when opened; results are displayed from the validated "
    "`app/data/model_returns.parquet` artifact."
)
