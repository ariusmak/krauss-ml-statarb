"""Paper summary and reproduction evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Paper & Reproduction", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_equity_curves,
    load_summary_table,
    missing_build_warning,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
equity = load_equity_curves()

st.title("Paper & Reproduction")
st.caption(
    "What Krauss, Do & Huck (2017) did, how closely our re-run matches it, "
    "and which differences remain."
)

paper_tab, reproduction_tab, deviations_tab = st.tabs(
    ["Original Paper", "Reproduction Evidence", "Known Deviations"]
)

with paper_tab:
    st.markdown(
        """
**Paper.** Krauss, Do & Huck (2017), *Deep neural networks, gradient-boosted
trees, random forests: Statistical arbitrage on the S&P 500*, European
Journal of Operational Research.

**Study design.** The paper uses Thomson Reuters Datastream S&P 500
ever-members with month-end, no-lookahead membership rules. It evaluates
23 rolling study periods: each period trains on 750 trading days and trades
the next 250 trading days, then advances by 250 days. That gives **5,750
strictly out-of-sample trading days**, from **1992-12-17 to 2015-10-15**.

**Prediction task.** Each stock-day is represented by 31 lagged total-return
features: `R1...R20, R40, R60, ..., R240`. The models estimate `P_hat`, the
probability that a stock's next-day return beats the next-day
cross-sectional median. The original paper is therefore a **direction-only**
system; the later app extensions add `U_hat`, a predicted excess-return
magnitude.

**Trading rule.** Each trading day, rank eligible stocks by predicted
probability, buy the top-k names, and short the bottom-k names. Both legs are
equal-weighted and the portfolio is dollar-neutral. The headline setting is
`k = 10`, with post-cost results subtracting **5 bps per half-turn**.

**Headline result.** The paper's ENS1 ensemble earns **0.45% per day
pre-cost** and **0.25% per day post-cost**, with a published post-cost Sharpe
of **1.81**. It also documents alpha decay: the edge is strongest early in
the sample and fades substantially through time.
"""
    )
    st.markdown(
        """
| App label | Paper label | Exact model | Key setting |
|---|---|---|---|
| `DNN` | DNN | H2O maxout neural network | 31 -> 31 -> 10 -> 5 -> 2 |
| `XGB` | GBT | Gradient-boosted trees | 100 trees, depth 3, learning rate 0.1 |
| `RF` | RAF | Random forest | 1,000 trees, depth 20, sqrt(31) features/split |
| `ENS1` | ENS1 | Equal-weight ensemble | (P_DNN + P_GBT + P_RAF) / 3 |
"""
    )
    st.caption(
        "ENS2 and ENS3 are alternative weighted averages of the same three "
        "base learners. ENS1 is the main baseline used throughout the app."
    )

with reproduction_tab:
    st.markdown(
        """
The reproduction target is not cosmetic curve matching; it is whether the
same trading rule, data discipline, and cost convention produce the same
economic result. The table below compares our pre-cost daily returns to the
headline values in Krauss et al. Table 2.
"""
    )

    paper_values = {
        "DNN": 0.0033,
        "XGB": 0.0037,
        "RF": 0.0043,
        "ENS1": 0.0045,
    }
    display_name = {"DNN": "DNN", "XGB": "GBT", "RF": "RAF", "ENS1": "ENS1"}
    rows = []
    for model, paper in paper_values.items():
        sub = summary.query(
            "model == @model and scheme == 'P-only' "
            "and era == '1992-2015 (CRSP)' and cost_regime == 'no_cost'"
        )
        if sub.empty:
            continue
        ours = float(sub["full_sample_return"].iloc[0])
        rows.append(
            {
                "Model": display_name[model],
                "Paper daily return": paper,
                "Our daily return": ours,
                "Reproduction ratio": ours / paper,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Paper daily return": st.column_config.NumberColumn(format="%.4f"),
            "Our daily return": st.column_config.NumberColumn(format="%.4f"),
            "Reproduction ratio": st.column_config.NumberColumn(format="%.0%"),
        },
    )

    ens1_post = summary.query(
        "model == 'ENS1' and scheme == 'P-only' "
        "and era == '1992-2015 (CRSP)' and cost_regime == '5bps_half_turn'"
    )
    if not ens1_post.empty:
        row = ens1_post.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("ENS1 post-cost Sharpe", f"{row['sharpe']:.2f}")
        c2.metric("ENS1 post-cost daily return", f"{row['daily_return'] * 100:.2f}%")
        c3.metric("Average daily turnover", f"{row['avg_turnover']:.2f}")

    st.subheader("Reproduction equity curves")
    curve_tab, cost_tab = st.tabs(["Paper models, pre-cost", "ENS1 cost drag"])

    with curve_tab:
        curves = equity.query(
            "era == '1992-2015 (CRSP)' and scheme == 'P-only' "
            "and cost_regime == 'no_cost' and model in ['DNN', 'XGB', 'RF', 'ENS1']"
        ).copy()
        curves["wealth"] = 1.0 + curves["cum_ret"]
        fig = go.Figure()
        colors = {
            "DNN": "#4C78A8",
            "XGB": "#F58518",
            "RF": "#54A24B",
            "ENS1": "#2E7D32",
        }
        model_labels = {"DNN": "DNN", "XGB": "GBT", "RF": "RAF", "ENS1": "ENS1"}
        for model, grp in curves.groupby("model", observed=True):
            grp = grp.sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=grp["date"],
                    y=grp["wealth"],
                    mode="lines",
                    name=model_labels.get(model, model),
                    line=dict(color=colors.get(model, "#666"), width=2),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>wealth = %{y:.2f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            height=430,
            yaxis_title="Compounded $1 wealth, pre-cost",
            xaxis_title=None,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=55, r=20, t=25, b=55),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This is the paper's P-only reproduction universe before costs. "
            "The labels use the paper terminology: GBT for XGB and RAF for RF."
        )

    with cost_tab:
        curves = equity.query(
            "era == '1992-2015 (CRSP)' and model == 'ENS1' "
            "and scheme == 'P-only'"
        ).copy()
        curves["wealth"] = 1.0 + curves["cum_ret"]
        fig = go.Figure()
        labels = {
            "no_cost": "Gross / no transaction cost",
            "5bps_half_turn": "Post-cost: 5 bps per half-turn",
        }
        colors = {"no_cost": "#1F77B4", "5bps_half_turn": "#2E7D32"}
        for cost_regime, grp in curves.groupby("cost_regime", observed=True):
            grp = grp.sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=grp["date"],
                    y=grp["wealth"],
                    mode="lines",
                    name=labels.get(cost_regime, cost_regime),
                    line=dict(color=colors.get(cost_regime, "#666"), width=2),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>wealth = %{y:.2f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            height=430,
            yaxis_title="Compounded $1 wealth",
            xaxis_title=None,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=55, r=20, t=25, b=55),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The post-cost curve applies the paper's 5 bps per half-turn "
            "assumption to the reproduced ENS1 strategy."
        )

    st.info(
        "Main finding: the paper reproduces well enough to treat the extension "
        "results as a research result, not a broken implementation artifact."
    )

with deviations_tab:
    st.markdown(
        """
**Where our reproduction differs.**

- **Vendor and membership reconstruction.** Small differences in historical
  constituent reconstruction matter most during stress periods, especially
  around the 2008-09 window.
- **Turnover arithmetic.** Our realised turnover is lower than the turnover
  implied by the paper's post-cost arithmetic, so the same 5 bps / half-turn
  convention bites slightly less in our reproduced Sharpe.
- **Extension data.** The post-2015 extension moves onto the Datastream
  panel and then a yfinance-backed live tail. That is intentionally separated
  from the CRSP-keyed reproduction panel.

These differences are why the project reports reproduction parity and
extension results separately rather than stitching them into one undocumented
claim.
"""
    )

st.divider()
st.markdown(
    "Next: **System Architecture** shows how the app moves from raw data to "
    "validated `model_returns.parquet`; **Methodology & Extensions** explains "
    "the added P-hat / U-hat scoring rules."
)
