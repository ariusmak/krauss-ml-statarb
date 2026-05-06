"""Consolidated results, costs, regimes and risk diagnostics."""
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

st.set_page_config(page_title="Results & Risk Diagnostics",
                   page_icon=":bar_chart:", layout="wide")

from lib.charts import MODEL_ORDER, SCHEME_ORDER  # noqa: E402
from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_cost_bands,
    load_equity_curves,
    load_regime_k_sensitivity,
    load_regime_leg_decomp,
    load_summary_table,
    missing_build_warning,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
equity = load_equity_curves()
cost_bands = load_cost_bands()
regime_k = load_regime_k_sensitivity()
regime_leg = load_regime_leg_decomp()

COST_LABELS = {
    "5bps_half_turn": "Post-cost: 5 bps per half-turn",
    "no_cost": "Gross / no transaction cost",
}

st.title("Results & Risk Diagnostics")
st.caption(
    "One evidence page for the grading rubric: headline performance, "
    "transaction costs, regime sensitivity, drawdowns, and the full matrix."
)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:+.{digits}f}%"


def _cost_label(value: str) -> str:
    return COST_LABELS.get(str(value), str(value))


def _metric(col, label: str, value: str, caption: str, color: str = "#142B4F"):
    with col:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                "<div style='font-size:26px;font-weight:750;color:"
                f"{color};line-height:1.1;margin-top:-4px'>{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(caption)


def _row(model: str, scheme: str, era: str, cost: str) -> pd.Series | None:
    sub = summary.query(
        "model == @model and scheme == @scheme and era == @era "
        "and cost_regime == @cost"
    )
    return None if sub.empty else sub.iloc[0]


headline_tab, cost_tab, regime_tab, risk_tab, matrix_tab = st.tabs(
    [
        "Headline Results",
        "Transaction Costs",
        "Regime Sensitivity",
        "Risk Diagnostics",
        "Full Matrix",
    ]
)

with headline_tab:
    era1 = "1992-2015 (CRSP)"
    era2 = "2015-2025 (extension)"
    ens1_base = _row("ENS1", "P-only", era1, "5bps_half_turn")
    ens1_z = _row("ENS1", "Z-comp", era1, "5bps_half_turn")
    ens1_prod = _row("ENS1", "Product", era1, "5bps_half_turn")
    era2_base = _row("ENS1", "P-only", era2, "5bps_half_turn")

    c1, c2, c3, c4 = st.columns(4)
    _metric(
        c1,
        "Reproduced baseline",
        f"Sharpe {ens1_base['sharpe']:.2f}" if ens1_base is not None else "n/a",
        "ENS1 P-only, CRSP era, post-cost.",
        "#2e7d32",
    )
    _metric(
        c2,
        "Magnitude composite",
        f"Sharpe {ens1_z['sharpe']:.2f}" if ens1_z is not None else "n/a",
        "Z-comp uses P-hat and U-hat; higher return, lower Sharpe than P-only.",
        "#1f77b4",
    )
    _metric(
        c3,
        "Product warning",
        f"Sharpe {ens1_prod['sharpe']:.2f}" if ens1_prod is not None else "n/a",
        "Multiplying disagreeing heads destroys the tails.",
        "#c62828",
    )
    _metric(
        c4,
        "Post-2015 baseline",
        (
            f"Sharpe {era2_base['sharpe']:.2f}"
            if era2_base is not None
            else "n/a"
        ),
        "ENS1 P-only after costs in the extension era.",
        "#c62828",
    )

    st.subheader("ENS1 P-only equity curve")
    ens1 = equity.query("model == 'ENS1' and scheme == 'P-only'").copy()
    ens1 = ens1.sort_values(["cost_regime", "date"])
    ens1["ret"] = ens1["ret"].fillna(0.0)
    ens1["wealth"] = ens1.groupby("cost_regime", observed=True)["ret"].transform(
        lambda r: (1.0 + r).cumprod()
    )
    fig = go.Figure()
    labels = {
        "no_cost": "pre-cost",
        "5bps_half_turn": "post-cost, 5 bps / half-turn",
    }
    colors = {"no_cost": "#1f77b4", "5bps_half_turn": "#2e7d32"}
    for cr, grp in ens1.groupby("cost_regime", observed=True):
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["wealth"],
                mode="lines",
                name=labels.get(cr, cr),
                line=dict(color=colors.get(cr, "#666"), width=2),
            )
        )
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
        text="extension",
        showarrow=False,
    )
    fig.update_layout(
        height=460,
        yaxis_title="Compounded wealth",
        xaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=55, r=20, t=30, b=55),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ENS1 by scoring scheme")
    ens1_scores = summary.query(
        "model == 'ENS1' and cost_regime == '5bps_half_turn'"
    ).copy()
    ens1_scores = ens1_scores[ens1_scores["scheme"].isin(SCHEME_ORDER)].copy()
    ens1_scores["scheme"] = pd.Categorical(
        ens1_scores["scheme"], categories=SCHEME_ORDER, ordered=True
    )
    ens1_scores = ens1_scores.sort_values(["era", "scheme"])

    fig_scores = go.Figure()
    for era_name, grp in ens1_scores.groupby("era", observed=True):
        fig_scores.add_trace(
            go.Bar(
                x=grp["scheme"].astype(str),
                y=grp["sharpe"],
                name=era_name,
                hovertemplate=(
                    "%{x}<br>Sharpe = %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig_scores.add_hline(y=0, line=dict(color="#777", width=1))
    fig_scores.update_layout(
        height=390,
        barmode="group",
        yaxis_title="Post-cost Sharpe",
        xaxis_title=None,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=55, r=20, t=25, b=75),
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    display = ens1_scores.copy()
    display["scheme"] = display["scheme"].astype(str)
    display["daily_return_pct"] = display["daily_return"] * 100
    display["ann_return_pct"] = display["ann_return"] * 100
    display["max_drawdown_pct"] = display["max_drawdown"] * 100
    display["hit_rate_pct"] = display["hit_rate"] * 100
    st.dataframe(
        display[
            [
                "era",
                "scheme",
                "daily_return_pct",
                "ann_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "hit_rate_pct",
                "avg_turnover",
                "trading_days",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "era": st.column_config.TextColumn("Era"),
            "scheme": st.column_config.TextColumn("Scoring scheme"),
            "daily_return_pct": st.column_config.NumberColumn(
                "Daily return", format="%.3f%%"
            ),
            "ann_return_pct": st.column_config.NumberColumn(
                "Annualized return", format="%.1f%%"
            ),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_drawdown_pct": st.column_config.NumberColumn(
                "Max drawdown", format="%.1f%%"
            ),
            "hit_rate_pct": st.column_config.NumberColumn(
                "Hit rate", format="%.1f%%"
            ),
            "avg_turnover": st.column_config.NumberColumn(
                "Avg turnover", format="%.3f"
            ),
            "trading_days": st.column_config.NumberColumn("Strategy days"),
        },
    )

    st.info(
        "Headline interpretation: the original P-only direction model "
        "reproduces. U-only and Z-comp test whether magnitude helps, Product "
        "shows that multiplying disagreeing heads is fragile, and P-gates are "
        "selective: they can stay positive in the extension table, but only by "
        "trading fewer days. The broader post-2015 lesson is alpha decay after "
        "costs, especially for the unconditional rankings."
    )

with cost_tab:
    st.subheader("Turnover versus Sharpe")
    st.markdown(
        """
Gross returns show whether the ranking signal exists before execution. The
post-cost view subtracts 5 bps per half-turn, so high-turnover strategies can
lose Sharpe even when their gross daily returns look strong.
        """
    )
    cost_options = sorted(summary["cost_regime"].unique().tolist())
    era = st.radio(
        "Era",
        sorted(summary["era"].unique().tolist()),
        horizontal=True,
        key="cost_era",
    )
    cost = st.radio(
        "Cost regime",
        cost_options,
        horizontal=True,
        index=cost_options.index("5bps_half_turn"),
        format_func=_cost_label,
        key="cost_regime",
    )
    sub = summary.query("era == @era and cost_regime == @cost").copy()
    sub["label"] = sub["model"].astype(str) + " · " + sub["scheme"].astype(str)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["avg_turnover"],
            y=sub["sharpe"],
            text=sub["label"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=10,
                color=sub["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
            ),
            hovertemplate=(
                "%{text}<br>turnover=%{x:.3f}<br>Sharpe=%{y:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=470,
        xaxis_title="Average daily turnover, fraction of NAV",
        yaxis_title="Sharpe",
        margin=dict(l=55, r=20, t=30, b=55),
    )
    st.plotly_chart(fig, use_container_width=True)

    if cost_bands is None or cost_bands.empty:
        st.warning("No no-trade-band table is available in app/data/.")
    else:
        st.subheader("No-trade band matrix")
        st.markdown(
            """
The 10 bps no-trade band is a cost-aware replacement rule for U-hat schemes.
It keeps an incumbent position unless the proposed replacement improves
predicted excess return by at least one round-trip cost. In this table the
rule helps XGB consistently, is mixed for RF, and does not automatically
improve ENS1. That is the useful grading point: transaction-cost controls need
validation, not just a plausible story.
            """
        )
        display = cost_bands.copy()
        display["model"] = pd.Categorical(
            display["model"], categories=MODEL_ORDER, ordered=True
        )
        display["scheme"] = pd.Categorical(
            display["scheme"], categories=SCHEME_ORDER, ordered=True
        )
        display = display.sort_values(["model", "scheme"])
        st.dataframe(
            display[
                [
                    "model",
                    "scheme",
                    "baseline_sharpe",
                    "band10_sharpe",
                    "sharpe_delta",
                    "daily_return_delta_pct",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "baseline_sharpe": st.column_config.NumberColumn(
                    "Baseline Sharpe", format="%.2f"
                ),
                "band10_sharpe": st.column_config.NumberColumn(
                    "Band Sharpe", format="%.2f"
                ),
                "sharpe_delta": st.column_config.NumberColumn(
                    "Delta Sharpe", format="%+.2f"
                ),
                "daily_return_delta_pct": st.column_config.NumberColumn(
                    "Delta daily return, percentage points", format="%+.4f"
                ),
            },
        )

with regime_tab:
    st.markdown(
        """
The project tested three regime ideas. The materialised VIX grid below is
representative: fixed k=10 remains the best allocation in every regime, and
regime-specific sizing does not rescue post-2015 alpha decay.
"""
    )
    if regime_k is None or regime_k.empty:
        st.warning("regime_k_sensitivity.parquet is not available.")
    else:
        pivot = regime_k.pivot(index="regime", columns="k", values="sharpe")
        order = [r for r in ["all", "low_vol", "mid_vol", "high_vol"]
                 if r in pivot.index]
        pivot = pivot.loc[order]
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                zmid=1.0,
                text=[[f"{v:.2f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                colorbar=dict(title="Sharpe"),
            )
        )
        fig.update_layout(
            height=350,
            xaxis_title="k per side",
            yaxis_title="VIX regime",
            margin=dict(l=80, r=20, t=30, b=45),
        )
        st.plotly_chart(fig, use_container_width=True)

    if regime_leg is not None and not regime_leg.empty:
        st.subheader("High-vol rescue-rule decomposition")
        st.dataframe(regime_leg.drop(columns=["source"], errors="ignore"),
                     use_container_width=True, hide_index=True)

with risk_tab:
    st.subheader("Risk profile by era")
    risk = summary.query("cost_regime == '5bps_half_turn'").copy()
    risk["key"] = risk["model"].astype(str) + " · " + risk["scheme"].astype(str)
    metric_choice = st.selectbox(
        "Risk metric",
        ["sharpe", "max_drawdown", "ann_vol", "hit_rate", "worst_day"],
        format_func={
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
            "ann_vol": "Annualized volatility",
            "hit_rate": "Hit rate",
            "worst_day": "Worst day",
        }.get,
    )
    fig = go.Figure()
    for era_name, grp in risk.groupby("era", observed=True):
        top = grp.sort_values("sharpe", ascending=False).head(12)
        fig.add_trace(
            go.Bar(
                x=top["key"],
                y=top[metric_choice],
                name=era_name,
            )
        )
    fig.update_layout(
        height=470,
        barmode="group",
        xaxis_tickangle=-40,
        yaxis_title=metric_choice,
        margin=dict(l=55, r=20, t=30, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The risk view makes the same point from another angle: after costs, "
        "the extension era has weaker unconditional rankings and the gated "
        "variants trade on a much narrower set of days."
    )

with matrix_tab:
    st.subheader("Filterable full matrix")
    c1, c2, c3, c4 = st.columns(4)
    eras = c1.multiselect(
        "Era",
        sorted(summary["era"].unique().tolist()),
        default=sorted(summary["era"].unique().tolist()),
    )
    models = c2.multiselect(
        "Model",
        [m for m in MODEL_ORDER if m in summary["model"].unique()],
        default=["ENS1"],
    )
    schemes = c3.multiselect(
        "Scheme",
        [s for s in SCHEME_ORDER if s in summary["scheme"].unique()],
        default=["P-only", "Z-comp"],
    )
    costs = c4.multiselect(
        "Cost",
        sorted(summary["cost_regime"].unique().tolist()),
        default=["5bps_half_turn"],
        format_func=_cost_label,
    )
    filt = summary[
        summary["era"].isin(eras)
        & summary["model"].isin(models)
        & summary["scheme"].isin(schemes)
        & summary["cost_regime"].isin(costs)
    ].copy()
    filt["cost_regime_label"] = filt["cost_regime"].map(_cost_label)
    st.dataframe(
        filt[
            [
                "era",
                "model",
                "scheme",
                "cost_regime_label",
                "daily_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "hit_rate",
                "avg_turnover",
            ]
        ].sort_values(["era", "scheme", "model"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "daily_return": st.column_config.NumberColumn(format="%.4f"),
            "cost_regime_label": st.column_config.TextColumn("Cost regime"),
            "ann_return": st.column_config.NumberColumn(format="%.3f"),
            "ann_vol": st.column_config.NumberColumn(format="%.3f"),
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn(format="%.3f"),
            "hit_rate": st.column_config.NumberColumn(format="%.3f"),
            "avg_turnover": st.column_config.NumberColumn(format="%.3f"),
        },
    )
