"""Page 5 — Results matrix.

Filterable summary table with linked equity curves for the highlighted rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Results matrix", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_equity_curves, load_spy_benchmark, load_summary_table,
)
from lib.charts import (  # noqa: E402
    EQUITY_MODE_OPTIONS, add_spy_overlay, equity_curve_figure,
    equity_mode_spec, MODEL_ORDER, SCHEME_ORDER,
)

if not data_build_is_complete():
    missing_build_warning()

equity = load_equity_curves()
summary = load_summary_table()
spy = load_spy_benchmark()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains "
    "what Sharpe, k = 10, P̂/Û and post-cost mean — read it first if any "
    "of the column names in the table below are unfamiliar."
)

st.title("Results matrix")
st.caption(
    "Every (era, scheme, model, cost regime) combination from the repo. "
    "Filter on the left, then pick rows to overlay their equity curves below. "
    "**Matched-days daily ret** is the mean return on the days P-gate(0.05) "
    "actively trades (2,703 in the CRSP era, 792 in the extension) — this is "
    "the like-for-like comparator for gated schemes."
)

# --- Filter sidebar ---------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    eras = sorted(summary["era"].unique().tolist())
    era_sel = st.multiselect("Era", options=eras, default=eras)

    model_opts = [m for m in MODEL_ORDER if m in summary["model"].unique()]
    model_sel = st.multiselect("Model", options=model_opts, default=model_opts)

    scheme_opts = [s for s in SCHEME_ORDER if s in summary["scheme"].unique()]
    scheme_sel = st.multiselect("Scheme", options=scheme_opts,
                                default=scheme_opts)

    cost_opts = sorted(summary["cost_regime"].unique().tolist())
    cost_sel = st.multiselect("Cost regime", options=cost_opts, default=cost_opts)

# Apply filters.
mask = (
    summary["era"].isin(era_sel)
    & summary["model"].isin(model_sel)
    & summary["scheme"].isin(scheme_sel)
    & summary["cost_regime"].isin(cost_sel)
)
filtered = summary[mask].copy()

if filtered.empty:
    st.warning("No rows match the current filter selection.")
    st.stop()

# --- Format the displayed table --------------------------------------------
display = filtered.rename(columns={
    "full_sample_return": "Full-sample daily ret",
    "matched_days_return": "Matched-days daily ret",
    "matched_days_count": "Matched days",
    "ann_return": "Ann. return",
    "ann_vol": "Ann. vol",
    "sharpe": "Sharpe",
    "trading_days": "Trading days",
    "avg_turnover": "Avg turnover",
    "total_pnl": "Cumulative daily return",
    "trailing_1y_return": "Trailing 1y ann. ret",
})

display = display[[
    "era", "model", "scheme", "cost_regime",
    "Full-sample daily ret", "Matched-days daily ret", "Matched days",
    "Ann. return", "Ann. vol", "Sharpe",
    "Trading days", "Avg turnover", "Cumulative daily return",
    "Trailing 1y ann. ret",
]].sort_values(["era", "cost_regime", "scheme", "model"]).reset_index(drop=True)

# Reorder column values for readable axis sorts.
display["model"] = pd.Categorical(display["model"], categories=MODEL_ORDER,
                                   ordered=True)
display["scheme"] = pd.Categorical(display["scheme"], categories=SCHEME_ORDER,
                                    ordered=True)

st.subheader("Summary table")
event = st.dataframe(
    display,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "Full-sample daily ret": st.column_config.NumberColumn(format="%.4f"),
        "Matched-days daily ret": st.column_config.NumberColumn(format="%.4f"),
        "Ann. return": st.column_config.NumberColumn(format="%.3f"),
        "Ann. vol": st.column_config.NumberColumn(format="%.3f"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Avg turnover": st.column_config.NumberColumn(format="%.3f"),
        "Cumulative daily return": st.column_config.NumberColumn(format="%.3f"),
        "Trailing 1y ann. ret": st.column_config.NumberColumn(format="%.3f"),
    },
)

selected_indices = event.selection.rows if hasattr(event, "selection") else []

# --- Overlay equity curves for selected rows -------------------------------
st.subheader("Equity curves for selected rows")
mode_col, spy_col = st.columns([1, 1])
with mode_col:
    equity_mode = st.radio(
        "Strategy curve mode",
        options=list(EQUITY_MODE_OPTIONS),
        horizontal=True,
        key="results_equity_mode",
    )
with spy_col:
    show_spy = st.checkbox("Show SPY overlay", value=True, key="results_show_spy")
mode_spec = equity_mode_spec(equity_mode)

if not selected_indices:
    st.info("Click one or more rows above to overlay their equity curves here.")
else:
    chosen = display.iloc[selected_indices][[
        "era", "model", "scheme", "cost_regime"
    ]]
    keys = chosen.to_dict("records")
    mask = pd.Series(False, index=equity.index)
    for k in keys:
        mask |= (
            (equity["era"] == k["era"])
            & (equity["model"] == k["model"])
            & (equity["scheme"] == k["scheme"])
            & (equity["cost_regime"] == k["cost_regime"])
        )
    sub = equity[mask].copy()
    sub["label"] = (
        sub["model"].astype(str)
        + " · "
        + sub["scheme"].astype(str)
        + " · "
        + sub["cost_regime"].astype(str)
        + " · "
        + sub["era"].astype(str)
    )
    fig = equity_curve_figure(
        sub,
        group_col="label",
        height=520,
        y_col=mode_spec["column"],
        y_axis_title=mode_spec["axis"],
    )
    if show_spy:
        add_spy_overlay(
            fig,
            spy,
            start=sub["date"].min(),
            end=sub["date"].max(),
            name="SPY total return (compounded, context only)",
        )
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

# --- Quick comparators -----------------------------------------------------
with st.expander("Best Sharpe per era + cost regime"):
    best = (
        summary.sort_values("sharpe", ascending=False)
        .groupby(["era", "cost_regime"], as_index=False).head(3)
        .sort_values(["era", "cost_regime", "sharpe"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    st.dataframe(
        best[[
            "era", "cost_regime", "model", "scheme", "sharpe",
            "ann_return", "avg_turnover",
        ]],
        width="stretch", hide_index=True,
        column_config={
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "ann_return": st.column_config.NumberColumn(format="%.3f"),
            "avg_turnover": st.column_config.NumberColumn(format="%.3f"),
        },
    )
