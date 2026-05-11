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

from lib.charts import MODEL_ORDER, SCHEME_ORDER, equity_curve_figure  # noqa: E402
from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_equity_curves,
    load_summary_table,
    missing_build_warning,
)

if not data_build_is_complete():
    missing_build_warning()

equity = load_equity_curves()
summary = load_summary_table()

COST_LABELS = {
    "5bps_half_turn": "Post-cost: 5 bps per half-turn",
    "no_cost": "Gross / no transaction cost",
}


def _cost_label(value: str) -> str:
    return COST_LABELS.get(str(value), str(value))

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains "
    "what Sharpe, k = 10, P̂/Û and post-cost mean — read it first if any "
    "of the column names in the table below are unfamiliar."
)

st.caption(
    ":books: **Appendix** — the full table behind Results & Risk Diagnostics."
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
    default_era = [e for e in eras if "CRSP" in e] or eras
    era_sel = st.multiselect("Era", options=eras, default=default_era)

    model_opts = [m for m in MODEL_ORDER if m in summary["model"].unique()]
    model_sel = st.multiselect("Model", options=model_opts, default=model_opts)

    scheme_opts = [s for s in SCHEME_ORDER if s in summary["scheme"].unique()]
    scheme_sel = st.multiselect("Scheme", options=scheme_opts,
                                default=scheme_opts)

    cost_opts = sorted(summary["cost_regime"].unique().tolist())
    cost_sel = st.multiselect(
        "Cost regime",
        options=cost_opts,
        default=cost_opts,
        format_func=_cost_label,
    )

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
    "cum_return": "Cum. return",
    "trailing_1y_return": "Trailing 1y ann. ret",
})

display["cost_regime_label"] = display["cost_regime"].map(_cost_label)
display = display.sort_values(["era", "cost_regime", "scheme", "model"])

display = display[[
    "era", "model", "scheme", "cost_regime_label",
    "Full-sample daily ret", "Matched-days daily ret", "Matched days",
    "Ann. return", "Ann. vol", "Sharpe",
    "Trading days", "Avg turnover", "Cum. return", "Trailing 1y ann. ret",
]].reset_index(drop=True)

# Reorder column values for readable axis sorts.
display["model"] = pd.Categorical(display["model"], categories=MODEL_ORDER,
                                   ordered=True)
display["scheme"] = pd.Categorical(display["scheme"], categories=SCHEME_ORDER,
                                    ordered=True)

st.subheader("Summary table")
event = st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "cost_regime_label": st.column_config.TextColumn("Cost regime"),
        "Full-sample daily ret": st.column_config.NumberColumn(format="%.4f"),
        "Matched-days daily ret": st.column_config.NumberColumn(format="%.4f"),
        "Ann. return": st.column_config.NumberColumn(format="%.3f"),
        "Ann. vol": st.column_config.NumberColumn(format="%.3f"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Avg turnover": st.column_config.NumberColumn(format="%.3f"),
        "Cum. return": st.column_config.NumberColumn(format="%.3f"),
        "Trailing 1y ann. ret": st.column_config.NumberColumn(format="%.3f"),
    },
)

selected_indices = event.selection.rows if hasattr(event, "selection") else []

# --- Overlay equity curves for selected rows -------------------------------
st.subheader("Equity curves for selected rows")

if not selected_indices:
    st.info("Click one or more rows above to overlay their equity curves here.")
else:
    chosen = display.iloc[selected_indices][[
        "era", "model", "scheme", "cost_regime_label"
    ]]
    keys = chosen.to_dict("records")
    mask = pd.Series(False, index=equity.index)
    for k in keys:
        mask |= (
            (equity["era"] == k["era"])
            & (equity["model"] == k["model"])
            & (equity["scheme"] == k["scheme"])
            & (equity["cost_regime"].map(_cost_label) == k["cost_regime_label"])
        )
    sub = equity[mask].copy()
    sub["label"] = (
        sub["model"].astype(str) + " · " + sub["scheme"].astype(str)
        + " · " + sub["cost_regime"].map(_cost_label).astype(str)
        + " · " + sub["era"].astype(str)
    )
    fig = equity_curve_figure(sub, group_col="label", height=520)
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

# --- Quick comparators -----------------------------------------------------
with st.expander("Best Sharpe per era + cost regime"):
    best = (
        summary.sort_values("sharpe", ascending=False)
        .groupby(["era", "cost_regime"], as_index=False, observed=True).head(3)
        .sort_values(["era", "cost_regime", "sharpe"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    st.dataframe(
        best[[
            "era", "cost_regime", "model", "scheme", "sharpe",
            "ann_return", "avg_turnover",
        ]].assign(cost_regime=lambda d: d["cost_regime"].map(_cost_label)),
        use_container_width=True, hide_index=True,
        column_config={
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "ann_return": st.column_config.NumberColumn(format="%.3f"),
            "avg_turnover": st.column_config.NumberColumn(format="%.3f"),
        },
    )
