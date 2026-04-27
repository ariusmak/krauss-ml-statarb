"""Reusable Plotly chart builders."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SCHEME_ORDER = [
    "P-only", "U-only", "Z-comp", "Product", "P-gate(0.03)", "P-gate(0.05)"
]
MODEL_ORDER = ["RF", "XGB", "DNN", "ENS1", "ENS2", "ENS3"]

EQUITY_MODE_OPTIONS = {
    "Cumulative P&L": {
        "column": "cum_pnl",
        "axis": "Cumulative daily return (sum)",
        "hover": "cum daily ret",
    },
    "Compounded return": {
        "column": "cum_ret",
        "axis": "Compounded return",
        "hover": "compounded ret",
    },
}


def equity_mode_spec(mode: str) -> dict[str, str]:
    """Return chart metadata for a strategy equity display mode."""
    return EQUITY_MODE_OPTIONS.get(mode, EQUITY_MODE_OPTIONS["Cumulative P&L"])


def equity_curve_figure(
    df: pd.DataFrame,
    *,
    group_col: str = "label",
    title: str | None = None,
    height: int = 460,
    log_y: bool = False,
    y_col: str = "cum_pnl",
    y_axis_title: str | None = None,
) -> go.Figure:
    """Plotly line chart of cumulative daily long-short returns.

    ``df`` must include columns ``date``, ``y_col`` and whatever ``group_col``
    identifies the distinct lines to draw.
    """
    if y_col not in df.columns:
        y_col = "cum_ret"
    label = y_axis_title or (
        "Compounded return" if y_col == "cum_ret"
        else "Cumulative daily return (sum)"
    )
    fig = px.line(
        df,
        x="date",
        y=y_col,
        color=group_col,
        labels={y_col: label},
        hover_data={y_col: ":.3f", "date": True, group_col: True},
    )
    layout = dict(
        height=height,
        hovermode="x unified",
        legend_title=None,
        xaxis_title=None,
        yaxis_title=label,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
    )
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_xaxes(rangeslider_visible=False, showgrid=True)
    return fig


def add_spy_overlay(
    fig: go.Figure,
    spy: pd.DataFrame | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    name: str = "SPY total return (compounded)",
) -> go.Figure:
    """Add a rebased SPY total-return trace to an existing equity figure."""
    if spy is None or spy.empty:
        return fig

    spy_in = spy.copy()
    spy_in["date"] = pd.to_datetime(spy_in["date"])
    mask = spy_in["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= spy_in["date"] <= pd.Timestamp(end)
    spy_in = spy_in[mask].sort_values("date").copy()
    if spy_in.empty:
        return fig

    spy_in["ret"] = spy_in["ret"].fillna(0.0)
    spy_in["cum_ret"] = (1.0 + spy_in["ret"]).cumprod() - 1.0
    fig.add_trace(
        go.Scatter(
            x=spy_in["date"],
            y=spy_in["cum_ret"],
            mode="lines",
            name=name,
            line=dict(color="#888", width=1.7, dash="dot"),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>SPY compounded ret = "
                "%{y:.2f}<extra></extra>"
            ),
        )
    )
    return fig
