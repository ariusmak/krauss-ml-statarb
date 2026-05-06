"""Historical and live simulation interface backed by model_returns.parquet."""
from __future__ import annotations

import calendar
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Simulator", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_latest_positions,
    load_model_returns,
    load_model_returns_metadata,
    load_spy_benchmark,
    missing_build_warning,
    model_returns_health,
    runtime_mode,
)

if not data_build_is_complete():
    missing_build_warning()

equity = load_model_returns()
metadata = load_model_returns_metadata() or {}
spy = load_spy_benchmark()
health_ok, health_errors = model_returns_health()
mode = runtime_mode()

equity["date"] = pd.to_datetime(equity["date"])
equity["next_date"] = pd.to_datetime(equity.get("next_date", equity["date"]))

signal_min = pd.Timestamp(equity["date"].min()).date()
signal_max = pd.Timestamp(equity["date"].max()).date()
realized_min = pd.Timestamp(equity["next_date"].min()).date()
realized_max = pd.Timestamp(equity["next_date"].max()).date()

SCHEME_ORDER = [
    "P-only",
    "U-only",
    "Z-comp",
    "Product",
    "P-gate(0.03)",
    "P-gate(0.05)",
]
MODEL_ORDER = ["ENS1", "RF", "XGB", "DNN", "ENS2", "ENS3"]
BAND_ELIGIBLE_SCHEMES = {
    "U-only",
    "Z-comp",
    "P-gate(0.03)",
    "P-gate(0.05)",
}
COST_REGIME_LABELS = {
    "Post-cost: 5 bps per half-turn": "baseline_5bps",
    "No-trade band 10 bps": "band_10bps",
}


def _clamp_date(value, lower: date, upper: date) -> date:
    day = pd.Timestamp(value).date()
    return max(min(day, upper), lower)


def _month_options(year: int, lower: date, upper: date) -> list[int]:
    lo = lower.month if year == lower.year else 1
    hi = upper.month if year == upper.year else 12
    return list(range(lo, hi + 1))


def _day_options(year: int, month: int, lower: date, upper: date) -> list[int]:
    lo = lower.day if year == lower.year and month == lower.month else 1
    hi = (
        upper.day
        if year == upper.year and month == upper.month
        else calendar.monthrange(year, month)[1]
    )
    return list(range(lo, hi + 1))


def _closest_option(value: int, options: list[int]) -> int:
    return max(min(value, options[-1]), options[0])


def _normalise_date_parts(
    prefix: str,
    default: date,
    lower: date,
    upper: date,
) -> None:
    date_key = f"{prefix}_date"
    year_key = f"{prefix}_year"
    month_key = f"{prefix}_month"
    day_key = f"{prefix}_day"

    current = _clamp_date(st.session_state.get(date_key, default), lower, upper)
    year = int(st.session_state.get(year_key, current.year))
    month = int(st.session_state.get(month_key, current.month))
    day = int(st.session_state.get(day_key, current.day))
    original = (year, month, day)

    year = max(min(year, upper.year), lower.year)
    if year == upper.year and (month, day) > (upper.month, upper.day):
        month, day = upper.month, upper.day
    elif year == lower.year and (month, day) < (lower.month, lower.day):
        month, day = lower.month, lower.day
    else:
        month = _closest_option(month, _month_options(year, lower, upper))
        day = min(day, calendar.monthrange(year, month)[1])
        day = _closest_option(day, _day_options(year, month, lower, upper))

    if (year, month, day) != original:
        st.session_state[f"_{prefix}_date_clamped"] = True

    st.session_state[date_key] = date(year, month, day)
    st.session_state[year_key] = year
    st.session_state[month_key] = month
    st.session_state[day_key] = day


def _bounded_date_picker(
    container,
    label: str,
    prefix: str,
    default: date,
    lower: date,
    upper: date,
) -> date:
    year_key = f"{prefix}_year"
    month_key = f"{prefix}_month"
    day_key = f"{prefix}_day"

    _normalise_date_parts(prefix, default, lower, upper)
    container.markdown(f"**{label}**")
    years = list(range(lower.year, upper.year + 1))

    year = container.selectbox(
        "Year",
        years,
        index=years.index(st.session_state[year_key]),
        key=year_key,
    )
    months = _month_options(year, lower, upper)
    if st.session_state[month_key] not in months:
        st.session_state[month_key] = _closest_option(
            int(st.session_state[month_key]), months
        )
    m_col, d_col = container.columns(2)
    month = m_col.selectbox(
        "Month",
        months,
        format_func=lambda m: f"{m:02d}",
        index=months.index(st.session_state[month_key]),
        key=month_key,
    )
    days = _day_options(year, month, lower, upper)
    if st.session_state[day_key] not in days:
        st.session_state[day_key] = _closest_option(
            int(st.session_state[day_key]), days
        )
    day = d_col.selectbox(
        "Day",
        days,
        format_func=lambda d: f"{d:02d}",
        index=days.index(st.session_state[day_key]),
        key=day_key,
    )
    picked = date(year, month, day)
    st.session_state[f"{prefix}_date"] = picked
    return picked


def _ordered_available(values: pd.Series, preferred: list[str]) -> list[str]:
    available = {str(v) for v in values.dropna().unique()}
    ordered = [v for v in preferred if v in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _cost_options(model_returns: pd.DataFrame, model: str, scheme: str) -> list[str]:
    available = set(
        model_returns.loc[
            (model_returns["model"] == model) & (model_returns["scheme"] == scheme),
            "cost_regime",
        ]
    )
    options = [
        label
        for label, regime in COST_REGIME_LABELS.items()
        if regime == "baseline_5bps" and regime in available
    ]
    if scheme in BAND_ELIGIBLE_SCHEMES and "band_10bps" in available:
        options.append("No-trade band 10 bps")
    return options


def _strategy_controls(frame: pd.DataFrame, prefix: str):
    c1, c2, c3 = st.columns(3)
    scheme_options = _ordered_available(frame["scheme"], SCHEME_ORDER)
    scheme = c1.selectbox("Scoring scheme", scheme_options, key=f"{prefix}_scheme")

    model_options = _ordered_available(
        frame.loc[frame["scheme"] == scheme, "model"], MODEL_ORDER
    )
    model = c2.selectbox("Model", model_options, key=f"{prefix}_model")

    cost_options = _cost_options(frame, model, scheme)
    if not cost_options:
        st.error("No cost-regime rows are available for this model/scheme pair.")
        return None
    cost_label = c3.selectbox("Cost regime", cost_options, key=f"{prefix}_cost")
    return model, scheme, cost_label, COST_REGIME_LABELS[cost_label]


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float, digits: int = 2) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:+.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _annualised_mean(r: pd.Series) -> float:
    return float(252 * r.mean())


def _annualised_std(r: pd.Series) -> float:
    s = r.std(ddof=0)
    return float(np.sqrt(252) * s) if s > 0 else 0.0


def _sharpe(r: pd.Series) -> float:
    s = r.std(ddof=0)
    return float(np.sqrt(252) * r.mean() / s) if s > 0 else float("nan")


def _sortino(r: pd.Series) -> float:
    downside = r[r < 0]
    if len(downside) < 2:
        return float("nan")
    s = downside.std(ddof=0)
    return float(np.sqrt(252) * r.mean() / s) if s > 0 else float("nan")


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def _metric(col, label: str, value: str, helptext: str = "") -> None:
    with col:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                "<div style='font-size:22px;font-weight:700;color:#142B4F;"
                "margin-top:-4px;line-height:1.1'>" + value + "</div>",
                unsafe_allow_html=True,
            )
            if helptext:
                st.caption(helptext)


def _strategy_window(
    frame: pd.DataFrame,
    *,
    model: str,
    scheme: str,
    cost_regime: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    strat = frame.query(
        "model == @model and scheme == @scheme and cost_regime == @cost_regime"
    ).copy()
    strat = strat[(strat["date"] >= start_ts) & (strat["next_date"] <= end_ts)]
    return strat.sort_values("date").reset_index(drop=True)


def _spy_window(realized_start: pd.Timestamp, realized_end: pd.Timestamp,
                capital: float) -> tuple[pd.DataFrame, float]:
    if spy is None or spy.empty:
        return pd.DataFrame(), float("nan")
    spy_in = spy[
        (spy["date"] >= realized_start) & (spy["date"] <= realized_end)
    ].copy()
    spy_in = spy_in.sort_values("date").reset_index(drop=True)
    if spy_in.empty:
        return spy_in, float("nan")
    spy_in["ret"] = spy_in["ret"].fillna(0.0)
    spy_in["nav"] = capital * (1.0 + spy_in["ret"]).cumprod()
    return spy_in, float(spy_in["nav"].iloc[-1] / capital - 1.0)


def _render_replay(
    strat: pd.DataFrame,
    *,
    model: str,
    scheme: str,
    cost_label: str,
    capital: float,
    include_spy: bool = True,
) -> None:
    if strat.empty:
        st.warning(
            "The selected window has no rows in model_returns.parquet. "
            "Try a different date window or parameter combination."
        )
        return
    if len(strat) < 20:
        st.warning(
            f"The selected window has only {len(strat)} trading days. "
            "Risk metrics on fewer than about 20 days are noisy."
        )
    min_long = int(strat["n_long"].min()) if "n_long" in strat.columns else 10
    min_short = int(strat["n_short"].min()) if "n_short" in strat.columns else 10
    if (scheme.startswith("P-gate") or "band" in cost_label.lower()) and (
        min_long < 10 or min_short < 10
    ):
        st.info(
            "This window includes days with fewer than 10 names on at least "
            "one side. Gated strategies can become concentrated when only a "
            "small number of stocks clear the P-hat confidence threshold."
        )

    ret = strat["ret"].fillna(0.0).reset_index(drop=True)
    strat = strat.copy()
    strat["nav"] = capital * (1.0 + ret).cumprod()
    strat["signal_date_label"] = strat["date"].dt.strftime("%Y-%m-%d")
    strat_cum_ret = float(strat["nav"].iloc[-1] / capital - 1.0)
    realized_start = pd.Timestamp(strat["next_date"].min())
    realized_end = pd.Timestamp(strat["next_date"].max())
    spy_in, spy_cum_ret = _spy_window(realized_start, realized_end, capital)

    st.subheader("Equity curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strat["next_date"],
            y=strat["nav"],
            mode="lines",
            name=f"Strategy · {model} · {scheme} · {cost_label}",
            line=dict(color="#2e7d32", width=2),
            customdata=strat[["signal_date_label"]],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>NAV = $%{y:,.2f}"
                "<br>signal = %{customdata[0]}<extra></extra>"
            ),
        )
    )
    if include_spy and not spy_in.empty:
        fig.add_trace(
            go.Scatter(
                x=spy_in["date"],
                y=spy_in["nav"],
                mode="lines",
                name="SPY total return, context only",
                line=dict(color="#888", width=1.5, dash="dot"),
            )
        )
    fig.update_layout(
        height=430,
        yaxis_title="Portfolio value ($)",
        xaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=60, r=20, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown")
    curve = (1.0 + ret).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=strat["next_date"],
            y=drawdown * 100,
            fill="tozeroy",
            mode="lines",
            line=dict(color="#c62828", width=1),
            fillcolor="rgba(198, 40, 40, 0.22)",
            hovertemplate="%{x|%Y-%m-%d}<br>drawdown = %{y:.2f}%<extra></extra>",
        )
    )
    fig_dd.update_layout(
        height=230,
        yaxis_title="Drawdown (%)",
        xaxis_title=None,
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=40),
    )
    fig_dd.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_dd, use_container_width=True)

    sharpe = _sharpe(ret)
    sortino = _sortino(ret)
    max_dd = _max_drawdown(ret)
    ann_ret = _annualised_mean(ret)
    ann_vol = _annualised_std(ret)
    hit_rate = float((ret > 0).mean())
    worst = float(ret.min())
    best = float(ret.max())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else float("nan")

    st.subheader("Risk metrics")
    r1 = st.columns(3)
    _metric(
        r1[0],
        "Cumulative return",
        f"{_money(strat['nav'].iloc[-1] - capital)}  ({_pct(strat_cum_ret)})",
        f"Final NAV {_money(strat['nav'].iloc[-1])}.",
    )
    _metric(r1[1], "Annualized return", _pct(ann_ret), "Mean daily return x 252.")
    _metric(r1[2], "Annualized volatility", _pct(ann_vol), "Std-dev x sqrt(252).")
    r2 = st.columns(3)
    _metric(r2[0], "Sharpe ratio", _num(sharpe), "Annualized return / volatility.")
    _metric(r2[1], "Sortino ratio", _num(sortino), "Downside-risk-adjusted return.")
    _metric(r2[2], "Max drawdown", _pct(max_dd), "Worst peak-to-trough loss.")
    r3 = st.columns(3)
    _metric(r3[0], "Calmar ratio", _num(calmar), "Annualized return / drawdown.")
    _metric(r3[1], "Hit rate", _pct(hit_rate, 1), "Share of positive days.")
    _metric(r3[2], "Best / worst day", f"{_pct(best)} / {_pct(worst)}")
    r4 = st.columns(3)
    _metric(r4[0], "Realized days", f"{len(ret):,}", "Rows in the selected window.")
    _metric(r4[1], "Avg turnover", _num(float(strat["turnover"].mean()), 3))
    _metric(
        r4[2],
        "SPY same-window return",
        _pct(spy_cum_ret) if not np.isnan(spy_cum_ret) else "n/a",
        "Context only; strategy is dollar-neutral.",
    )


st.title("Simulator")
st.caption(
    "Two modes over the same validated artifact. Historical Backtest replays "
    "any realized window in `model_returns.parquet`; Live Simulation auto-ends "
    "at the latest realized return date and shows the latest signal."
)
st.caption(
    "Rows are keyed by signal date. "
    f"Signals through {metadata.get('signal_date_max', signal_max)} have P&L "
    f"realized through {metadata.get('realized_return_date_max', realized_max)}."
)
if mode["can_refresh"]:
    st.caption("Runtime mode: repo-backed live refresh is available.")
else:
    st.caption(
        "Runtime mode: frozen standalone snapshot. "
        "Repo-backed refresh is unavailable."
    )

historical_tab, live_tab = st.tabs(["Historical Backtest", "Live Simulation"])

with historical_tab:
    st.markdown("Replay a fixed historical window with explicit start and end dates.")
    with st.container(border=True):
        st.markdown("**Inputs**")
        c1, c2, c3 = st.columns(3)
        hist_start = _bounded_date_picker(
            c1,
            "Start signal date",
            "hist_start",
            pd.Timestamp("2010-01-01").date(),
            signal_min,
            signal_max,
        )
        hist_end = _bounded_date_picker(
            c2,
            "P&L through date",
            "hist_end",
            realized_max,
            realized_min,
            realized_max,
        )
        hist_capital = c3.number_input(
            "Initial capital ($)",
            min_value=1_000.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=1_000.0,
            format="%.2f",
            key="hist_capital",
        )
        hist_controls = _strategy_controls(equity, "hist")
        if st.session_state.pop("_hist_end_date_clamped", False):
            st.info(
                "End date was moved to the latest realized return date: "
                f"{realized_max.isoformat()}."
            )
        if st.session_state.pop("_hist_start_date_clamped", False):
            st.info(
                "Start date was moved to the latest available signal date: "
                f"{signal_max.isoformat()}."
            )

    if pd.Timestamp(hist_end) < pd.Timestamp(hist_start):
        st.error("End date must be on or after the start date.")
    elif hist_controls is not None:
        hist_model, hist_scheme, hist_cost_label, hist_cost_regime = hist_controls
        hist = _strategy_window(
            equity,
            model=hist_model,
            scheme=hist_scheme,
            cost_regime=hist_cost_regime,
            start=hist_start,
            end=hist_end,
        )
        _render_replay(
            hist,
            model=hist_model,
            scheme=hist_scheme,
            cost_label=hist_cost_label,
            capital=hist_capital,
        )

with live_tab:
    st.markdown(
        "Live Simulation uses the latest realized model-return rows. The end "
        "date is fixed to the latest realized return date; the latest signal "
        "itself becomes realized only after the following trading session."
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Runtime mode", "Repo-backed live" if mode["can_refresh"] else "Frozen")
    s2.metric(
        "Latest realized signal",
        str(metadata.get("signal_date_max", signal_max)),
    )
    s3.metric("P&L realized through", str(metadata.get(
        "realized_return_date_max", realized_max
    )))
    s4.metric("Strategy groups", f"{metadata.get('groups_count', 'n/a')}")
    if not health_ok:
        with st.expander("Health-check errors", expanded=True):
            for err in health_errors:
                st.write(f"- {err}")
    if not mode["can_refresh"]:
        with st.expander("Why live refresh is unavailable"):
            st.write(mode["reason"])
            for missing in mode["missing"]:
                st.write(f"- missing {missing}")

    live_equity = equity[equity["era"].astype(str).str.contains("2015")].copy()
    with st.container(border=True):
        st.markdown("**Inputs**")
        c1, c2 = st.columns([1, 2])
        live_start = _bounded_date_picker(
            c1,
            "Simulation start signal date",
            "live_start",
            pd.Timestamp("2025-09-25").date(),
            max(pd.Timestamp("2015-10-16").date(), signal_min),
            signal_max,
        )
        live_capital = c2.number_input(
            "Initial capital ($)",
            min_value=1_000.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=1_000.0,
            format="%.2f",
            key="live_capital",
        )
        live_controls = _strategy_controls(live_equity, "live")
        if st.session_state.pop("_live_start_date_clamped", False):
            st.info(
                "Start date was moved to the latest available signal date: "
                f"{signal_max.isoformat()}."
            )

    if live_controls is not None:
        live_model, live_scheme, live_cost_label, live_cost_regime = live_controls
        live = _strategy_window(
            live_equity,
            model=live_model,
            scheme=live_scheme,
            cost_regime=live_cost_regime,
            start=live_start,
            end=realized_max,
        )
        _render_replay(
            live,
            model=live_model,
            scheme=live_scheme,
            cost_label=live_cost_label,
            capital=live_capital,
            include_spy=False,
        )

        st.subheader("Latest recommended positions")
        positions_all = load_latest_positions()
        if positions_all is None or positions_all.empty:
            st.warning("Latest frozen recommendations are unavailable.")
        else:
            no_trade_band = live_cost_regime == "band_10bps"
            positions = positions_all[
                (positions_all["family"] == live_model)
                & (positions_all["scheme"] == live_scheme)
                & (positions_all["no_trade_band"] == no_trade_band)
            ].copy()
            if positions.empty:
                st.info("No latest positions are available for this configuration.")
            else:
                meta_row = positions.iloc[0]
                st.caption(
                    f"Signal date {meta_row.get('signal_date')}; latest "
                    f"realized signal {meta_row.get('last_realized_signal_date')}; "
                    f"source returns through {meta_row.get('last_return_date')}."
                )
                display = positions.copy()
                drop_cols = [
                    "family",
                    "model",
                    "scheme",
                    "no_trade_band",
                    "cost_regime",
                    "signal_date",
                    "last_realized_signal_date",
                    "last_return_date",
                ]
                display = display.drop(columns=[c for c in drop_cols if c in display])
                for col in ["score", "p", "u"]:
                    if col in display.columns:
                        display[col] = pd.to_numeric(display[col], errors="coerce")
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "score": st.column_config.NumberColumn(format="%.6f"),
                        "p": st.column_config.NumberColumn(format="%.4f"),
                        "u": st.column_config.NumberColumn(format="%.6f"),
                    },
                )

st.warning(
    "Historical simulation, not a live trading recommendation. This page "
    "replays frozen model outputs and validated realized returns. The "
    "post-2015 edge materially decays after costs."
)
