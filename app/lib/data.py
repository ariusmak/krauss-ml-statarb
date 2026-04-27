"""Shared data loaders for the Streamlit app.

All loaders wrap a parquet read with ``st.cache_data`` so repeated page renders
do not re-read the files.  The app assumes every file under ``app/data/`` was
produced by ``scripts/build_app_data.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"


def _path(name: str) -> Path:
    return DATA_DIR / name


@st.cache_data(show_spinner=False)
def load_equity_curves() -> pd.DataFrame:
    df = pd.read_parquet(_path("equity_curves.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    group_cols = ["era", "model", "scheme", "cost_regime"]
    for col in group_cols:
        df[col] = df[col].astype(str)
    df = df.sort_values(group_cols + ["date"]).copy()
    df["ret"] = df["ret"].fillna(0.0)

    # Some gated strategies only have rows on days when the gate produced a
    # portfolio. Fill missing era dates as no-trade days so curves stay flat
    # instead of visually starting late or cutting off early.
    era_dates = {
        era: pd.DataFrame({"date": dates.sort_values().unique()})
        for era, dates in df.groupby("era")["date"]
    }
    dense = []
    for key, grp in df.groupby(group_cols, sort=False):
        era, model, scheme, cost_regime = key
        full = era_dates[era].merge(
            grp[["date", "ret", "turnover"]],
            on="date",
            how="left",
        )
        full["era"] = era
        full["model"] = model
        full["scheme"] = scheme
        full["cost_regime"] = cost_regime
        full["ret"] = full["ret"].fillna(0.0)
        full["turnover"] = full["turnover"].fillna(0.0)
        dense.append(full)
    df = pd.concat(dense, ignore_index=True)
    df = df.sort_values(group_cols + ["date"]).copy()
    df["cum_pnl"] = df.groupby(group_cols, observed=True)["ret"].cumsum()
    df["cum_ret"] = df.groupby(group_cols, observed=True)["ret"].transform(
        lambda r: (1.0 + r).cumprod() - 1.0
    )
    return df


@st.cache_data(show_spinner=False)
def load_summary_table() -> pd.DataFrame:
    df = pd.read_parquet(_path("summary_table.parquet"))
    df["total_pnl"] = df["daily_return"] * df["trading_days"]
    return df


@st.cache_data(show_spinner=False)
def load_regime_labels() -> pd.DataFrame:
    df = pd.read_parquet(_path("regime_labels.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_daily_holdings() -> pd.DataFrame | None:
    p = _path("daily_holdings.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_disagreement_panel() -> pd.DataFrame | None:
    p = _path("disagreement_panel.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_pipeline_metadata() -> dict:
    with _path("pipeline_metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cost_bands() -> pd.DataFrame | None:
    p = _path("cost_bands.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_regime_k_sensitivity() -> pd.DataFrame | None:
    p = _path("regime_k_sensitivity.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_spy_benchmark() -> pd.DataFrame | None:
    p = _path("spy_benchmark.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_regime_leg_decomp() -> pd.DataFrame | None:
    p = _path("regime_leg_decomp.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


def data_build_is_complete() -> bool:
    required = ["equity_curves.parquet", "summary_table.parquet",
                "regime_labels.parquet", "pipeline_metadata.json"]
    return all(_path(p).exists() for p in required)


def missing_build_warning() -> None:
    st.error(
        "App data is missing. Run `python scripts/build_app_data.py` from the "
        "repo root, then reload the page."
    )
    st.stop()
