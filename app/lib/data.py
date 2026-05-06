"""Shared data loaders for the Streamlit app.

All loaders wrap a parquet read with ``st.cache_data`` so repeated page renders
do not re-read the files.  The app assumes every file under ``app/data/`` was
precomputed by the research pipeline and checked in as a runtime artifact.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
ROOT = APP_ROOT.parent
DATA_DIR = APP_ROOT / "data"
MODEL_RETURNS_REFRESH_LOG = DATA_DIR / "model_returns_refresh.log"
LATEST_POSITIONS_PATH = DATA_DIR / "live_latest_positions.parquet"

MODEL_RETURNS_REQUIRED_COLUMNS = {
    "date",
    "era",
    "model",
    "scheme",
    "cost_regime",
    "ret",
    "turnover",
    "active",
    "equity",
}
MODEL_RETURNS_GROUP_COLS = [
    "era",
    "model",
    "scheme",
    "k",
    "no_trade_band",
    "cost_regime",
    "band_threshold_bps",
    "half_turn_cost_bps",
]
MAX_ABS_MODEL_RETURN = 5.0


def _path(name: str) -> Path:
    return DATA_DIR / name


@lru_cache(maxsize=1)
def runtime_mode() -> dict:
    """Detect whether the app can run repo-backed live refreshes.

    The app itself stays usable from ``app/data`` in frozen mode.  Live mode
    is enabled only when the full repo backend and frozen period-33 models are
    present beside the app folder.
    """
    if os.environ.get("KRAUSS_FORCE_FROZEN_MODE") == "1":
        return {
            "mode": "frozen",
            "can_refresh": False,
            "reason": "frozen mode forced by KRAUSS_FORCE_FROZEN_MODE=1",
            "missing": [],
        }

    required = {
        "backend source": ROOT / "src" / "krauss" / "simulator" / "api.py",
        "refresh wrapper": ROOT / "scripts" / "refresh_app_model_returns.py",
        "model-return builder": ROOT / "scripts" / "build_model_returns.py",
        "prediction panel": ROOT / "data" / "processed" / "predictions_unified.parquet",
        "return panel": ROOT / "data" / "processed" / "returns_unified.parquet",
        "RF classifier": ROOT / "data" / "models_p2_ds" / "period_33" / "rf_cls.pkl",
        "RF regressor": ROOT / "data" / "models_p2_ds" / "period_33" / "rf_reg.pkl",
        "XGB classifier": ROOT / "data" / "models_p2_ds" / "period_33" / "xgb_cls.json",
        "XGB regressor": ROOT / "data" / "models_p2_ds" / "period_33" / "xgb_reg.json",
        "MT-DNN": ROOT / "data" / "models_p2_ds" / "period_33" / "mt_dnn.pt",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        return {
            "mode": "frozen",
            "can_refresh": False,
            "reason": "repo-backed live backend is unavailable",
            "missing": missing,
        }

    return {
        "mode": "live",
        "can_refresh": True,
        "reason": "repo-backed live backend detected",
        "missing": [],
    }


def maybe_start_model_returns_refresh() -> bool:
    """Kick off one detached incremental refresh per Streamlit browser session."""
    if os.environ.get("KRAUSS_DISABLE_AUTO_REFRESH") == "1":
        return False
    mode = runtime_mode()
    if not mode["can_refresh"]:
        return False
    if st.session_state.get("_krauss_model_returns_refresh_started"):
        return False
    st.session_state["_krauss_model_returns_refresh_started"] = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    refresh_script = ROOT / "scripts" / "refresh_app_model_returns.py"
    cmd = [sys.executable, str(refresh_script)]
    with MODEL_RETURNS_REFRESH_LOG.open("a", encoding="utf-8") as log:
        log.write("\n--- Streamlit session refresh requested ---\n")
        subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    return True


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
def _load_model_returns_cached(mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    df = pd.read_parquet(_path("model_returns.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    if "next_date" in df.columns:
        df["next_date"] = pd.to_datetime(df["next_date"])
    for col in ["era", "family", "model", "scheme", "cost_regime"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df.sort_values(MODEL_RETURNS_GROUP_COLS + ["date"]).copy()


def load_model_returns() -> pd.DataFrame:
    return _load_model_returns_cached(_path("model_returns.parquet").stat().st_mtime_ns)


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
def _load_model_returns_metadata_cached(mtime_ns: int) -> dict | None:
    _ = mtime_ns
    p = _path("model_returns_metadata.json")
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_returns_metadata() -> dict | None:
    p = _path("model_returns_metadata.json")
    if not p.exists():
        return None
    return _load_model_returns_metadata_cached(p.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _load_latest_positions_cached(mtime_ns: int) -> pd.DataFrame | None:
    _ = mtime_ns
    if not LATEST_POSITIONS_PATH.exists():
        return None
    df = pd.read_parquet(LATEST_POSITIONS_PATH)
    for col in ["signal_date", "last_realized_signal_date", "last_return_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date.astype(str)
    for col in ["family", "model", "scheme", "side", "cost_regime"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "no_trade_band" in df.columns:
        df["no_trade_band"] = df["no_trade_band"].astype(bool)
    return df


def load_latest_positions() -> pd.DataFrame | None:
    if not LATEST_POSITIONS_PATH.exists():
        return None
    return _load_latest_positions_cached(LATEST_POSITIONS_PATH.stat().st_mtime_ns)


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
    maybe_start_model_returns_refresh()
    required = ["equity_curves.parquet", "summary_table.parquet",
                "regime_labels.parquet", "pipeline_metadata.json"]
    return all(_path(p).exists() for p in required) and model_returns_is_valid()


def model_returns_is_valid() -> bool:
    ok, _ = model_returns_health()
    return ok


def model_returns_health() -> tuple[bool, list[str]]:
    parquet_path = _path("model_returns.parquet")
    metadata = load_model_returns_metadata()
    errors: list[str] = []

    if not parquet_path.exists():
        return False, ["missing app/data/model_returns.parquet"]
    if metadata is None:
        return False, ["missing app/data/model_returns_metadata.json"]

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        return False, [f"model_returns.parquet is unreadable: {exc}"]

    missing = sorted(MODEL_RETURNS_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        errors.append(f"model_returns.parquet is missing columns: {missing}")
        return False, errors

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as exc:
        errors.append(f"date column is invalid: {exc}")
        return False, errors

    dupes = int(df.duplicated(MODEL_RETURNS_GROUP_COLS + ["date"]).sum())
    if dupes:
        errors.append(f"duplicate strategy/date rows: {dupes:,}")

    if len(df) != metadata.get("row_count"):
        errors.append("row count does not match model_returns_metadata.json")

    era_days = metadata.get("era_trading_days", {})
    for era, expected in era_days.items():
        sub = df[df["era"].astype(str) == str(era)]
        if sub.empty:
            errors.append(f"missing era in model_returns.parquet: {era}")
            continue
        counts = sub.groupby(MODEL_RETURNS_GROUP_COLS, observed=True)["date"].nunique()
        if (counts != int(expected)).any():
            errors.append(f"incomplete date coverage for era {era}")

    for col in ["ret", "turnover", "active", "equity"]:
        if df[col].isna().any():
            errors.append(f"{col} contains nulls")

    extreme = df["ret"].abs() > MAX_ABS_MODEL_RETURN
    if extreme.any():
        sample = (
            df.loc[
                extreme,
                ["date", "era", "model", "scheme", "cost_regime", "ret"],
            ]
            .sort_values("date")
            .head(5)
            .to_dict(orient="records")
        )
        errors.append(
            f"ret contains {int(extreme.sum()):,} implausible daily return(s) "
            f"with |ret| > {MAX_ABS_MODEL_RETURN:g}; sample={sample}"
        )

    return not errors, errors


def missing_build_warning() -> None:
    _, model_return_errors = model_returns_health()
    mode = runtime_mode()
    st.error(
        "App data is missing or invalid. Restore the committed `app/data/` "
        "artifacts, or regenerate model returns from the repo root with "
        "`python scripts/build_model_returns.py --ensure`, then reload."
    )
    if mode["can_refresh"]:
        st.info(
            "A background model-return refresh is started automatically when "
            "the app opens. Progress is written to "
            "`app/data/model_returns_refresh.log`."
        )
    else:
        st.info(
            "The standalone app is in frozen mode, so no backend refresh will "
            "run from this folder alone."
        )
    if model_return_errors:
        with st.expander("Model returns health check"):
            for error in model_return_errors:
                st.write(f"- {error}")
    st.stop()
