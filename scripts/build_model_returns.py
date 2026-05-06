"""Build and validate the app-facing model returns artifact.

The Streamlit app should consume this small daily-return table instead of
loading raw predictions or running backtests at page-render time.

Outputs:
    app/data/model_returns.parquet
    app/data/model_returns_metadata.json

Usage:
    python scripts/build_model_returns.py
    python scripts/build_model_returns.py --check-only
    python scripts/build_model_returns.py --ensure
    python scripts/build_model_returns.py --refresh-sources

Plain invocation is a full rebuild. The refresh/ensure modes use the
incremental path: lower-level returns/predictions are updated first, then only
missing model-return dates are appended. No-trade-band rows are the exception:
they are path-dependent, so the refresh replaces only those banded strategy
groups instead of rebuilding the whole artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from krauss.backtest.costs import (  # noqa: E402
    apply_transaction_costs,
    compute_turnover,
)
from krauss.backtest.no_trade_band import backtest_with_band  # noqa: E402
from krauss.backtest.portfolio import (  # noqa: E402
    aggregate_portfolio_returns,
    build_daily_portfolios,
)
from krauss.simulator.api import (  # noqa: E402
    _FAMILY_TO_TAG,
    DEFAULT_BAND_THRESHOLD_BPS,
    DEFAULT_HALF_TURN_BPS,
    PHASE2_ERA_START,
    _build_band_next_positions,
    _build_next_positions,
    _latest_realized_signal_date,
    _load_predictions,
    _load_returns,
    _materialise_score_columns,
    _select_long_short,
)

APP_DATA = ROOT / "app" / "data"
MODEL_RETURNS_PATH = APP_DATA / "model_returns.parquet"
METADATA_PATH = APP_DATA / "model_returns_metadata.json"
LATEST_POSITIONS_PATH = APP_DATA / "live_latest_positions.parquet"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions_unified.parquet"
RETURNS_PATH = ROOT / "data" / "processed" / "returns_unified.parquet"

PHASE1_START = pd.Timestamp("1992-12-17")
PHASE1_END = pd.Timestamp("2015-10-15")

PHASE1_FAMILIES = ("DNN", "XGB", "RF", "ENS1", "ENS2", "ENS3")
PHASE2_FAMILIES = ("DNN", "XGB", "RF", "ENS1")
SCHEMES = (
    "P-only",
    "U-only",
    "Z-comp",
    "Product",
    "P-gate(0.03)",
    "P-gate(0.05)",
)
BAND_SCHEMES = ("U-only", "Z-comp", "P-gate(0.03)", "P-gate(0.05)")
K = 10
ARTIFACT_VERSION = 1
MAX_ABS_MODEL_RETURN = 5.0
LIVE_POSITIONS_START = pd.Timestamp("2025-09-25")

REQUIRED_COLUMNS = {
    "date",
    "next_date",
    "era",
    "family",
    "model",
    "scheme",
    "k",
    "no_trade_band",
    "cost_regime",
    "band_threshold_bps",
    "half_turn_cost_bps",
    "gross_return",
    "net_return",
    "ret",
    "turnover",
    "cost",
    "long_ret",
    "short_ret",
    "n_long",
    "n_short",
    "active",
    "equity",
}

GROUP_COLS = [
    "era",
    "family",
    "model",
    "scheme",
    "k",
    "no_trade_band",
    "cost_regime",
    "band_threshold_bps",
    "half_turn_cost_bps",
]

MODEL_RETURN_COLUMNS = [
    "date",
    "next_date",
    *GROUP_COLS,
    "gross_return",
    "net_return",
    "ret",
    "turnover",
    "cost",
    "long_ret",
    "short_ret",
    "n_long",
    "n_short",
    "active",
    "equity",
]


def _log(message: str) -> None:
    print(f"[model_returns] {message}", flush=True)


def _source_snapshot() -> dict:
    """Cheap source-file snapshot for stale-artifact detection."""
    out: dict[str, dict] = {}
    for label, path in {
        "predictions_unified": PREDICTIONS_PATH,
        "returns_unified": RETURNS_PATH,
    }.items():
        if not path.exists():
            out[label] = {"exists": False}
            continue
        stat = path.stat()
        out[label] = {
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return out


def _run(cmd: list[str]) -> None:
    _log("running " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_predictions_ready(*, rebuild_returns: bool = False) -> None:
    """Refresh lower-level prediction artifacts before rebuilding returns."""
    if not PREDICTIONS_PATH.exists():
        _run([sys.executable, "scripts/build_predictions_unified.py"])
    cmd = [sys.executable, "scripts/refresh_live.py"]
    if rebuild_returns:
        cmd.append("--rebuild-returns")
    _run(cmd)


def _run_loaded_backtest(
    preds_window: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    family: str,
    scheme: str,
    k: int,
    no_trade_band: bool,
    band_threshold_bps: float,
    half_turn_cost_bps: float,
) -> pd.DataFrame:
    """Run one simulator configuration against preloaded source frames."""
    preds = _materialise_score_columns(preds_window, family, scheme)
    preds_for_engine = preds.rename(columns={"infocode": "permno"})

    if no_trade_band:
        u_col = f"u_{_FAMILY_TO_TAG[family]}"
        out = backtest_with_band(
            predictions=preds_for_engine,
            returns=returns,
            k=k,
            long_score_col="score_long",
            short_score_col="score_short",
            u_col=u_col,
            half_turn_bps=half_turn_cost_bps,
            swap_threshold_bps=band_threshold_bps,
        )
        daily = out["daily"].copy()
    else:
        selections = _select_long_short(preds_for_engine, k=k)
        if selections.empty:
            return pd.DataFrame()
        holdings = build_daily_portfolios(selections, returns, k=k)
        if holdings.empty:
            return pd.DataFrame()
        daily = aggregate_portfolio_returns(holdings)
        turn = compute_turnover(holdings, k=k)
        daily = apply_transaction_costs(daily, turn, half_turn_cost_bps)

    daily["date"] = pd.to_datetime(daily["date"])
    if "next_date" in daily.columns:
        daily["next_date"] = pd.to_datetime(daily["next_date"])
    return daily.sort_values("date").reset_index(drop=True)


def _tag_daily(
    daily: pd.DataFrame,
    *,
    era: str,
    family: str,
    scheme: str,
    no_trade_band: bool,
    band_threshold_bps: float,
    half_turn_cost_bps: float,
) -> pd.DataFrame:
    df = daily.copy()
    df["era"] = era
    df["family"] = family
    df["model"] = family
    df["scheme"] = scheme
    df["k"] = K
    df["no_trade_band"] = bool(no_trade_band)
    df["band_threshold_bps"] = float(band_threshold_bps if no_trade_band else 0.0)
    df["half_turn_cost_bps"] = float(half_turn_cost_bps)
    df["cost_regime"] = "band_10bps" if no_trade_band else "baseline_5bps"
    df["gross_return"] = df["port_ret"]
    df["net_return"] = df["port_ret_net"]
    df["ret"] = df["net_return"]
    df["active"] = True
    return df[
        [
            "date",
            "next_date",
            "era",
            "family",
            "model",
            "scheme",
            "k",
            "no_trade_band",
            "cost_regime",
            "band_threshold_bps",
            "half_turn_cost_bps",
            "gross_return",
            "net_return",
            "ret",
            "turnover",
            "cost",
            "long_ret",
            "short_ret",
            "n_long",
            "n_short",
            "active",
        ]
    ]


def _densify(rows: pd.DataFrame, era_axes: dict[str, pd.Series]) -> pd.DataFrame:
    """Fill no-trade days as explicit zero-return rows per configuration."""
    dense = []
    for keys, grp in rows.groupby(GROUP_COLS, sort=False, dropna=False):
        era = keys[0]
        spine = pd.DataFrame({"date": era_axes[era]})
        full = spine.merge(grp, on="date", how="left")
        for col, value in zip(GROUP_COLS, keys, strict=True):
            full[col] = value

        full["next_date"] = pd.to_datetime(full["next_date"])
        for col in [
            "gross_return",
            "net_return",
            "ret",
            "turnover",
            "cost",
            "long_ret",
            "short_ret",
        ]:
            full[col] = full[col].fillna(0.0)
        for col in ["n_long", "n_short"]:
            full[col] = full[col].fillna(0).astype("int16")
        full.loc[full["active"].isna(), "active"] = False
        full["active"] = full["active"].astype(bool)
        dense.append(full)

    out = pd.concat(dense, ignore_index=True)
    out = out.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    out["equity"] = out.groupby(GROUP_COLS, observed=True)["net_return"].transform(
        lambda r: (1.0 + r).cumprod()
    )
    return out


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["era", "family", "model", "scheme", "cost_regime"]:
        out[col] = out[col].astype("category")
    for col in [
        "gross_return",
        "net_return",
        "ret",
        "turnover",
        "cost",
        "long_ret",
        "short_ret",
        "equity",
    ]:
        out[col] = out[col].astype("float32")
    out["k"] = out["k"].astype("int16")
    out["n_long"] = out["n_long"].astype("int16")
    out["n_short"] = out["n_short"].astype("int16")
    return out


def _build_metadata(model_returns: pd.DataFrame) -> dict:
    groups = (
        model_returns[GROUP_COLS]
        .drop_duplicates()
        .sort_values(GROUP_COLS)
        .to_dict(orient="records")
    )
    era_axes = model_returns.groupby("era", observed=True)["date"].nunique()
    return {
        "artifact": "model_returns.parquet",
        "version": ARTIFACT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(model_returns)),
        "date_min": str(pd.Timestamp(model_returns["date"].min()).date()),
        "date_max": str(pd.Timestamp(model_returns["date"].max()).date()),
        "signal_date_min": str(pd.Timestamp(model_returns["date"].min()).date()),
        "signal_date_max": str(pd.Timestamp(model_returns["date"].max()).date()),
        "realized_return_date_max": str(
            pd.Timestamp(model_returns["next_date"].max()).date()
        ),
        "groups_count": int(len(groups)),
        "groups": groups,
        "era_trading_days": {str(k): int(v) for k, v in era_axes.items()},
        "source_snapshot": _source_snapshot(),
    }


def build_model_returns() -> tuple[pd.DataFrame, dict]:
    """Build the complete app-facing daily model returns table."""
    _log("loading unified predictions and returns")
    predictions = _load_predictions(None)
    returns = _load_returns(None)

    last_realized_signal_date = _latest_realized_signal_date(returns)
    if last_realized_signal_date is None:
        raise RuntimeError("returns_unified has fewer than two trading dates")
    phase2_end = min(last_realized_signal_date, pd.Timestamp(predictions["date"].max()))

    windows = [
        {
            "era": "1992-2015",
            "start": PHASE1_START,
            "end": PHASE1_END,
            "families": PHASE1_FAMILIES,
            "schemes": ("P-only",),
        },
        {
            "era": "2015-live",
            "start": PHASE2_ERA_START,
            "end": phase2_end,
            "families": PHASE2_FAMILIES,
            "schemes": SCHEMES,
        },
    ]

    rows: list[pd.DataFrame] = []
    era_axes: dict[str, pd.Series] = {}

    for window in windows:
        era = window["era"]
        start = window["start"]
        end = window["end"]
        preds_window = predictions[
            (predictions["date"] >= start) & (predictions["date"] <= end)
        ].copy()
        if preds_window.empty:
            raise RuntimeError(
                f"No predictions for {era}: {start.date()}->{end.date()}"
            )

        baseline = _run_loaded_backtest(
            preds_window,
            returns,
            family="ENS1",
            scheme="P-only",
            k=K,
            no_trade_band=False,
            band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
            half_turn_cost_bps=DEFAULT_HALF_TURN_BPS,
        )
        if baseline.empty:
            raise RuntimeError(f"Could not build date axis for {era}")
        era_axes[era] = baseline["date"].drop_duplicates().sort_values()

        for family in window["families"]:
            for scheme in window["schemes"]:
                band_options = [False]
                if era == "2015-live" and scheme in BAND_SCHEMES:
                    band_options.append(True)

                for no_trade_band in band_options:
                    _log(
                        f"{era} · {family} · {scheme} · "
                        f"{'band' if no_trade_band else 'baseline'}"
                    )
                    daily = _run_loaded_backtest(
                        preds_window,
                        returns,
                        family=family,
                        scheme=scheme,
                        k=K,
                        no_trade_band=no_trade_band,
                        band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
                        half_turn_cost_bps=DEFAULT_HALF_TURN_BPS,
                    )
                    if daily.empty:
                        continue
                    rows.append(
                        _tag_daily(
                            daily,
                            era=era,
                            family=family,
                            scheme=scheme,
                            no_trade_band=no_trade_band,
                            band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
                            half_turn_cost_bps=DEFAULT_HALF_TURN_BPS,
                        )
                    )

    sparse = pd.concat(rows, ignore_index=True)
    model_returns = _compact(_densify(sparse, era_axes))
    metadata = _build_metadata(model_returns)
    return model_returns, metadata


def write_model_returns() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    model_returns, metadata = build_model_returns()
    model_returns.to_parquet(
        MODEL_RETURNS_PATH,
        index=False,
        compression="zstd",
        compression_level=9,
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    size_mb = MODEL_RETURNS_PATH.stat().st_size / 1024 / 1024
    _log(
        f"wrote {MODEL_RETURNS_PATH.relative_to(ROOT)} "
        f"({len(model_returns):,} rows, {size_mb:.1f} MB)"
    )
    write_latest_positions()


def write_latest_positions() -> None:
    """Write latest live recommendations for every app-selectable live group."""
    _log("writing latest live recommendations")
    predictions = _load_predictions(PREDICTIONS_PATH)
    returns = _load_returns(RETURNS_PATH)
    last_pred_date = pd.Timestamp(predictions["date"].max())
    last_return_date = pd.Timestamp(returns["date"].max())
    last_realized_signal = _latest_realized_signal_date(returns)

    rows: list[pd.DataFrame] = []
    for family in PHASE2_FAMILIES:
        for scheme in SCHEMES:
            band_options = [False, True] if scheme in BAND_SCHEMES else [False]
            for no_trade_band in band_options:
                if no_trade_band:
                    positions = _build_band_next_positions(
                        predictions,
                        returns,
                        family=family,
                        scheme=scheme,
                        k=K,
                        last_pred_date=last_pred_date,
                        start=LIVE_POSITIONS_START,
                        band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
                        half_turn_cost_bps=DEFAULT_HALF_TURN_BPS,
                    )
                else:
                    positions = _build_next_positions(
                        predictions,
                        family=family,
                        scheme=scheme,
                        k=K,
                        last_pred_date=last_pred_date,
                        start=LIVE_POSITIONS_START,
                    )
                if positions is None or positions.empty:
                    continue

                tagged = positions.copy()
                tagged["family"] = family
                tagged["model"] = family
                tagged["scheme"] = scheme
                tagged["no_trade_band"] = bool(no_trade_band)
                tagged["cost_regime"] = (
                    "band_10bps" if no_trade_band else "baseline_5bps"
                )
                tagged["signal_date"] = str(last_pred_date.date())
                tagged["last_realized_signal_date"] = (
                    str(last_realized_signal.date())
                    if last_realized_signal is not None
                    else None
                )
                tagged["last_return_date"] = str(last_return_date.date())
                rows.append(tagged)

    latest = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    latest.to_parquet(
        LATEST_POSITIONS_PATH,
        index=False,
        compression="zstd",
        compression_level=9,
    )
    _log(
        f"wrote {LATEST_POSITIONS_PATH.relative_to(ROOT)} "
        f"({len(latest):,} rows)"
    )


def _current_groups(model_returns: pd.DataFrame) -> list[dict]:
    return (
        model_returns[GROUP_COLS]
        .drop_duplicates()
        .sort_values(GROUP_COLS)
        .to_dict(orient="records")
    )


def _group_mask(df: pd.DataFrame, group: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in GROUP_COLS:
        mask &= df[col] == group[col]
    return mask


def _normalise_dates(dates: pd.Series | list[pd.Timestamp]) -> pd.Series:
    return pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()


def _complete_group_rows(
    tagged: pd.DataFrame,
    dates: pd.Series,
    group: dict,
) -> pd.DataFrame:
    """Add explicit zero-return rows for a single group/date spine."""
    dates = _normalise_dates(dates)
    full = pd.DataFrame({"date": dates})
    if tagged.empty:
        for col in MODEL_RETURN_COLUMNS:
            if col not in full.columns:
                full[col] = pd.NA
    else:
        full = full.merge(tagged, on="date", how="left")

    for col in GROUP_COLS:
        full[col] = group[col]

    if "next_date" not in full.columns:
        full["next_date"] = pd.NaT
    full["next_date"] = pd.to_datetime(full["next_date"])

    for col in [
        "gross_return",
        "net_return",
        "ret",
        "turnover",
        "cost",
        "long_ret",
        "short_ret",
    ]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)
    for col in ["n_long", "n_short"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0).astype("int16")
    full.loc[full["active"].isna(), "active"] = False
    full["active"] = full["active"].astype(bool)
    if "equity" not in full.columns:
        full["equity"] = pd.NA
    return full[MODEL_RETURN_COLUMNS]


def _recompute_equity(model_returns: pd.DataFrame) -> pd.DataFrame:
    out = model_returns.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    out["equity"] = out.groupby(GROUP_COLS, observed=True)["net_return"].transform(
        lambda r: (1.0 + r).cumprod()
    )
    return out


def _latest_model_return_target(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Timestamp:
    last_realized_signal_date = _latest_realized_signal_date(returns)
    if last_realized_signal_date is None:
        raise RuntimeError("returns_unified has fewer than two trading dates")
    return min(last_realized_signal_date, pd.Timestamp(predictions["date"].max()))


def _append_non_band_group(
    *,
    group: dict,
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    current_max: pd.Timestamp,
    target_end: pd.Timestamp,
    new_dates: pd.Series,
) -> pd.DataFrame:
    preds_window = predictions[
        (predictions["date"] >= current_max) & (predictions["date"] <= target_end)
    ].copy()
    daily = _run_loaded_backtest(
        preds_window,
        returns,
        family=str(group["family"]),
        scheme=str(group["scheme"]),
        k=int(group["k"]),
        no_trade_band=False,
        band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
        half_turn_cost_bps=float(group["half_turn_cost_bps"]),
    )
    if daily.empty:
        return _complete_group_rows(pd.DataFrame(), new_dates, group)

    daily = daily[daily["date"] > current_max].copy()
    tagged = _tag_daily(
        daily,
        era=str(group["era"]),
        family=str(group["family"]),
        scheme=str(group["scheme"]),
        no_trade_band=False,
        band_threshold_bps=DEFAULT_BAND_THRESHOLD_BPS,
        half_turn_cost_bps=float(group["half_turn_cost_bps"]),
    )
    return _complete_group_rows(tagged, new_dates, group)


def _rebuild_band_groups(
    *,
    groups: list[dict],
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    full_axis: pd.Series,
    target_end: pd.Timestamp,
) -> pd.DataFrame:
    preds_window = predictions[
        (predictions["date"] >= PHASE2_ERA_START)
        & (predictions["date"] <= target_end)
    ].copy()
    rows: list[pd.DataFrame] = []
    for group in groups:
        _log(f"refresh band group · {group['family']} · {group['scheme']}")
        daily = _run_loaded_backtest(
            preds_window,
            returns,
            family=str(group["family"]),
            scheme=str(group["scheme"]),
            k=int(group["k"]),
            no_trade_band=True,
            band_threshold_bps=float(group["band_threshold_bps"]),
            half_turn_cost_bps=float(group["half_turn_cost_bps"]),
        )
        if daily.empty:
            tagged = pd.DataFrame()
        else:
            tagged = _tag_daily(
                daily,
                era=str(group["era"]),
                family=str(group["family"]),
                scheme=str(group["scheme"]),
                no_trade_band=True,
                band_threshold_bps=float(group["band_threshold_bps"]),
                half_turn_cost_bps=float(group["half_turn_cost_bps"]),
            )
        rows.append(_complete_group_rows(tagged, full_axis, group))
    if not rows:
        return pd.DataFrame(columns=MODEL_RETURN_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def append_model_returns() -> str:
    """Append new model-return dates, rebuilding only path-dependent band groups."""
    if not MODEL_RETURNS_PATH.exists() or not METADATA_PATH.exists():
        _log("model_returns artifact is missing; running full build")
        write_model_returns()
        return "rebuilt"

    ok, errors = validate_model_returns(check_sources=False)
    if not ok:
        for error in errors:
            _log(f"invalid: {error}")
        _log("artifact shape is invalid; running full build")
        write_model_returns()
        return "rebuilt"

    _log("loading current model_returns and source panels")
    current = pd.read_parquet(MODEL_RETURNS_PATH)
    current["date"] = pd.to_datetime(current["date"])
    predictions = _load_predictions(None)
    returns = _load_returns(None)
    target_end = _latest_model_return_target(predictions, returns)
    current_max = pd.Timestamp(current["date"].max())

    if target_end < current_max:
        raise RuntimeError(
            "Source data appears older than model_returns: "
            f"{target_end.date()} < {current_max.date()}"
        )

    if target_end == current_max:
        metadata = _build_metadata(current)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_latest_positions()
        _log(
            "model_returns already covers realized signals through "
            f"{target_end.date()}"
        )
        return "noop"

    new_dates = _normalise_dates(
        predictions[
            (predictions["date"] > current_max) & (predictions["date"] <= target_end)
        ]["date"]
    )
    if new_dates.empty:
        metadata = _build_metadata(current)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_latest_positions()
        _log("no realized prediction dates to append")
        return "noop"

    _log(
        f"appending {len(new_dates):,} model-return date(s): "
        f"{new_dates.min().date()} -> {new_dates.max().date()}"
    )
    groups = _current_groups(current)
    non_band_groups = [
        g
        for g in groups
        if str(g["era"]) == "2015-live" and not bool(g["no_trade_band"])
    ]
    band_groups = [
        g
        for g in groups
        if str(g["era"]) == "2015-live" and bool(g["no_trade_band"])
    ]

    appended: list[pd.DataFrame] = []
    for group in non_band_groups:
        _log(f"append group · {group['family']} · {group['scheme']}")
        appended.append(
            _append_non_band_group(
                group=group,
                predictions=predictions,
                returns=returns,
                current_max=current_max,
                target_end=target_end,
                new_dates=new_dates,
            )
        )

    band_rows = pd.DataFrame(columns=MODEL_RETURN_COLUMNS)
    if band_groups:
        full_phase2_axis = _normalise_dates(
            pd.concat(
                [
                    current.loc[
                        current["era"].astype(str) == "2015-live",
                        "date",
                    ],
                    new_dates,
                ],
                ignore_index=True,
            )
        )
        band_rows = _rebuild_band_groups(
            groups=band_groups,
            predictions=predictions,
            returns=returns,
            full_axis=full_phase2_axis,
            target_end=target_end,
        )

    appended_rows = (
        pd.concat(appended, ignore_index=True) if appended else pd.DataFrame()
    )
    keep_mask = ~(
        (current["era"].astype(str) == "2015-live")
        & current["no_trade_band"].astype(bool)
    )
    final = pd.concat(
        [current.loc[keep_mask, MODEL_RETURN_COLUMNS], appended_rows, band_rows],
        ignore_index=True,
    )
    final = _compact(_recompute_equity(final))
    metadata = _build_metadata(final)
    final.to_parquet(
        MODEL_RETURNS_PATH,
        index=False,
        compression="zstd",
        compression_level=9,
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_latest_positions()
    _log(f"appended model_returns through {target_end.date()}")
    return "appended"


def validate_model_returns(
    returns_path: Path = MODEL_RETURNS_PATH,
    metadata_path: Path = METADATA_PATH,
    *,
    check_sources: bool = True,
) -> tuple[bool, list[str]]:
    """Validate the app-facing model returns artifact."""
    errors: list[str] = []
    if not returns_path.exists():
        return False, [f"missing {returns_path}"]
    if not metadata_path.exists():
        return False, [f"missing {metadata_path}"]

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"metadata unreadable: {exc}"]

    if metadata.get("version") != ARTIFACT_VERSION:
        errors.append(
            f"version mismatch: {metadata.get('version')} != {ARTIFACT_VERSION}"
        )

    if check_sources and metadata.get("source_snapshot") != _source_snapshot():
        errors.append("source files changed since model_returns was built")

    try:
        df = pd.read_parquet(returns_path)
    except Exception as exc:
        return False, [f"parquet unreadable: {exc}"]

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return False, errors

    df["date"] = pd.to_datetime(df["date"])
    dupes = int(df.duplicated(GROUP_COLS + ["date"]).sum())
    if dupes:
        errors.append(f"duplicate group/date rows: {dupes:,}")

    if len(df) != metadata.get("row_count"):
        errors.append(f"row count mismatch: {len(df):,} != {metadata.get('row_count')}")

    expected_groups = metadata.get("groups", [])
    actual_groups = df[GROUP_COLS].drop_duplicates().to_dict(orient="records")
    if len(actual_groups) != metadata.get("groups_count"):
        errors.append("group count mismatch")
    if expected_groups and sorted(actual_groups, key=str) != sorted(
        expected_groups, key=str
    ):
        errors.append("group set mismatch")

    required_numeric = [
        "gross_return",
        "net_return",
        "ret",
        "turnover",
        "cost",
        "n_long",
        "n_short",
        "equity",
    ]
    for col in required_numeric:
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

    for era, expected_days in metadata.get("era_trading_days", {}).items():
        sub = df[df["era"].astype(str) == era]
        if sub.empty:
            errors.append(f"missing era {era}")
            continue
        counts = sub.groupby(GROUP_COLS, observed=True)["date"].nunique()
        bad = counts[counts != expected_days]
        if len(bad):
            errors.append(f"{era} has incomplete date coverage in {len(bad)} groups")

    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--ensure",
        action="store_true",
        help=(
            "If possible, append missing dates; rebuild only if the artifact "
            "is invalid."
        ),
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Refresh live returns/predictions first, then append model_returns.",
    )
    args = parser.parse_args()

    if args.refresh_sources:
        ensure_predictions_ready(rebuild_returns=True)
        append_model_returns()
        ok, errors = validate_model_returns()
        if not ok:
            for error in errors:
                _log(f"invalid after refresh: {error}")
            raise SystemExit(1)
        _log("validation passed")
        return

    ok, errors = validate_model_returns()
    if args.check_only:
        if ok:
            _log("model_returns.parquet is valid")
            return
        for error in errors:
            _log(f"invalid: {error}")
        raise SystemExit(1)

    if args.ensure:
        if not ok:
            for error in errors:
                _log(f"invalid: {error}")
            ensure_predictions_ready()
        append_model_returns()
        ok, errors = validate_model_returns()
        if not ok:
            for error in errors:
                _log(f"invalid after ensure: {error}")
            raise SystemExit(1)
        _log("validation passed")
        return

    write_model_returns()
    ok, errors = validate_model_returns()
    if not ok:
        for error in errors:
            _log(f"invalid after build: {error}")
        raise SystemExit(1)
    _log("validation passed")


if __name__ == "__main__":
    main()
