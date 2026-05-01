"""
Run Phase 2 extension on the Datastream US-only dataset.

This is the Datastream analogue of `scripts/run_phase2.py`, using:

- data/datastream/ds_features_usonly.parquet
- data/datastream/ds_labels_usonly.parquet
- data/datastream/ds_daily_returns_usonly.parquet
- data/datastream/ds_universe_daily_usonly.parquet

The script trains the Phase 2 model families:

- RF classifier + regressor
- XGB classifier + regressor
- multitask DNN

and writes P-hat / U-hat predictions for the requested periods.

By default it runs only the new extension periods beyond the original 23:

- periods 23 through the latest available period

Outputs:
    data/models_p2_ds/period_{id}/...
    data/datastream/predictions_phase2_ds.parquet
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods
from krauss.models.ensembles_phase2 import (
    compute_score_families,
    ens1_p_hat,
    ens1_u_hat,
)

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"
MODELS = ROOT / "data" / "models_p2_ds"

FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)] + [f"R{i}" for i in range(40, 241, 20)]
)


def load_data():
    """Load Datastream US-only Phase 2 inputs."""
    features = pd.read_parquet(DS_DIR / "ds_features_usonly.parquet")
    labels = pd.read_parquet(DS_DIR / "ds_labels_usonly.parquet")
    returns = pd.read_parquet(DS_DIR / "ds_daily_returns_usonly.parquet")
    eligible = pd.read_parquet(DS_DIR / "ds_universe_daily_usonly.parquet")
    return features, labels, returns, eligible


def build_panel(features, labels, eligible, dates_set):
    """
    Build the Datastream modeling panel for a set of dates.
    Includes both y_binary and u_excess for Phase 2.
    """
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][
        ["date", "infocode", "y_binary", "u_excess"]
    ].copy()

    panel = feat.merge(lab, on=["date", "infocode"], how="inner")
    panel = panel.merge(eligible, on=["date", "infocode"], how="inner")
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def save_period_meta(model_dir, period_id, sp, train_panel, trade_panel):
    """Save period metadata as JSON."""
    meta = {
        "period_id": period_id,
        "phase": 2,
        "dataset": "datastream_usonly",
        "train_start": str(sp.train_start.date()),
        "train_end": str(sp.train_end.date()),
        "usable_train_start": str(sp.usable_train_start.date()),
        "trade_start": str(sp.trade_start.date()),
        "trade_end": str(sp.trade_end.date()),
        "n_train_obs": len(train_panel),
        "n_trade_obs": len(trade_panel),
        "n_train_stocks": int(train_panel["infocode"].nunique()),
        "n_trade_stocks": int(trade_panel["infocode"].nunique()),
        "n_train_days": int(len(sp.usable_train_dates)),
        "n_trade_days": int(len(sp.trade_dates)),
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _run_model_subprocess(model_name, script, model_dir):
    """Run a model training script in an isolated subprocess."""
    t1 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in proc.stdout:
        print(f"      {line}", end="", flush=True)
    proc.wait()
    elapsed = time.time() - t1
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        print(f"    {model_name} STDERR: {stderr}")
        raise RuntimeError(f"{model_name} subprocess failed (exit {proc.returncode})")
    print(f"    {model_name}: {elapsed:.1f}s -> {model_dir.name}/")
    return elapsed


def run_period(
    period_id, sp, features, labels, eligible, run_rf=True, run_xgb=True, run_dnn=True
):
    """Train, predict, and save Datastream Phase 2 models for one study period."""
    t0 = time.time()
    model_dir = MODELS / f"period_{period_id:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dates = set(pd.to_datetime(sp.usable_train_dates))
    trade_dates = set(pd.to_datetime(sp.trade_dates))

    train_panel = build_panel(features, labels, eligible, train_dates)
    trade_panel = build_panel(features, labels, eligible, trade_dates)

    print(
        f"  Train: {len(train_panel):,} obs, "
        f"{train_panel['infocode'].nunique()} stocks, "
        f"{len(train_dates)} days"
    )
    print(
        f"  Trade: {len(trade_panel):,} obs, "
        f"{trade_panel['infocode'].nunique()} stocks, "
        f"{len(trade_dates)} days"
    )

    save_period_meta(model_dir, period_id, sp, train_panel, trade_panel)

    result = trade_panel[["date", "infocode"]].copy()
    result["period_id"] = period_id

    train_panel.to_parquet(model_dir / "_train_panel.parquet", index=False)
    trade_panel.to_parquet(model_dir / "_trade_panel.parquet", index=False)

    src_path = str(ROOT / "src")

    if run_rf:
        rf_cls_path = model_dir / "rf_cls.pkl"
        rf_reg_path = model_dir / "rf_reg.pkl"
        if rf_cls_path.exists() and rf_reg_path.exists():
            _run_model_subprocess(
                "RF-ext (cached)",
                f"""
import sys; sys.path.insert(0, {src_path!r})
import joblib, pandas as pd
from krauss.models.rf_extension import predict_rf_extension
tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
clf = joblib.load({str(rf_cls_path)!r})
reg = joblib.load({str(rf_reg_path)!r})
p_trade, u_trade = predict_rf_extension(clf, reg, tdp)
p_train, u_train = predict_rf_extension(clf, reg, tp)
pd.DataFrame({{'p': p_trade, 'u': u_trade}}).to_parquet({str(model_dir / '_rf_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train, 'u': u_train}}).to_parquet({str(model_dir / '_rf_train.parquet')!r}, index=False)
""",
                model_dir,
            )
        else:
            _run_model_subprocess(
                "RF-ext",
                f"""
import sys; sys.path.insert(0, {src_path!r})
import joblib, pandas as pd
from krauss.models.rf_extension import (
    build_rf_classifier, build_rf_regressor, train_rf_extension, predict_rf_extension
)
tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
clf = build_rf_classifier()
reg = build_rf_regressor()
clf, reg = train_rf_extension(clf, reg, tp, tp['y_binary'], tp['u_excess'])
p_trade, u_trade = predict_rf_extension(clf, reg, tdp)
p_train, u_train = predict_rf_extension(clf, reg, tp)
pd.DataFrame({{'p': p_trade, 'u': u_trade}}).to_parquet({str(model_dir / '_rf_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train, 'u': u_train}}).to_parquet({str(model_dir / '_rf_train.parquet')!r}, index=False)
joblib.dump(clf, {str(rf_cls_path)!r})
joblib.dump(reg, {str(rf_reg_path)!r})
""",
                model_dir,
            )

        rf_trade = pd.read_parquet(model_dir / "_rf_trade.parquet")
        result["p_rf"] = rf_trade["p"].values
        result["u_rf"] = rf_trade["u"].values

    if run_xgb:
        xgb_cls_path = model_dir / "xgb_cls.json"
        xgb_reg_path = model_dir / "xgb_reg.json"
        if xgb_cls_path.exists() and xgb_reg_path.exists():
            _run_model_subprocess(
                "XGB-ext (cached)",
                f"""
import sys; sys.path.insert(0, {src_path!r})
import pandas as pd, xgboost as xgb
tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
clf = xgb.XGBClassifier()
clf.load_model({str(xgb_cls_path)!r})
reg = xgb.XGBRegressor()
reg.load_model({str(xgb_reg_path)!r})
p_trade = clf.predict_proba(tdp[{FEATURE_COLS!r}])[:, 1]
u_trade = reg.predict(tdp[{FEATURE_COLS!r}])
p_train = clf.predict_proba(tp[{FEATURE_COLS!r}])[:, 1]
u_train = reg.predict(tp[{FEATURE_COLS!r}])
pd.DataFrame({{'p': p_trade, 'u': u_trade}}).to_parquet({str(model_dir / '_xgb_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train, 'u': u_train}}).to_parquet({str(model_dir / '_xgb_train.parquet')!r}, index=False)
""",
                model_dir,
            )
        else:
            _run_model_subprocess(
                "XGB-ext",
                f"""
import sys; sys.path.insert(0, {src_path!r})
import pandas as pd
from krauss.models.xgb_extension import (
    build_xgb_classifier, build_xgb_regressor, train_xgb_extension, predict_xgb_extension
)
tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
clf = build_xgb_classifier()
reg = build_xgb_regressor()
clf, reg = train_xgb_extension(clf, reg, tp, tp['y_binary'], tp['u_excess'])
p_trade, u_trade = predict_xgb_extension(clf, reg, tdp)
p_train, u_train = predict_xgb_extension(clf, reg, tp)
pd.DataFrame({{'p': p_trade, 'u': u_trade}}).to_parquet({str(model_dir / '_xgb_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train, 'u': u_train}}).to_parquet({str(model_dir / '_xgb_train.parquet')!r}, index=False)
clf.save_model({str(xgb_cls_path)!r})
reg.save_model({str(xgb_reg_path)!r})
""",
                model_dir,
            )

        xgb_trade = pd.read_parquet(model_dir / "_xgb_trade.parquet")
        result["p_xgb"] = xgb_trade["p"].values
        result["u_xgb"] = xgb_trade["u"].values

    if run_dnn:
        _run_model_subprocess(
            "MT-DNN",
            f"""
import sys; sys.path.insert(0, {src_path!r})
print('MT-DNN subprocess started', flush=True)
import pandas as pd, torch
from krauss.models.dnn_multitask import (
    build_multitask_dnn, train_multitask_dnn, predict_multitask_dnn
)
tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
print(f'MT-DNN data loaded: train={{len(tp)}}, trade={{len(tdp)}}', flush=True)
dnn = build_multitask_dnn()
print('MT-DNN model built, starting training...', flush=True)
dnn = train_multitask_dnn(dnn, tp, tp['y_binary'], tp['u_excess'])
p_trade, u_trade = predict_multitask_dnn(dnn, tdp)
p_train, u_train = predict_multitask_dnn(dnn, tp)
pd.DataFrame({{'p': p_trade, 'u': u_trade}}).to_parquet({str(model_dir / '_dnn_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train, 'u': u_train}}).to_parquet({str(model_dir / '_dnn_train.parquet')!r}, index=False)
torch.save(dnn.state_dict(), {str(model_dir / 'mt_dnn.pt')!r})
print('MT-DNN done', flush=True)
""",
            model_dir,
        )

        dnn_trade = pd.read_parquet(model_dir / "_dnn_trade.parquet")
        result["p_dnn"] = dnn_trade["p"].values
        result["u_dnn"] = dnn_trade["u"].values

    for f in [model_dir / "_train_panel.parquet", model_dir / "_trade_panel.parquet"]:
        if f.exists():
            f.unlink()

    if run_rf and run_xgb and run_dnn:
        result["p_ens1"] = ens1_p_hat(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values
        )
        result["u_ens1"] = ens1_u_hat(
            result["u_dnn"].values, result["u_xgb"].values, result["u_rf"].values
        )

        for prefix in ["dnn", "xgb", "rf", "ens1"]:
            scores = compute_score_families(
                result[f"p_{prefix}"].values,
                result[f"u_{prefix}"].values,
            )
            result[f"score_p_{prefix}"] = scores["p_only"]
            result[f"score_u_{prefix}"] = scores["u_only"]
            result[f"score_comp_{prefix}"] = scores["composite"]

        print("    ENS1 + score families computed")

    result.to_parquet(model_dir / "predictions.parquet", index=False)
    print(f"    Predictions -> {model_dir.name}/predictions.parquet")
    print(f"  Period {period_id} total: {time.time()-t0:.1f}s\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="all", choices=["rf", "xgb", "dnn", "all"]
    )
    parser.add_argument("--periods", type=int, nargs="*", default=None)
    args = parser.parse_args()

    run_rf = args.model in ("rf", "all")
    run_xgb = args.model in ("xgb", "all")
    run_dnn = args.model in ("dnn", "all")

    print("=" * 60)
    print("PHASE 2 — Datastream Extension Model Training & Prediction")
    print("=" * 60)
    print(f"Models: RF={run_rf}, XGB={run_xgb}, MT-DNN={run_dnn}\n")

    print("Loading Datastream US-only data...")
    features, labels, returns, eligible = load_data()
    print(f"  Features: {len(features):,}")
    print(f"  Labels:   {len(labels):,}")

    trading_dates = np.sort(pd.to_datetime(returns["date"]).unique())
    periods = build_study_periods(trading_dates)
    print(f"\n{len(periods)} study periods")

    if args.periods is not None:
        period_ids = args.periods
    else:
        period_ids = list(range(23, len(periods)))

    print(f"Running periods: {period_ids}\n")

    all_results = []
    out_path = DS_DIR / "predictions_phase2_ds.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        keep = existing[existing["period_id"].isin(period_ids)]
        if len(keep):
            for pid in sorted(keep["period_id"].unique()):
                all_results.append(keep[keep["period_id"] == pid].copy())
                print(f"  Period {pid}: loaded from consolidated cache")

    for pid in period_ids:
        pred_path = MODELS / f"period_{pid:02d}" / "predictions.parquet"
        if pred_path.exists() and pid not in {
            r["period_id"].iloc[0] for r in all_results
        }:
            all_results.append(pd.read_parquet(pred_path))
            print(f"  Period {pid}: loaded from period cache")

    completed = {int(r["period_id"].iloc[0]) for r in all_results if len(r)}
    remaining = [pid for pid in period_ids if pid not in completed]

    if remaining:
        print(f"  {len(completed)} cached, {len(remaining)} remaining\n")
    else:
        print(f"  All {len(completed)} periods cached, nothing to run\n")

    from tqdm import tqdm

    for pid in tqdm(remaining, desc="Study periods", unit="period"):
        sp = periods[pid]
        print(
            f"\n--- Period {pid}: trade {sp.trade_start.date()} "
            f"to {sp.trade_end.date()} ---"
        )
        result = run_period(
            pid,
            sp,
            features,
            labels,
            eligible,
            run_rf=run_rf,
            run_xgb=run_xgb,
            run_dnn=run_dnn,
        )
        all_results.append(result)
        pd.concat(all_results, ignore_index=True).to_parquet(out_path, index=False)

    predictions = pd.concat(all_results, ignore_index=True)
    predictions.to_parquet(out_path, index=False)

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nPredictions: {len(predictions):,} rows")
    print(f"Saved -> {out_path.relative_to(ROOT)}")

    for col in ["p_rf", "p_xgb", "p_dnn", "p_ens1"]:
        if col in predictions.columns:
            print(f"  {col} mean: {predictions[col].mean():.4f}")
    for col in ["u_rf", "u_xgb", "u_dnn", "u_ens1"]:
        if col in predictions.columns:
            print(
                f"  {col} mean: {predictions[col].mean():.6f}, "
                f"std: {predictions[col].std():.6f}"
            )


if __name__ == "__main__":
    main()
