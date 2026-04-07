"""
Run Phase 1 reproduction: train RF, XGB, DNN across all 23 study periods,
generate predictions, compute ensembles, and save all outputs.

Usage:
    python scripts/run_phase1.py
    python scripts/run_phase1.py --model rf         # RF only
    python scripts/run_phase1.py --model xgb        # XGB only
    python scripts/run_phase1.py --model dnn        # DNN only
    python scripts/run_phase1.py --periods 0 1 2    # specific periods only

Outputs:
    data/models/period_{id}/rf.pkl        — trained RF per period
    data/models/period_{id}/xgb.json      — trained XGB per period
    data/models/period_{id}/dnn.pt        — trained DNN per period
    data/models/period_{id}/meta.json     — period metadata
    data/processed/predictions_phase1.parquet — all predictions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods, study_periods_summary
from krauss.data.features import FEATURE_COLS
from krauss.models.rf_phase1 import build_rf_model, train_rf, predict_rf
from krauss.models.xgb_phase1 import build_xgb_model, train_xgb, predict_xgb
from krauss.models.dnn_phase1 import build_dnn_model, train_dnn, predict_dnn
from krauss.models.ensembles_phase1 import (
    ens1_predictions, ens2_predictions, ens3_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"


def load_data():
    """Load features, labels, returns, eligibility."""
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    eligible = pd.read_parquet(PROCESSED / "universe_daily.parquet")
    return features, labels, returns, eligible


def build_panel(features, labels, eligible, dates_set):
    """
    Build the modeling panel for a set of dates:
    inner join of features + labels + eligibility.
    """
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][
        ["date", "permno", "y_binary"]
    ].copy()

    panel = feat.merge(lab, on=["date", "permno"], how="inner")
    panel = panel.merge(eligible, on=["date", "permno"], how="inner")

    # Drop any rows with NaN features
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def save_period_meta(model_dir, period_id, sp, train_panel, trade_panel):
    """Save period metadata as JSON."""
    meta = {
        "period_id": period_id,
        "train_start": str(sp.train_start.date()),
        "train_end": str(sp.train_end.date()),
        "usable_train_start": str(sp.usable_train_start.date()),
        "trade_start": str(sp.trade_start.date()),
        "trade_end": str(sp.trade_end.date()),
        "n_train_obs": len(train_panel),
        "n_trade_obs": len(trade_panel),
        "n_train_stocks": int(train_panel["permno"].nunique()),
        "n_trade_stocks": int(trade_panel["permno"].nunique()),
        "n_train_days": int(len(sp.usable_train_dates)),
        "n_trade_days": int(len(sp.trade_dates)),
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_period(
    period_id, sp, features, labels, eligible,
    run_rf=True, run_xgb=True, run_dnn=True,
):
    """Train, predict, and save models for one study period."""
    t0 = time.time()

    # Model output directory
    model_dir = MODELS / f"period_{period_id:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build train and trade panels
    train_dates = set(pd.to_datetime(sp.usable_train_dates))
    trade_dates = set(pd.to_datetime(sp.trade_dates))

    train_panel = build_panel(features, labels, eligible, train_dates)
    trade_panel = build_panel(features, labels, eligible, trade_dates)

    y_train = train_panel["y_binary"]

    print(f"  Train: {len(train_panel):,} obs, "
          f"{train_panel['permno'].nunique()} stocks, "
          f"{len(train_dates)} days")
    print(f"  Trade: {len(trade_panel):,} obs, "
          f"{trade_panel['permno'].nunique()} stocks, "
          f"{len(trade_dates)} days")

    # Save period metadata
    save_period_meta(model_dir, period_id, sp, train_panel, trade_panel)

    # Initialize prediction columns
    result = trade_panel[["date", "permno"]].copy()
    result["period_id"] = period_id

    # Train/predict/save each model sequentially
    p_rf_train = p_xgb_train = p_dnn_train = None

    if run_rf:
        t1 = time.time()
        rf = build_rf_model()
        rf = train_rf(rf, train_panel, y_train)
        result["p_rf"] = predict_rf(rf, trade_panel)
        p_rf_train = predict_rf(rf, train_panel)
        joblib.dump(rf, model_dir / "rf.pkl")
        print(f"    RF:  {time.time()-t1:.1f}s -> {model_dir.name}/rf.pkl")

    if run_xgb:
        t1 = time.time()
        xgb_model = build_xgb_model()
        xgb_model = train_xgb(xgb_model, train_panel, y_train)
        result["p_xgb"] = predict_xgb(xgb_model, trade_panel)
        p_xgb_train = predict_xgb(xgb_model, train_panel)
        xgb_model.save_model(str(model_dir / "xgb.json"))
        print(f"    XGB: {time.time()-t1:.1f}s -> {model_dir.name}/xgb.json")

    if run_dnn:
        t1 = time.time()
        dnn = build_dnn_model()
        dnn = train_dnn(dnn, train_panel, y_train)
        result["p_dnn"] = predict_dnn(dnn, trade_panel)
        p_dnn_train = predict_dnn(dnn, train_panel)
        torch.save(dnn.state_dict(), model_dir / "dnn.pt")
        print(f"    DNN: {time.time()-t1:.1f}s -> {model_dir.name}/dnn.pt")

    # ---- Ensembles ----
    if run_rf and run_xgb and run_dnn:
        result["p_ens1"] = ens1_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values
        )
        result["p_ens2"] = ens2_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values,
            y_train.values, p_dnn_train, p_xgb_train, p_rf_train,
        )
        result["p_ens3"] = ens3_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values,
            y_train.values, p_dnn_train, p_xgb_train, p_rf_train,
        )
        # Save ensemble weights for reproducibility
        from sklearn.metrics import roc_auc_score
        ens_meta = {}
        for name, p_tr in [("rf", p_rf_train), ("xgb", p_xgb_train), ("dnn", p_dnn_train)]:
            auc = roc_auc_score(y_train.values, p_tr)
            gini = 2 * auc - 1
            ens_meta[name] = {"auc": round(auc, 6), "gini": round(gini, 6)}
        with open(model_dir / "ensemble_meta.json", "w") as f:
            json.dump(ens_meta, f, indent=2)
        print(f"    ENS: computed -> {model_dir.name}/ensemble_meta.json")

    # Save per-period predictions
    result.to_parquet(model_dir / "predictions.parquet", index=False)
    print(f"    Predictions -> {model_dir.name}/predictions.parquet")

    print(f"  Period {period_id} total: {time.time()-t0:.1f}s\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["rf", "xgb", "dnn", "all"])
    parser.add_argument("--periods", type=int, nargs="*", default=None)
    args = parser.parse_args()

    run_rf = args.model in ("rf", "all")
    run_xgb = args.model in ("xgb", "all")
    run_dnn = args.model in ("dnn", "all")

    print("=" * 60)
    print("PHASE 1 — Model Training & Prediction")
    print("=" * 60)
    print(f"Models: RF={run_rf}, XGB={run_xgb}, DNN={run_dnn}\n")

    # Load data
    print("Loading data...")
    features, labels, returns, eligible = load_data()
    print(f"  Features: {len(features):,}")
    print(f"  Labels:   {len(labels):,}")

    # Build study periods
    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)
    print(f"\n{len(periods)} study periods")

    if args.periods is not None:
        period_ids = args.periods
    else:
        period_ids = list(range(len(periods)))

    print(f"Running periods: {period_ids}\n")

    # Run each period
    from tqdm import tqdm
    all_results = []
    for pid in tqdm(period_ids, desc="Study periods", unit="period"):
        sp = periods[pid]
        print(f"\n--- Period {pid}: trade {sp.trade_start.date()} "
              f"to {sp.trade_end.date()} ---")
        result = run_period(
            pid, sp, features, labels, eligible,
            run_rf=run_rf, run_xgb=run_xgb, run_dnn=run_dnn,
        )
        all_results.append(result)

    # Combine and save consolidated predictions
    predictions = pd.concat(all_results, ignore_index=True)
    out_path = PROCESSED / "predictions_phase1.parquet"
    predictions.to_parquet(out_path, index=False)

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nPredictions: {len(predictions):,} rows")
    print(f"Saved -> {out_path.relative_to(ROOT)}")
    print(f"Columns: {list(predictions.columns)}")

    # Quick sanity
    for col in ["p_rf", "p_xgb", "p_dnn", "p_ens1"]:
        if col in predictions.columns:
            m = predictions[col].mean()
            print(f"  {col} mean: {m:.4f} (should be ~0.50)")

    # Summary of saved models
    print(f"\nModels saved to data/models/:")
    for pid in period_ids:
        d = MODELS / f"period_{pid:02d}"
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        print(f"  period_{pid:02d}/: {', '.join(files)}")


if __name__ == "__main__":
    main()
