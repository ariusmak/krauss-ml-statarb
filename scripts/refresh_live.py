"""
Daily refresh orchestrator for the live-update path.

Walks the loop:

    1. Refresh ``data/processed/returns_unified.parquet`` so it covers
       through today (append-only yfinance tail update).
    2. Identify dates that exist in returns_unified but NOT in
       predictions_unified.parquet (the "live tail" we need to predict).
    3. Compute the 31 lagged-return features for the live tail using
       ``krauss.data.features_live``.
    4. Run the frozen period-33 models on the feature panel.  Each model
       family runs in its OWN subprocess because sklearn (loky), xgboost
       (libomp) and torch (libomp) are not fork-safe together on macOS.
    5. Combine into ENS1 P̂ / Û and the three score families.
    6. Append the new rows to ``predictions_unified.parquet``.

Idempotent.  Re-running on a day where there's no new data is a no-op.

Run::

    python scripts/refresh_live.py
    python scripts/refresh_live.py --rebuild-returns      # also append live returns
    python scripts/refresh_live.py --period 33            # use a non-default period
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from krauss.data.features_live import (  # noqa: E402
    compute_features,
    load_unified_returns,
)
from krauss.models.ensembles_phase2 import (  # noqa: E402
    compute_score_families,
    ens1_p_hat,
    ens1_u_hat,
)

PROCESSED = ROOT / "data" / "processed"
MODELS_P2_DS = ROOT / "data" / "models_p2_ds"
PREDS_UNIFIED = PROCESSED / "predictions_unified.parquet"

DEFAULT_PERIOD_ID = 33
SOURCE_TAG = "phase2_ds_live"


def rebuild_returns_unified() -> None:
    print("[1/6] Appending live returns via build_returns_unified.py ...")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_returns_unified.py"),
            "--append-live",
        ],
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("build_returns_unified.py --append-live failed")


def _run_model_subprocess(name: str, script: str) -> None:
    """Run a model-inference snippet in an isolated subprocess.

    The snippet must not return anything; it writes a parquet to disk and
    we read it back.  This mirrors the libomp-isolation pattern in
    ``run_phase2_datastream.py``.
    """
    print(f"  [{name}] starting subprocess ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for line in proc.stdout:
        print(f"    {name}: {line}", end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        print(f"  [{name}] STDERR:\n{stderr}", flush=True)
        raise RuntimeError(f"{name} subprocess exited {proc.returncode}")


def predict_live_isolated(features: pd.DataFrame, period_id: int) -> pd.DataFrame:
    """Run RF, XGB, MT-DNN inference each in its own subprocess.

    Writes feature panel + per-model output parquets to a temp directory,
    spawns three subprocesses (one per model family), and merges the
    P̂ / Û columns back into a single dataframe.
    """
    pdir = MODELS_P2_DS / f"period_{period_id:02d}"
    if not pdir.exists():
        raise FileNotFoundError(f"Period directory not found: {pdir}")

    src_path = str(ROOT / "src")

    with tempfile.TemporaryDirectory(prefix="refresh_live_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        feat_path = tmpdir / "features.parquet"
        # Persist features (with infocode / date) so subprocesses can re-read.
        features.to_parquet(feat_path, index=False)

        rf_out = tmpdir / "rf_out.parquet"
        xgb_out = tmpdir / "xgb_out.parquet"
        dnn_out = tmpdir / "dnn_out.parquet"

        rf_cls_path = pdir / "rf_cls.pkl"
        rf_reg_path = pdir / "rf_reg.pkl"
        xgb_cls_path = pdir / "xgb_cls.json"
        xgb_reg_path = pdir / "xgb_reg.json"
        dnn_path = pdir / "mt_dnn.pt"

        for p in [rf_cls_path, rf_reg_path, xgb_cls_path, xgb_reg_path, dnn_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing model artifact: {p}")

        # ---- RF ---------------------------------------------------------
        _run_model_subprocess("RF", f"""
import sys; sys.path.insert(0, {src_path!r})
import joblib, pandas as pd
from krauss.models.rf_extension import predict_rf_extension
feat = pd.read_parquet({str(feat_path)!r})
clf = joblib.load({str(rf_cls_path)!r})
reg = joblib.load({str(rf_reg_path)!r})
p, u = predict_rf_extension(clf, reg, feat)
pd.DataFrame({{'p_rf': p, 'u_rf': u}}).to_parquet({str(rf_out)!r}, index=False)
print('done.')
""")

        # ---- XGB --------------------------------------------------------
        _run_model_subprocess("XGB", f"""
import sys; sys.path.insert(0, {src_path!r})
import pandas as pd, xgboost as xgb
from krauss.models.xgb_extension import predict_xgb_extension
feat = pd.read_parquet({str(feat_path)!r})
clf = xgb.XGBClassifier(); clf.load_model({str(xgb_cls_path)!r})
reg = xgb.XGBRegressor();  reg.load_model({str(xgb_reg_path)!r})
p, u = predict_xgb_extension(clf, reg, feat)
pd.DataFrame({{'p_xgb': p, 'u_xgb': u}}).to_parquet({str(xgb_out)!r}, index=False)
print('done.')
""")

        # ---- Multitask DNN ----------------------------------------------
        _run_model_subprocess("MT-DNN", f"""
import sys; sys.path.insert(0, {src_path!r})
import pandas as pd, torch
from krauss.models.dnn_multitask import build_multitask_dnn, predict_multitask_dnn
feat = pd.read_parquet({str(feat_path)!r})
dnn = build_multitask_dnn()
dnn.load_state_dict(torch.load({str(dnn_path)!r}, weights_only=True))
dnn.eval()
p, u = predict_multitask_dnn(dnn, feat)
pd.DataFrame({{'p_dnn': p, 'u_dnn': u}}).to_parquet({str(dnn_out)!r}, index=False)
print('done.')
""")

        rf_df = pd.read_parquet(rf_out)
        xgb_df = pd.read_parquet(xgb_out)
        dnn_df = pd.read_parquet(dnn_out)

    # Stitch on row order (subprocess output is in the same order as features).
    if not (len(rf_df) == len(xgb_df) == len(dnn_df) == len(features)):
        raise RuntimeError(
            f"Row count mismatch: RF={len(rf_df)} XGB={len(xgb_df)} "
            f"DNN={len(dnn_df)} features={len(features)}"
        )

    out = features[["date", "infocode"]].copy().reset_index(drop=True)
    out["period_id"] = int(period_id)
    out["p_rf"] = rf_df["p_rf"].values
    out["u_rf"] = rf_df["u_rf"].values
    out["p_xgb"] = xgb_df["p_xgb"].values
    out["u_xgb"] = xgb_df["u_xgb"].values
    out["p_dnn"] = dnn_df["p_dnn"].values
    out["u_dnn"] = dnn_df["u_dnn"].values

    out["p_ens1"] = ens1_p_hat(
        out["p_dnn"].values, out["p_xgb"].values, out["p_rf"].values
    )
    out["u_ens1"] = ens1_u_hat(
        out["u_dnn"].values, out["u_xgb"].values, out["u_rf"].values
    )

    for prefix in ("dnn", "xgb", "rf", "ens1"):
        scores = compute_score_families(
            out[f"p_{prefix}"].values, out[f"u_{prefix}"].values
        )
        out[f"score_p_{prefix}"] = scores["p_only"]
        out[f"score_u_{prefix}"] = scores["u_only"]
        out[f"score_comp_{prefix}"] = scores["composite"]

    out["source"] = SOURCE_TAG
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-returns", action="store_true",
                        help="Append live yfinance rows before prediction.")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD_ID,
                        help="Period id whose frozen models to use (default 33).")
    args = parser.parse_args()

    if args.rebuild_returns:
        rebuild_returns_unified()
    else:
        print("[1/6] Skipping returns rebuild (use --rebuild-returns to refresh).")

    print("\n[2/6] Loading unified returns and predictions ...")
    returns = load_unified_returns()
    print(f"  returns_unified: {len(returns):,} rows, "
          f"{returns['date'].min().date()} -> {returns['date'].max().date()}")

    if not PREDS_UNIFIED.exists():
        raise FileNotFoundError(
            f"{PREDS_UNIFIED} not found. "
            "Run scripts/build_predictions_unified.py first."
        )
    preds = pd.read_parquet(PREDS_UNIFIED)
    preds["date"] = pd.to_datetime(preds["date"])
    last_pred_date = preds["date"].max()
    print(f"  predictions_unified: {len(preds):,} rows, "
          f"last date: {last_pred_date.date()}")

    last_return_date = returns["date"].max()
    if last_return_date <= last_pred_date:
        print(f"\nNo new days to predict (returns end {last_return_date.date()}, "
              f"predictions end {last_pred_date.date()}). Done.")
        return

    print(f"\n[3/6] Computing features for live tail "
          f"{(last_pred_date + pd.Timedelta(days=1)).date()} -> "
          f"{last_return_date.date()} ...")
    features = compute_features(
        returns,
        start=last_pred_date + pd.Timedelta(days=1),
        end=last_return_date,
    )
    print(f"  Feature rows: {len(features):,}, "
          f"{features['date'].nunique()} dates, "
          f"{features['infocode'].nunique()} infocodes")
    if len(features) == 0:
        print("\n  No feature rows produced (insufficient history). Done.")
        return

    print(
        "\n[4/6] Running predictions in isolated subprocesses "
        f"(period {args.period}) ..."
    )
    new_preds = predict_live_isolated(features, args.period)

    print("\n[5/6] Attaching ticker (display only) ...")
    ticker_map = (preds[["infocode", "ticker"]]
                  .dropna(subset=["ticker"])
                  .drop_duplicates(subset="infocode", keep="last"))
    new_preds = new_preds.merge(ticker_map, on="infocode", how="left")
    print(f"  Predicted {len(new_preds):,} rows "
          f"(ticker mapped: {new_preds['ticker'].notna().sum():,})")

    print(f"\n[6/6] Appending to {PREDS_UNIFIED.relative_to(ROOT)} ...")
    combined = pd.concat([preds, new_preds], ignore_index=True, sort=False)
    combined = combined.sort_values(["date", "infocode"]).reset_index(drop=True)
    combined.to_parquet(PREDS_UNIFIED, index=False)
    print(f"  Wrote {len(combined):,} rows total, "
          f"date range: {combined['date'].min().date()} -> "
          f"{combined['date'].max().date()}")
    for src, cnt in combined["source"].value_counts().items():
        sub = combined[combined["source"] == src]
        print(f"    {src:18s}: {cnt:>10,} rows, "
              f"{sub['date'].min().date()} -> {sub['date'].max().date()}")


if __name__ == "__main__":
    main()
