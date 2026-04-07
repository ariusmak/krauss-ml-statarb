"""
Build features and labels from processed data.

Usage:
    python scripts/build_features_labels.py

Reads:
    data/processed/daily_returns.parquet
    data/processed/universe_daily.parquet

Produces:
    data/processed/features.parquet      — 31 lagged return features
    data/processed/labels.parquet        — y_binary + u_excess targets

Features (31 total):
    R1..R20  — simple returns over 1..20 day lookbacks
    R40, R60, ..., R240 — multi-period simple returns

Labels:
    y_binary  — 1 if next-day return > next-day cross-sectional median, else 0
    u_excess  — next-day return minus next-day cross-sectional median return
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.features import compute_lagged_returns, ALL_LAGS
from krauss.data.labels import compute_labels

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("FEATURE & LABEL CONSTRUCTION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load processed data
    # ------------------------------------------------------------------
    print("\nLoading processed data...")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    eligible = pd.read_parquet(PROCESSED / "universe_daily.parquet")
    print(f"  Returns:     {len(returns):,} rows, "
          f"{returns['permno'].nunique():,} stocks")
    print(f"  Eligibility: {len(eligible):,} stock-day rows")

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    print(f"\nComputing {len(ALL_LAGS)} lagged return features...")
    print("  (This may take a few minutes — building price index "
          "and shifting per stock)")
    features = compute_lagged_returns(returns)
    print(f"  {len(features):,} rows with complete features "
          f"({features['permno'].nunique():,} stocks)")

    # Date range after dropping incomplete lookbacks
    print(f"  Date range: {features['date'].min().date()} "
          f"to {features['date'].max().date()}")

    features.to_parquet(PROCESSED / "features.parquet", index=False)
    print(f"  Saved -> data/processed/features.parquet")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    print("\nComputing labels (y_binary + u_excess)...")
    labels = compute_labels(returns, eligible)
    print(f"  {len(labels):,} labeled stock-day rows "
          f"({labels['permno'].nunique():,} stocks)")
    print(f"  Date range: {labels['date'].min().date()} "
          f"to {labels['date'].max().date()}")

    # Sanity checks
    print(f"\n  Label distribution:")
    print(f"    y_binary=1: {labels['y_binary'].mean():.4f} "
          f"(should be ~0.50)")
    print(f"    u_excess mean:   {labels['u_excess'].mean():.6f} "
          f"(should be ~0)")
    print(f"    u_excess median: {labels['u_excess'].median():.6f}")
    print(f"    u_excess std:    {labels['u_excess'].std():.6f}")

    labels.to_parquet(PROCESSED / "labels.parquet", index=False)
    print(f"  Saved -> data/processed/labels.parquet")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("DONE")
    print("=" * 60)
    for f in ["features.parquet", "labels.parquet"]:
        p = PROCESSED / f
        size_mb = p.stat().st_size / 1e6
        print(f"  data/processed/{f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
