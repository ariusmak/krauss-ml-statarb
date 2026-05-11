"""
Consolidate the existing prediction parquets into a single source-of-truth
panel keyed by ``(date, infocode)``::

    data/datastream/predictions_ds_h2o.parquet        (1992-12-17 -> 2015-10-15)
    data/datastream/predictions_phase2_ds.parquet     (2015-10-16 -> 2026-04-21)

Output:
    data/processed/predictions_unified.parquet

Identifier convention: ``infocode`` is the canonical join key.  ``ticker`` is
attached for display only via ``data/datastream/ds_names.parquet`` and may
be NaN for a handful of historical infocodes that never had a ticker
recorded.

The CRSP-keyed PyTorch parquets (``predictions_phase1.parquet``,
``predictions_phase2.parquet``) are intentionally excluded -- they live on a
different ever-member universe and would conflict with the Datastream-keyed
panel.  They remain as side artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DS_DIR = ROOT / "data" / "datastream"
OUT_PATH = PROCESSED / "predictions_unified.parquet"


def load_phase1_h2o() -> pd.DataFrame:
    """Phase 1 H2O paper-parity predictions (DNN/GBT/RAF/ENS1/ENS2/ENS3).

    Uses the Datastream-keyed file ``data/datastream/predictions_ds_h2o.parquet``.
    Note: the column name is ``permno`` for shared-engine compatibility, but
    the values are Datastream infocodes -- per the H2O training pipeline that
    fed off ``ds_daily_returns_usonly.parquet`` and renamed at load time.
    The CRSP-keyed sibling ``data/processed/predictions_phase1_h2o.parquet``
    is intentionally NOT used (different ever-member universe).
    """
    df = pd.read_parquet(DS_DIR / "predictions_ds_h2o.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if "infocode" not in df.columns and "permno" in df.columns:
        df = df.rename(columns={"permno": "infocode"})
    df["infocode"] = df["infocode"].astype("Int64")
    df["source"] = "phase1_h2o"
    return df


def load_phase2_ds() -> pd.DataFrame:
    """Phase 2 Datastream extension predictions (periods 23-33, with Û + scores)."""
    df = pd.read_parquet(DS_DIR / "predictions_phase2_ds.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["infocode"] = df["infocode"].astype("Int64")
    df["source"] = "phase2_ds"
    return df


def load_ticker_map() -> pd.DataFrame:
    """One ticker per infocode, latest non-null spelling."""
    nm = pd.read_parquet(DS_DIR / "ds_names.parquet")
    nm = nm.dropna(subset=["ticker"]).drop_duplicates(
        subset="infocode", keep="last"
    )[["infocode", "ticker"]]
    nm["infocode"] = nm["infocode"].astype("Int64")
    return nm


def _load_existing_live_tail(base_max_date: pd.Timestamp) -> pd.DataFrame:
    """Preserve existing live-refresh rows beyond the rebuilt base panel."""
    if not OUT_PATH.exists():
        return pd.DataFrame()

    existing = pd.read_parquet(OUT_PATH)
    if "source" not in existing.columns:
        return pd.DataFrame()

    existing["date"] = pd.to_datetime(existing["date"])
    live = existing[
        (existing["source"] == "phase2_ds_live")
        & (existing["date"] > base_max_date)
    ].copy()
    if "infocode" in live.columns:
        live["infocode"] = live["infocode"].astype("Int64")
    return live


def main(*, preserve_live_tail: bool = True) -> None:
    print("Loading Phase 1 H2O ...")
    p1 = load_phase1_h2o()
    print(
        f"  {len(p1):,} rows, {p1['date'].min().date()} -> "
        f"{p1['date'].max().date()}, {p1['infocode'].nunique():,} infocodes"
    )
    print(f"  columns: {sorted(p1.columns.tolist())}")

    print("\nLoading Phase 2 Datastream ...")
    p2 = load_phase2_ds()
    print(
        f"  {len(p2):,} rows, {p2['date'].min().date()} -> "
        f"{p2['date'].max().date()}, {p2['infocode'].nunique():,} infocodes"
    )
    print(f"  columns: {sorted(p2.columns.tolist())}")

    # Sanity: no overlap on (date, infocode) across the two sources
    p1_keys = pd.MultiIndex.from_frame(p1[["date", "infocode"]])
    p2_keys = pd.MultiIndex.from_frame(p2[["date", "infocode"]])
    overlap = p1_keys.intersection(p2_keys)
    if len(overlap) > 0:
        raise ValueError(
            f"Overlapping (date, infocode) keys across sources: {len(overlap)}"
        )

    # Stack the rebuilt historical base.
    unified = pd.concat([p1, p2], ignore_index=True, sort=False)

    if preserve_live_tail:
        live_tail = _load_existing_live_tail(unified["date"].max())
        if len(live_tail):
            print(
                f"\nPreserving existing live tail: {len(live_tail):,} rows, "
                f"{live_tail['date'].min().date()} -> "
                f"{live_tail['date'].max().date()}"
            )
            unified = pd.concat([unified, live_tail], ignore_index=True, sort=False)

    # Attach ticker (display only)
    tickers = load_ticker_map()
    if "ticker" in unified.columns:
        unified = unified.drop(columns=["ticker"])
    unified = unified.merge(tickers, on="infocode", how="left")

    dupes = unified.duplicated(["date", "infocode"]).sum()
    if dupes:
        raise ValueError(f"Unified panel contains {dupes:,} duplicate keys")

    # Sort and write
    unified = unified.sort_values(["date", "infocode"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUT_PATH, index=False)

    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  Total rows: {len(unified):,}")
    print(
        f"  Date range: {unified['date'].min().date()} -> "
        f"{unified['date'].max().date()}"
    )
    print(f"  Unique infocodes: {unified['infocode'].nunique():,}")
    ticker_count = unified["ticker"].notna().sum()
    print(
        f"  Tickers attached: {ticker_count:,} / {len(unified):,} "
        f"({100 * ticker_count / len(unified):.1f}%)"
    )
    for src, cnt in unified["source"].value_counts().items():
        sub = unified[unified["source"] == src]
        print(f"    {src:10s}: {cnt:>10,} rows, "
              f"{sub['date'].min().date()} -> {sub['date'].max().date()}")
    print()
    print("All columns:")
    for c in unified.columns:
        non_null = unified[c].notna().sum()
        pct = 100 * non_null / len(unified)
        print(f"  {c:24s} non-null: {non_null:>10,} ({pct:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-live-tail",
        action="store_true",
        help="Rebuild only from historical source parquets, dropping live rows.",
    )
    args = parser.parse_args()
    main(preserve_live_tail=not args.reset_live_tail)
