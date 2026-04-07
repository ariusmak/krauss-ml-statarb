"""
Build raw data pipeline for Phase 1 reproduction.

Usage:
    python scripts/build_data.py

Pulls from WRDS:
    1. S&P 500 historical membership  -> data/raw/sp500_membership.parquet
    2. CRSP daily stock data           -> data/raw/crsp_daily.parquet
       (only for stocks that were ever S&P 500 members)
    3. CRSP delisting returns          -> data/raw/crsp_delist.parquet

Then builds:
    4. Monthly membership panel        -> data/processed/membership_monthly.parquet
    5. Daily eligibility table         -> data/processed/universe_daily.parquet
    6. Delisting-adjusted returns      -> data/processed/daily_returns.parquet

All dates driven by configs/phase1_repro.yaml.
"""

import sys
from pathlib import Path

import yaml
import pandas as pd

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.wrds_extract import (
    get_connection,
    fetch_sp500_membership,
    fetch_daily_stock_data,
    fetch_delisting_returns,
)
from krauss.data.universe import build_membership_matrix, build_daily_eligibility
from krauss.data.prices_returns import build_return_panel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "phase1_repro.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    start = cfg["data"]["raw_start_date"]
    end = cfg["data"]["raw_end_date"]

    print("=" * 60)
    print("KRAUSS DATA PIPELINE — Phase 1")
    print("=" * 60)
    print(f"Date range: {start} to {end}\n")

    # ------------------------------------------------------------------
    # Step 1: Connect to WRDS
    # ------------------------------------------------------------------
    print("[1/6] Connecting to WRDS...")
    conn = get_connection()
    print("      Connected.\n")

    # ------------------------------------------------------------------
    # Step 2: S&P 500 membership spells
    # ------------------------------------------------------------------
    print("[2/6] Fetching S&P 500 membership history...")
    sp500 = fetch_sp500_membership(conn)
    sp500.to_parquet(RAW / "sp500_membership.parquet", index=False)

    n_spells = len(sp500)
    n_permnos = sp500["permno"].nunique()
    print(f"      {n_spells} membership spells, {n_permnos} unique PERMNOs.")
    print(f"      Saved -> data/raw/sp500_membership.parquet\n")

    # Master list: all PERMNOs that were ever S&P 500 members
    ever_members = sp500["permno"].unique().tolist()
    print(f"      {len(ever_members)} stocks ever in S&P 500 — "
          f"will pull daily data only for these.\n")

    # ------------------------------------------------------------------
    # Step 3: CRSP daily stock data (only ever-members)
    # ------------------------------------------------------------------
    print(f"[3/6] Fetching CRSP daily data ({start} to {end})...")
    daily = fetch_daily_stock_data(conn, start, end)

    # Restrict to ever-members
    ever_set = set(ever_members)
    daily = daily[daily["permno"].isin(ever_set)].reset_index(drop=True)

    daily.to_parquet(RAW / "crsp_daily.parquet", index=False)
    n_rows = len(daily)
    n_stocks = daily["permno"].nunique()
    print(f"      {n_rows:,} rows, {n_stocks:,} unique PERMNOs (ever-members only).")
    print(f"      Saved -> data/raw/crsp_daily.parquet\n")

    # ------------------------------------------------------------------
    # Step 4: CRSP delisting returns
    # ------------------------------------------------------------------
    print(f"[4/6] Fetching delisting returns ({start} to {end})...")
    delist = fetch_delisting_returns(conn, start, end)
    delist = delist[delist["permno"].isin(ever_set)].reset_index(drop=True)
    delist.to_parquet(RAW / "crsp_delist.parquet", index=False)
    print(f"      {len(delist)} delisting events (ever-members only).")
    print(f"      Saved -> data/raw/crsp_delist.parquet\n")

    conn.close()
    print("      WRDS connection closed.\n")

    # ------------------------------------------------------------------
    # Step 5: Build monthly membership panel (no-lookahead)
    # ------------------------------------------------------------------
    print("[5/6] Building monthly membership panel...")
    membership = build_membership_matrix(sp500, start, end)
    membership.to_parquet(PROCESSED / "membership_monthly.parquet", index=False)

    n_months = membership["effective_month"].nunique()
    avg_per_month = len(membership) / max(n_months, 1)
    print(f"      {len(membership):,} rows over {n_months} effective months.")
    print(f"      Avg {avg_per_month:.0f} eligible stocks/month.")
    print(f"      Saved -> data/processed/membership_monthly.parquet")

    # Expand to daily eligibility
    trading_dates = pd.Series(sorted(daily["date"].unique()))
    daily_elig = build_daily_eligibility(membership, trading_dates)
    daily_elig.to_parquet(PROCESSED / "universe_daily.parquet", index=False)
    print(f"      {len(daily_elig):,} stock-day eligibility rows.")
    print(f"      Saved -> data/processed/universe_daily.parquet\n")

    # ------------------------------------------------------------------
    # Step 6: Delisting-adjusted returns
    # ------------------------------------------------------------------
    print("[6/6] Building delisting-adjusted return panel...")
    returns = build_return_panel(daily, delist)
    returns.to_parquet(PROCESSED / "daily_returns.parquet", index=False)
    print(f"      {len(returns):,} return observations.")
    print(f"      Saved -> data/processed/daily_returns.parquet\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nRaw files:")
    for f in sorted(RAW.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print("\nProcessed files:")
    for f in sorted(PROCESSED.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
