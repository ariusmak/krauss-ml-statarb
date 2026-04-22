"""
Fetch CBOE VIX daily close from FRED (VIXCLS) and save to data/raw/vix_daily.parquet.

FRED requires no authentication. Runs in a few seconds.

Schema: date (datetime64[ns]), vix (float64)
Date range: 1990-01-02 through most recent trading day.
Rows where VIX is missing (FRED marks these with '.') are dropped.

Usage
-----
From the repo root:
    python scripts/fetch_vix.py

Re-run any time to refresh the series. Overwrites existing file.
"""

from __future__ import annotations

from pathlib import Path
import io
import sys
import urllib.request

import pandas as pd


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "raw" / "vix_daily.parquet"


def fetch_vix() -> pd.DataFrame:
    """Download VIXCLS from FRED and return a clean DataFrame."""
    print(f"Fetching VIX from FRED: {FRED_URL}")
    req = urllib.request.Request(
        FRED_URL,
        headers={"User-Agent": "krauss-ml-statarb/vix-fetch"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    # FRED CSV has columns: observation_date, VIXCLS
    # Missing values come through as "." in the CSV.
    df = pd.read_csv(io.StringIO(raw), na_values=["."])
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"])
    n_raw = len(df)
    df = df.dropna(subset=["vix"]).reset_index(drop=True)
    df["vix"] = df["vix"].astype("float64")

    print(f"  {n_raw:,} raw rows, {len(df):,} rows after dropping missing")
    print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Summary: min={df['vix'].min():.2f}, mean={df['vix'].mean():.2f}, "
          f"max={df['vix'].max():.2f}")

    return df


def main() -> int:
    df = fetch_vix()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"  Saved to {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
