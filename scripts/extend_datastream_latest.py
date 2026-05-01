"""
Extend the Datastream paper-parity dataset from the current local cutoff to the
latest date available on WRDS.

This script preserves the existing methodology used elsewhere in the repo:

- S&P 500 membership from Datastream `tr_ds_equities.ds2constmth`
- Daily total return index from Datastream `tr_ds_equities.ds2primqtri`
- Monthly no-lookahead universe: month-end M governs month M+1
- US-trading-calendar filter using CRSP trading dates
- Feature/label construction on the filtered (`*_usonly`) panel

It is intentionally additive:

- refreshes the full membership spell table
- appends new RI rows from the last local raw Datastream date forward
- refreshes the CRSP trading calendar from the last local date forward
- rebuilds the derived `*_usonly` artifacts using the combined data

Usage:
    python scripts/extend_datastream_latest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import wrds
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"

SP500_CODE = 4408
RI_CHUNK_SIZE = 200

DAILY_LAGS = list(range(1, 21))
MULTI_PERIOD_LAGS = list(range(40, 241, 20))
ALL_LAGS = DAILY_LAGS + MULTI_PERIOD_LAGS
FEATURE_COLS = [f"R{m}" for m in ALL_LAGS]


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE full-day holiday calendar for standard U.S. equity trading."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def detect_wrds_username() -> str:
    """Infer WRDS username from env or ~/.pgpass before falling back."""
    env_user = os.environ.get("WRDS_USERNAME")
    if env_user:
        return env_user

    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        try:
            line = pgpass.read_text().splitlines()[0].strip()
            parts = line.split(":")
            if len(parts) >= 4 and parts[3]:
                return parts[3]
        except Exception:
            pass

    return "ariusmak"


def compute_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute 31 lagged return features from daily returns."""
    df = returns[["infocode", "date", "ret"]].copy()
    df = df.sort_values(["infocode", "date"]).reset_index(drop=True)
    df["price_idx"] = df.groupby("infocode")["ret"].transform(
        lambda r: (1 + r).cumprod()
    )
    for m in ALL_LAGS:
        lagged = df.groupby("infocode")["price_idx"].shift(m)
        df[f"R{m}"] = df["price_idx"] / lagged - 1
    df = df.dropna(subset=FEATURE_COLS)
    return df[["infocode", "date"] + FEATURE_COLS].reset_index(drop=True)


def compute_labels(returns: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Compute classification and regression labels."""
    ret = returns[["infocode", "date", "ret"]].copy()
    ret = ret.sort_values(["infocode", "date"]).reset_index(drop=True)
    ret["next_day_ret"] = ret.groupby("infocode")["ret"].shift(-1)
    ret["next_day_date"] = ret.groupby("infocode")["date"].shift(-1)
    ret = ret.dropna(subset=["next_day_ret"])
    ret["next_day_date"] = ret["next_day_date"].astype("datetime64[ns]")

    labels = eligible.merge(ret, on=["date", "infocode"], how="inner")
    eligible_next = eligible.rename(columns={"date": "next_day_date"})
    labels = labels.merge(eligible_next, on=["next_day_date", "infocode"], how="inner")

    median_by_day = (
        labels.groupby("next_day_date")["next_day_ret"]
        .median()
        .rename("next_day_median")
    )
    labels = labels.merge(median_by_day, on="next_day_date", how="left")
    labels["u_excess"] = labels["next_day_ret"] - labels["next_day_median"]
    labels["y_binary"] = (labels["u_excess"] > 0).astype(int)

    return labels[
        [
            "date",
            "infocode",
            "next_day_date",
            "next_day_ret",
            "next_day_median",
            "u_excess",
            "y_binary",
        ]
    ].sort_values(["date", "infocode"]).reset_index(drop=True)


def fetch_sp500_membership(conn: wrds.Connection) -> pd.DataFrame:
    """Fetch full S&P 500 Datastream membership spell table."""
    sp500 = conn.raw_sql(
        "SELECT constintcode, infocode, startdate, enddate "
        "FROM tr_ds_equities.ds2constmth "
        f"WHERE indexlistintcode = {SP500_CODE}"
    )
    sp500["startdate"] = pd.to_datetime(sp500["startdate"])
    sp500["enddate"] = pd.to_datetime(sp500["enddate"])
    sp500["infocode"] = sp500["infocode"].astype("Int64")
    return sp500


def build_monthly_membership(
    sp500: pd.DataFrame, latest_raw_date: pd.Timestamp
) -> pd.DataFrame:
    """Build month-end membership panel through the latest completed month."""
    last_month_end = latest_raw_date.to_period("M").to_timestamp("M")
    if latest_raw_date.normalize() < last_month_end.normalize():
        last_month_end = (latest_raw_date - pd.offsets.MonthEnd(1)).normalize()

    month_ends = pd.date_range("1989-12-31", last_month_end, freq="ME")
    records = []
    for me_date in month_ends:
        mask = (sp500["startdate"] <= me_date) & (sp500["enddate"] >= me_date)
        members = sp500.loc[mask, "infocode"].dropna().unique()
        effective = (me_date + pd.offsets.MonthBegin(1)).to_period("M")
        for ic in members:
            records.append(
                {
                    "infocode": int(ic),
                    "month_end_date": me_date,
                    "effective_month": effective,
                }
            )
    return pd.DataFrame(records)


def fetch_latest_ds_marketdate(
    conn: wrds.Connection, min_date: str = "2016-01-01"
) -> pd.Timestamp:
    """Ask WRDS for the latest Datastream marketdate available."""
    df = conn.raw_sql(
        "SELECT MAX(marketdate) AS max_date "
        "FROM tr_ds_equities.ds2primqtri "
        f"WHERE marketdate >= '{min_date}'"
    )
    return pd.to_datetime(df.loc[0, "max_date"])


def fetch_crsp_calendar(conn: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch US trading dates from CRSP index daily file."""
    cal = conn.raw_sql(
        "SELECT DISTINCT date "
        "FROM crsp.dsi "
        f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
        "ORDER BY date"
    )
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def build_nyse_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Build a standard NYSE trading calendar for dates not yet in CRSP."""
    nyse_bday = CustomBusinessDay(calendar=NYSEHolidayCalendar())
    dates = pd.date_range(start=start_date, end=end_date, freq=nyse_bday)
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def fetch_new_ri_rows(
    conn: wrds.Connection, infocodes: list[int], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch Datastream RI rows for the requested date span."""
    chunks = []
    total_chunks = (len(infocodes) + RI_CHUNK_SIZE - 1) // RI_CHUNK_SIZE
    for i in range(0, len(infocodes), RI_CHUNK_SIZE):
        chunk = infocodes[i : i + RI_CHUNK_SIZE]
        codes_str = ",".join(str(int(c)) for c in chunk)
        df = conn.raw_sql(
            f"SELECT infocode, marketdate, ri "
            f"FROM tr_ds_equities.ds2primqtri "
            f"WHERE infocode IN ({codes_str}) "
            f"  AND marketdate BETWEEN '{start_date}' AND '{end_date}'"
        )
        chunks.append(df)
        print(
            f"   RI chunk {i // RI_CHUNK_SIZE + 1}/{total_chunks}: {len(df):,} rows"
        )

    if not chunks:
        return pd.DataFrame(columns=["infocode", "marketdate", "ri"])

    ri = pd.concat(chunks, ignore_index=True)
    ri["marketdate"] = pd.to_datetime(ri["marketdate"])
    ri["infocode"] = ri["infocode"].astype("Int64")
    ri["ri"] = pd.to_numeric(ri["ri"], errors="coerce")
    ri = ri.dropna(subset=["ri"])
    return ri


def main():
    DS_DIR.mkdir(parents=True, exist_ok=True)

    raw_returns_path = DS_DIR / "ds_daily_returns.parquet"
    if not raw_returns_path.exists():
        raise FileNotFoundError(
            "Expected existing raw Datastream returns at data/datastream/ds_daily_returns.parquet"
        )

    print("=" * 72)
    print("EXTEND DATASTREAM PAPER-PARITY DATASET")
    print("=" * 72)

    old_raw = pd.read_parquet(raw_returns_path)
    old_raw["date"] = pd.to_datetime(old_raw["date"])
    last_local_raw_date = pd.to_datetime(old_raw["date"]).max()
    fetch_start_date = (last_local_raw_date).date().isoformat()
    usonly_path = DS_DIR / "ds_daily_returns_usonly.parquet"
    if usonly_path.exists():
        old_usonly = pd.read_parquet(usonly_path, columns=["date"])
        last_local_usonly_date = pd.to_datetime(old_usonly["date"]).max()
    else:
        last_local_usonly_date = pd.Timestamp("1989-01-01")

    print(f"Local raw Datastream data currently ends on {last_local_raw_date.date()}")
    print(f"Local US-only Datastream data currently ends on {last_local_usonly_date.date()}")

    print("\nConnecting to WRDS...")
    wrds_username = detect_wrds_username()
    conn = wrds.Connection(wrds_username=wrds_username)
    print("Connected.")

    # 1. Refresh full membership spell table.
    print("\n1. Refreshing Datastream S&P 500 membership spells...")
    sp500 = fetch_sp500_membership(conn)
    sp500.to_parquet(DS_DIR / "ds_sp500_membership.parquet", index=False)
    print(f"   {len(sp500):,} spells, {sp500['infocode'].nunique()} unique stocks")

    # 2. Discover latest available raw market date in Datastream.
    print("\n2. Discovering latest Datastream market date on WRDS...")
    latest_ds_date = fetch_latest_ds_marketdate(conn, min_date="2016-01-01")
    print(f"   Latest Datastream marketdate available: {latest_ds_date.date()}")

    raw_needs_extension = latest_ds_date > last_local_raw_date
    if raw_needs_extension:
        print("   Newer raw Datastream dates are available.")
    else:
        print("   Raw Datastream file is already current; refreshing filtered artifacts.")

    # 3. Pull new RI rows for all ever-members.
    ever_infocodes = sp500["infocode"].dropna().astype(int).unique().tolist()
    if raw_needs_extension:
        print("\n3. Pulling new Datastream RI rows...")
        new_ri = fetch_new_ri_rows(
            conn,
            ever_infocodes,
            start_date=fetch_start_date,
            end_date=latest_ds_date.date().isoformat(),
        )
        print(f"   New raw RI rows fetched: {len(new_ri):,}")
    else:
        print("\n3. No raw RI pull needed.")
        new_ri = pd.DataFrame(columns=["infocode", "marketdate", "ri"])

    # 4. Pull CRSP trading dates over the same extension range for US-only filter.
    print("\n4. Pulling CRSP trading calendar for extension range...")
    crsp_cal = fetch_crsp_calendar(
        conn,
        start_date=(last_local_usonly_date + pd.Timedelta(days=1)).date().isoformat(),
        end_date=latest_ds_date.date().isoformat(),
    )
    print(f"   New CRSP trading dates fetched: {len(crsp_cal):,}")

    conn.close()
    print("WRDS connection closed.")

    # 5. Rebuild full raw Datastream returns with dedupe and ret recomputation.
    print("\n5. Rebuilding raw Datastream daily returns...")
    old_ri = old_raw[["infocode", "date", "ri"]].rename(columns={"date": "marketdate"})
    combined_ri = pd.concat([old_ri, new_ri], ignore_index=True)
    combined_ri = combined_ri.dropna(subset=["infocode", "marketdate", "ri"])
    combined_ri["marketdate"] = pd.to_datetime(combined_ri["marketdate"])
    combined_ri["infocode"] = combined_ri["infocode"].astype("Int64")
    combined_ri["ri"] = pd.to_numeric(combined_ri["ri"], errors="coerce")
    combined_ri = combined_ri.drop_duplicates(
        subset=["infocode", "marketdate"], keep="last"
    ).sort_values(["infocode", "marketdate"]).reset_index(drop=True)
    combined_ri["ret"] = combined_ri.groupby("infocode")["ri"].pct_change()
    combined_returns = combined_ri.dropna(subset=["ret"]).rename(
        columns={"marketdate": "date"}
    )
    combined_returns.to_parquet(DS_DIR / "ds_daily_returns.parquet", index=False)
    print(
        f"   Rebuilt raw returns: {len(combined_returns):,} rows, "
        f"{combined_returns['date'].min().date()} to {combined_returns['date'].max().date()}"
    )

    # 6. Rebuild monthly membership and daily eligibility through latest complete month.
    print("\n6. Rebuilding monthly membership and daily eligibility...")
    membership = build_monthly_membership(sp500, latest_raw_date=combined_returns["date"].max())
    membership.to_parquet(DS_DIR / "ds_membership_monthly.parquet", index=False)

    trading_dates = combined_returns["date"].drop_duplicates().sort_values()
    dates_df = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    dates_df["effective_month"] = dates_df["date"].dt.to_period("M")
    eligible = dates_df.merge(
        membership[["effective_month", "infocode"]],
        on="effective_month",
        how="inner",
    )[["date", "infocode"]].sort_values(["date", "infocode"]).reset_index(drop=True)
    eligible.to_parquet(DS_DIR / "ds_universe_daily.parquet", index=False)
    print(
        f"   Rebuilt daily eligibility: {len(eligible):,} rows, "
        f"{eligible['date'].min().date()} to {eligible['date'].max().date()}"
    )

    # 7. Rebuild US-only filtered returns and eligibility.
    print("\n7. Rebuilding US-trading-calendar-filtered artifacts...")
    existing_usonly = DS_DIR / "ds_daily_returns_usonly.parquet"
    if existing_usonly.exists():
        old_us = pd.read_parquet(existing_usonly, columns=["date"])
        old_us_dates = pd.to_datetime(old_us["date"]).drop_duplicates()
    else:
        old_us_dates = pd.Series(dtype="datetime64[ns]")

    crsp_max_date = pd.to_datetime(crsp_cal["date"]).max() if len(crsp_cal) else pd.NaT
    if pd.isna(crsp_max_date):
        supplement = build_nyse_calendar(
            start_date=(last_local_usonly_date + pd.Timedelta(days=1)).date().isoformat(),
            end_date=latest_ds_date.date().isoformat(),
        )
        print(
            f"   CRSP calendar unavailable in extension range; "
            f"using NYSE holiday calendar for {len(supplement):,} dates"
        )
    elif crsp_max_date < latest_ds_date:
        supplement = build_nyse_calendar(
            start_date=(crsp_max_date + pd.Timedelta(days=1)).date().isoformat(),
            end_date=latest_ds_date.date().isoformat(),
        )
        print(
            f"   CRSP calendar ends at {crsp_max_date.date()}; "
            f"supplementing with NYSE holiday calendar for {len(supplement):,} dates "
            f"through {latest_ds_date.date()}"
        )
    else:
        supplement = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    full_us_dates = pd.concat(
        [old_us_dates, crsp_cal["date"], supplement["date"]], ignore_index=True
    )
    full_us_dates = pd.Series(
        pd.to_datetime(full_us_dates).drop_duplicates().sort_values()
    )

    us_date_set = set(full_us_dates.tolist())
    returns_us = combined_returns[combined_returns["date"].isin(us_date_set)].copy()
    eligible_us = eligible[eligible["date"].isin(us_date_set)].copy()

    returns_us.to_parquet(DS_DIR / "ds_daily_returns_usonly.parquet", index=False)
    eligible_us.to_parquet(DS_DIR / "ds_universe_daily_usonly.parquet", index=False)
    print(
        f"   US-only returns: {len(returns_us):,} rows, "
        f"{returns_us['date'].min().date()} to {returns_us['date'].max().date()}"
    )
    print(
        f"   US-only eligibility: {len(eligible_us):,} rows, "
        f"{eligible_us['date'].min().date()} to {eligible_us['date'].max().date()}"
    )

    # 8. Recompute features and labels on the US-only panel.
    print("\n8. Recomputing US-only features and labels...")
    features_us = compute_features(returns_us)
    labels_us = compute_labels(returns_us, eligible_us)
    features_us.to_parquet(DS_DIR / "ds_features_usonly.parquet", index=False)
    labels_us.to_parquet(DS_DIR / "ds_labels_usonly.parquet", index=False)
    print(
        f"   US-only features: {len(features_us):,} rows, "
        f"{features_us['date'].min().date()} to {features_us['date'].max().date()}"
    )
    print(
        f"   US-only labels: {len(labels_us):,} rows, "
        f"{labels_us['date'].min().date()} to {labels_us['date'].max().date()}"
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    for name in [
        "ds_sp500_membership.parquet",
        "ds_membership_monthly.parquet",
        "ds_daily_returns.parquet",
        "ds_universe_daily.parquet",
        "ds_daily_returns_usonly.parquet",
        "ds_universe_daily_usonly.parquet",
        "ds_features_usonly.parquet",
        "ds_labels_usonly.parquet",
    ]:
        p = DS_DIR / name
        mb = p.stat().st_size / 1e6
        print(f"  {name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
