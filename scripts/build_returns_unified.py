"""
Build a unified daily-returns parquet covering 1989-01-06 -> today.

Sources, in order of priority:
  1. Datastream US-only RI (1989-01-06 -> 2026-04-22) — primary historical
     backbone, matches the paper's data source.  Keyed by ``infocode``.
  2. yfinance daily Adj Close (2026-04-23 -> today) — fills the tail gap
     between the latest Datastream pull and today.  Keyed by ``ticker``,
     mapped back to ``infocode`` via ``data/datastream/ds_names.parquet``.

Output schema (data/processed/returns_unified.parquet)::

    date     : Timestamp
    infocode : Int64 (nullable; missing for yfinance rows whose ticker has
               no Datastream mapping)
    ticker   : str  (nullable; missing for purely-historical Datastream rows
               whose infocode never had a ticker recorded -- a few dozen names)
    ret      : float, simple daily return  P_t / P_{t-1} - 1
    ri       : float (nullable; only present for Datastream rows)
    source   : {'datastream', 'yfinance'}

Run::

    python scripts/build_returns_unified.py
    python scripts/build_returns_unified.py --append-live  # only new yfinance dates
    python scripts/build_returns_unified.py --skip-yfinance      # historical only
    python scripts/build_returns_unified.py --start 2026-04-21   # custom yfinance start

Notes
-----
- The yfinance leg only pulls tickers that are currently active in our
  Datastream universe (i.e. tickers that have at least one return observation
  in the last 60 days of the Datastream panel).  This keeps the universe
  consistent with what our period-33 model was trained on.
- yfinance returns are computed from ``Adj Close`` with ``auto_adjust=True``
  (split + dividend adjusted), giving a total-return analogue of Datastream's
  RI.  The two will not match to the bp on a single day's return for a stock
  that paid a dividend that day, but they're close.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"
PROCESSED = ROOT / "data" / "processed"
OUT_PATH = PROCESSED / "returns_unified.parquet"
UNIVERSE_DAILY_USONLY_PATH = DS_DIR / "ds_universe_daily_usonly.parquet"
MEMBERSHIP_MONTHLY_PATH = DS_DIR / "ds_membership_monthly.parquet"

# Infocodes that are part of our trained universe -- restrict yfinance pull
# to these so the unified panel is consistent with what the model expects.
# We use the period 33 prediction infocodes as the canonical "today's universe".
PREDICTIONS_PATH = DS_DIR / "predictions_phase2_ds.parquet"

# Datastream-style ticker -> Yahoo Finance ticker.  Yahoo uses dashes for
# share-class designations whereas Datastream / Bloomberg / Reuters use dots.
# Add new entries as needed.
TICKER_YAHOO_REMAP: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# A few failed / OTC names can keep emitting penny quotes after their common
# stock is no longer a realistic member of the live trading universe.  They
# create enormous percentage moves from tiny prices and can dominate the
# simulator.  The RI threshold is deliberately conservative: on the current
# Datastream tail it removes only the near-zero return-index names.
MIN_LIVE_UNIVERSE_RI = 1.0
MAX_ABS_LIVE_RETURN = 5.0


def load_datastream() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Datastream US-only daily returns + names (infocode -> ticker)."""
    rets = pd.read_parquet(DS_DIR / "ds_daily_returns_usonly.parquet")
    rets["date"] = pd.to_datetime(rets["date"])
    rets["infocode"] = rets["infocode"].astype("Int64")

    names = pd.read_parquet(DS_DIR / "ds_names.parquet")
    names["infocode"] = names["infocode"].astype("Int64")
    # Keep one ticker per infocode -- prefer non-null, prefer the most recent
    # spelling.  ds_names is sorted by infocode, multiple rows per infocode
    # represent ISIN/ticker history.  Take the last non-null ticker.
    names = names.dropna(subset=["ticker"]).drop_duplicates(
        subset="infocode", keep="last"
    )[["infocode", "ticker"]]
    return rets, names


def build_datastream_block(rets: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Datastream historical block, augmented with ticker."""
    df = rets.merge(names, on="infocode", how="left")
    df["source"] = "datastream"
    df = df[["date", "infocode", "ticker", "ret", "ri", "source"]]
    return df


def detect_partial_trailing_dates(rets: pd.DataFrame,
                                  threshold: float = 0.5,
                                  lookback_days: int = 10) -> list[pd.Timestamp]:
    """Find trailing dates whose row count is < ``threshold`` * the median
    count of the last ``lookback_days`` days.  Walks backwards from the last
    date and stops as soon as a date hits the threshold.  Returns the partial
    dates in ascending order; empty list if the last date already looks full.
    """
    counts = rets.groupby("date").size().sort_index()
    if len(counts) < 2:
        return []
    recent_window = counts.tail(lookback_days)
    typical = float(recent_window.median())
    bad_dates: list[pd.Timestamp] = []
    for d in reversed(counts.index):
        if counts.loc[d] >= threshold * typical:
            break
        bad_dates.append(pd.Timestamp(d))
    return list(reversed(bad_dates))


def _latest_sp500_universe_from_disk() -> pd.Series:
    """Latest point-in-time S&P 500 Datastream universe available locally."""
    if UNIVERSE_DAILY_USONLY_PATH.exists():
        universe_daily = pd.read_parquet(UNIVERSE_DAILY_USONLY_PATH)
        universe_daily["date"] = pd.to_datetime(universe_daily["date"])
        last_date = universe_daily["date"].max()
        latest = universe_daily.loc[
            universe_daily["date"].eq(last_date), "infocode"
        ]
        print(
            f"  Live universe source: S&P 500 eligibility on "
            f"{last_date.date()} ({latest.nunique():,} infocodes)"
        )
        return latest.astype("Int64").dropna().drop_duplicates()

    if MEMBERSHIP_MONTHLY_PATH.exists():
        membership = pd.read_parquet(MEMBERSHIP_MONTHLY_PATH)
        latest_month = membership["effective_month"].max()
        latest = membership.loc[
            membership["effective_month"].eq(latest_month), "infocode"
        ]
        print(
            f"  Live universe source: S&P 500 membership effective "
            f"{latest_month} ({latest.nunique():,} infocodes)"
        )
        return latest.astype("Int64").dropna().drop_duplicates()

    raise FileNotFoundError(
        "No Datastream S&P 500 universe file found. Expected "
        f"{UNIVERSE_DAILY_USONLY_PATH.relative_to(ROOT)} or "
        f"{MEMBERSHIP_MONTHLY_PATH.relative_to(ROOT)}."
    )


def select_live_universe(ds_rets: pd.DataFrame, names: pd.DataFrame,
                        cutoff_days: int = 60) -> pd.DataFrame:
    """Tickers in the latest point-in-time S&P 500 Datastream universe.

    ``cutoff_days`` is retained for backwards compatibility only.  The live
    universe must not be inferred from recent return activity, because failed
    ever-members can keep emitting OTC/penny-price rows long after they leave
    the S&P 500.
    """
    _ = cutoff_days
    try:
        active_codes = _latest_sp500_universe_from_disk()
    except FileNotFoundError as exc:
        print(f"  WARNING: {exc}")
        print("  Falling back to recent Datastream return activity.")
        last_date = ds_rets["date"].max()
        cutoff = last_date - pd.Timedelta(days=60)
        recent = ds_rets[ds_rets["date"] >= cutoff]
        active_codes = recent["infocode"].dropna().astype("Int64").drop_duplicates()

    universe = names[names["infocode"].isin(active_codes)].copy()

    latest_ri = (
        ds_rets.sort_values(["infocode", "date"])
        .dropna(subset=["ri"])
        .drop_duplicates(subset="infocode", keep="last")
        [["infocode", "ri"]]
        .rename(columns={"ri": "latest_ri"})
    )
    universe = universe.merge(latest_ri, on="infocode", how="left")
    before = len(universe)
    universe = universe[
        universe["latest_ri"].isna()
        | (universe["latest_ri"] >= MIN_LIVE_UNIVERSE_RI)
    ].copy()
    dropped = before - len(universe)
    if dropped:
        print(
            f"  Excluded {dropped} live-universe ticker(s) with latest "
            f"Datastream RI < {MIN_LIVE_UNIVERSE_RI:g}"
        )

    universe = universe.dropna(subset=["ticker"]).drop_duplicates(subset="ticker")
    return universe[["infocode", "ticker"]]


def _bad_live_return_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows whose yfinance return is outside plausible daily-return bounds."""
    ret = pd.to_numeric(frame["ret"], errors="coerce")
    return ret.le(-1.0) | ret.abs().gt(MAX_ABS_LIVE_RETURN)


def filter_live_return_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible yfinance live-tail returns with an audit print."""
    if frame.empty:
        return frame
    bad = _bad_live_return_mask(frame)
    if bad.any():
        sample = (
            frame.loc[bad, ["date", "ticker", "ret"]]
            .sort_values(["date", "ticker"])
            .head(10)
            .to_dict(orient="records")
        )
        print(
            f"  Dropping {int(bad.sum()):,} yfinance row(s) with "
            f"ret <= -100% or |ret| > {MAX_ABS_LIVE_RETURN:g}; "
            f"sample={sample}"
        )
    return frame.loc[~bad].copy()


def clean_existing_live_rows(
    current: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Remove stale or impossible yfinance rows already present on disk."""
    if current.empty or "source" not in current.columns:
        return current, False

    live_mask = current["source"].astype(str).eq("yfinance")
    if not live_mask.any():
        return current, False

    valid_codes = set(universe["infocode"].dropna().astype("Int64"))
    stale_universe = live_mask & ~current["infocode"].astype("Int64").isin(valid_codes)
    impossible_return = live_mask & _bad_live_return_mask(current)
    drop_mask = stale_universe | impossible_return

    if not drop_mask.any():
        return current, False

    print(
        f"  Cleaning existing yfinance rows: dropping {int(drop_mask.sum()):,} "
        f"row(s) ({int(stale_universe.sum()):,} outside live universe, "
        f"{int(impossible_return.sum()):,} impossible return rows)"
    )
    return current.loc[~drop_mask].copy(), True


def pull_yfinance(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Fetch yfinance Adj Close for a list of tickers and compute daily returns."""
    import yfinance as yf

    print(f"  Pulling {len(tickers)} tickers from yfinance "
          f"{start.date()} -> {end.date() if end is not None else 'today'} ...")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty and allow_empty:
        print("  yfinance returned no price rows.")
        return pd.DataFrame(columns=["date", "ticker", "ret"])
    if data.empty:
        raise RuntimeError(
            "yfinance returned no data -- check ticker list and date range"
        )

    # data has MultiIndex columns (price_field, ticker).  We want Close.
    closes = data["Close"]
    print(f"  Close panel: {closes.shape[0]} dates x {closes.shape[1]} tickers")

    # Daily simple return = P_t / P_{t-1} - 1.  fill_method=None to avoid
    # forward-filling NaNs (a stock that didn't trade should produce NaN, not
    # a fake zero-return next day).
    rets = closes.pct_change(fill_method=None)

    # Stack to long form
    rets_long = rets.stack(future_stack=True).reset_index()
    rets_long.columns = ["date", "ticker", "ret"]
    rets_long = rets_long.dropna(subset=["ret"])
    rets_long["date"] = pd.to_datetime(rets_long["date"])
    return rets_long


def build_yfinance_block(ds_rets: pd.DataFrame, names: pd.DataFrame,
                         start_date: pd.Timestamp,
                         end_date: pd.Timestamp | None,
                         *,
                         allow_empty: bool = False) -> pd.DataFrame:
    """Pull yfinance daily returns for the live ticker universe and key by infocode."""
    universe = select_live_universe(ds_rets, names, cutoff_days=60)
    ds_tickers = universe["ticker"].tolist()
    print(f"  Live universe size: {len(ds_tickers)} tickers")

    # Translate any Datastream-style class-share tickers to Yahoo notation
    # before the pull, then translate back so the output keys match the
    # historical Datastream block.
    yahoo_to_ds: dict[str, str] = {}
    pull_tickers: list[str] = []
    for t in ds_tickers:
        y = TICKER_YAHOO_REMAP.get(t, t)
        yahoo_to_ds[y] = t
        pull_tickers.append(y)
    if any(y != t for y, t in zip(pull_tickers, ds_tickers)):
        remapped = [(t, y) for t, y in zip(ds_tickers, pull_tickers) if y != t]
        print(f"  Yahoo ticker remaps: {remapped}")

    yf_rets = pull_yfinance(
        pull_tickers,
        start=start_date,
        end=end_date,
        allow_empty=allow_empty,
    )

    # Translate Yahoo tickers back to Datastream-style ticker before the merge.
    yf_rets["ticker"] = yf_rets["ticker"].map(lambda y: yahoo_to_ds.get(y, y))

    # Map ticker -> infocode
    yf_rets = yf_rets.merge(universe[["infocode", "ticker"]], on="ticker", how="left")
    yf_rets["infocode"] = yf_rets["infocode"].astype("Int64")
    yf_rets["ri"] = pd.NA  # yfinance doesn't expose an RI series
    yf_rets["source"] = "yfinance"

    # Keep only dates STRICTLY after the Datastream tail
    yf_rets = yf_rets[yf_rets["date"] > start_date].copy()
    yf_rets = filter_live_return_rows(yf_rets)
    return yf_rets[["date", "infocode", "ticker", "ret", "ri", "source"]]


def append_live_returns(end_date: pd.Timestamp | None) -> None:
    """Append only newly available yfinance dates to returns_unified.parquet."""
    if not OUT_PATH.exists():
        print("returns_unified.parquet is missing; running a full build.")
        args = argparse.Namespace(skip_yfinance=False, start=None, end=end_date)
        build_full(args)
        return

    print("=" * 70)
    print("Appending live daily returns")
    print("=" * 70)
    print("\n[1/4] Loading current unified returns...")
    try:
        current = pd.read_parquet(OUT_PATH)
    except Exception as exc:
        print(f"  Existing returns parquet is unreadable ({exc}); running full build.")
        args = argparse.Namespace(skip_yfinance=False, start=None, end=end_date)
        build_full(args)
        return
    required = {"date", "infocode", "ticker", "ret", "ri", "source"}
    missing = sorted(required - set(current.columns))
    if missing:
        print(f"  Existing returns parquet is missing {missing}; running full build.")
        args = argparse.Namespace(skip_yfinance=False, start=None, end=end_date)
        build_full(args)
        return
    current["date"] = pd.to_datetime(current["date"])
    print(f"  Current: {len(current):,} rows, "
          f"{current['date'].min().date()} -> {current['date'].max().date()}")

    print("\n[2/4] Loading Datastream universe metadata...")
    ds_rets, names = load_datastream()
    live_universe = select_live_universe(ds_rets, names, cutoff_days=60)
    current, cleaned_existing = clean_existing_live_rows(current, live_universe)

    # If a trailing live date looks sparse, drop it and refetch from the
    # previous available date. That lets the next refresh repair partial pulls.
    live = current[current["source"] == "yfinance"].copy()
    partial_dates = detect_partial_trailing_dates(live) if len(live) else []
    if partial_dates:
        first_partial = min(partial_dates)
        start = current.loc[current["date"] < first_partial, "date"].max()
        current = current[~current["date"].isin(partial_dates)].copy()
        print(f"  Dropping partial yfinance date(s): "
              f"{[d.date() for d in partial_dates]}")
    else:
        start = current["date"].max()
    start = pd.Timestamp(start)
    print(f"  Pull start: {start.date()} "
          f"(new rows must be strictly after this date)")

    print("\n[3/4] Pulling live yfinance rows...")
    yf_block = build_yfinance_block(
        ds_rets,
        names,
        start_date=start,
        end_date=end_date,
        allow_empty=True,
    )
    if yf_block.empty:
        if cleaned_existing:
            print("  No new yfinance rows; saving cleaned existing file.")
            current = current.sort_values(
                ["date", "infocode", "ticker"]
            ).reset_index(drop=True)
            current.to_parquet(OUT_PATH, index=False)
            print(f"  Wrote {len(current):,} rows, "
                  f"{current['date'].min().date()} -> "
                  f"{current['date'].max().date()}")
        else:
            print("  No new yfinance rows. Existing file is already current.")
        return

    print(f"  New yfinance block: {len(yf_block):,} rows, "
          f"{yf_block['date'].min().date()} -> {yf_block['date'].max().date()}, "
          f"{yf_block['infocode'].nunique()} infocodes mapped")

    print("\n[4/4] Saving appended unified parquet...")
    append_dates = set(pd.to_datetime(yf_block["date"]).unique())
    current = current[~current["date"].isin(append_dates)].copy()
    unified = pd.concat([current, yf_block], ignore_index=True)
    unified = unified.sort_values(["date", "infocode", "ticker"]).reset_index(drop=True)
    unified.to_parquet(OUT_PATH, index=False)

    print(f"  Wrote {len(unified):,} rows, "
          f"{unified['date'].min().date()} -> {unified['date'].max().date()}")


def build_full(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("Building unified daily-returns parquet")
    print("=" * 70)

    print("\n[1/3] Loading Datastream historical block...")
    ds_rets, names = load_datastream()
    print(f"  Datastream: {len(ds_rets):,} rows, "
          f"{ds_rets['date'].min().date()} -> {ds_rets['date'].max().date()}, "
          f"{ds_rets['infocode'].nunique():,} infocodes")

    # Drop any sparse trailing dates from Datastream so yfinance owns the
    # tail seam cleanly.  E.g. if the WRDS Datastream pull ended mid-day on
    # the last date and only a handful of stocks have a return that day,
    # the row count for that date will be << the median of recent days.
    partial_dates = (
        detect_partial_trailing_dates(ds_rets) if not args.skip_yfinance else []
    )
    if partial_dates:
        print(f"  Detected partial trailing date(s) in Datastream: "
              f"{[d.date() for d in partial_dates]} -- dropping so yfinance owns them")
        ds_rets = ds_rets[~ds_rets["date"].isin(partial_dates)].copy()
        print(f"  Datastream after trim: {len(ds_rets):,} rows, "
              f"new last date: {ds_rets['date'].max().date()}")

    historical = build_datastream_block(ds_rets, names)
    print(f"  Historical block: {len(historical):,} rows "
          f"({historical['ticker'].notna().sum():,} have ticker)")

    if args.skip_yfinance:
        print("\n[2/3] yfinance step skipped.")
        unified = historical
    else:
        print("\n[2/3] Pulling yfinance gap-fill...")
        ds_last = ds_rets["date"].max()
        start = pd.Timestamp(args.start) if args.start else ds_last
        end = pd.Timestamp(args.end) if args.end else None
        yf_block = build_yfinance_block(ds_rets, names, start_date=start, end_date=end)
        print(f"  yfinance block: {len(yf_block):,} rows, "
              f"{yf_block['date'].min().date() if len(yf_block) else 'n/a'} -> "
              f"{yf_block['date'].max().date() if len(yf_block) else 'n/a'}, "
              f"{yf_block['infocode'].nunique() if len(yf_block) else 0} "
              "infocodes mapped")
        if len(yf_block):
            unified = pd.concat([historical, yf_block], ignore_index=True)
        else:
            unified = historical

    print("\n[3/3] Saving unified parquet...")
    unified = unified.sort_values(["date", "infocode", "ticker"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUT_PATH, index=False)

    print()
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  Total rows: {len(unified):,}")
    print(
        f"  Date range: {unified['date'].min().date()} -> "
        f"{unified['date'].max().date()}"
    )
    print("  Sources:")
    for src, cnt in unified["source"].value_counts().items():
        sub = unified[unified["source"] == src]
        print(f"    {src:10s}: {cnt:>10,} rows, "
              f"{sub['date'].min().date()} -> {sub['date'].max().date()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--append-live",
        action="store_true",
        help="Append only new yfinance rows to returns_unified.parquet.",
    )
    parser.add_argument(
        "--skip-yfinance",
        action="store_true",
        help="Build only the Datastream historical block.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="yfinance start date (default = Datastream's last date, "
             "so only strictly-newer days are appended).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="yfinance end date (default = today / yfinance's latest).",
    )
    args = parser.parse_args()

    if args.append_live:
        end = pd.Timestamp(args.end) if args.end else None
        append_live_returns(end)
    else:
        build_full(args)


if __name__ == "__main__":
    main()
