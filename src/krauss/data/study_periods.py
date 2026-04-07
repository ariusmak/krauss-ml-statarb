"""
Study period construction per Krauss et al. (2017).

Each study period consists of:
    - 750 trading days for training
    - 250 trading days for trading
    - The first 240 days of training are consumed by feature lookbacks,
      leaving ~510 usable training observations per stock.

Study periods are built by rolling forward through the sorted trading
dates. Each new period advances by trade_days (250), so training
windows overlap. This matches the paper's coverage through Oct 2015.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class StudyPeriod:
    """One train/trade block."""
    period_id: int
    train_dates: np.ndarray   # all 750 training dates
    trade_dates: np.ndarray   # all 250 trading dates
    # Usable training dates (after 240-day lookback consumed)
    usable_train_dates: np.ndarray

    @property
    def train_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates[0])

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates[-1])

    @property
    def trade_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.trade_dates[0])

    @property
    def trade_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.trade_dates[-1])

    @property
    def usable_train_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.usable_train_dates[0])


def build_study_periods(
    trading_dates: np.ndarray,
    train_days: int = 750,
    trade_days: int = 250,
    lookback_days: int = 240,
    first_train_date: str = "1990-01-01",
) -> list[StudyPeriod]:
    """
    Partition trading dates into rolling study periods.

    Parameters
    ----------
    trading_dates : np.ndarray of datetime64
        Sorted unique trading dates from the return panel.
    train_days : int
        Number of trading days in each training window.
    trade_days : int
        Number of trading days in each trading window.
    lookback_days : int
        Feature lookback that consumes the start of each training window.
    first_train_date : str
        Earliest allowed start date for the first training window.
        Paper uses Jan 1990 data start; raw data goes back to 1989
        for lookback room.

    Returns
    -------
    list[StudyPeriod]
    """
    dates = np.sort(trading_dates)
    total_needed = train_days + trade_days
    periods = []
    period_id = 0

    # Anchor first training window at or after first_train_date
    anchor = np.datetime64(first_train_date)
    start_idx = int(np.searchsorted(dates, anchor))

    while start_idx + total_needed <= len(dates):
        train = dates[start_idx : start_idx + train_days]
        trade = dates[start_idx + train_days : start_idx + total_needed]
        usable_train = train[lookback_days:]

        periods.append(StudyPeriod(
            period_id=period_id,
            train_dates=train,
            trade_dates=trade,
            usable_train_dates=usable_train,
        ))

        # Advance by trade_days (rolling windows, training overlaps)
        start_idx += trade_days
        period_id += 1

    return periods


def study_periods_summary(periods: list[StudyPeriod]) -> pd.DataFrame:
    """Return a summary DataFrame of all study periods."""
    rows = []
    for sp in periods:
        rows.append({
            "period_id": sp.period_id,
            "train_start": sp.train_start,
            "train_end": sp.train_end,
            "usable_train_start": sp.usable_train_start,
            "usable_train_days": len(sp.usable_train_dates),
            "trade_start": sp.trade_start,
            "trade_end": sp.trade_end,
            "trade_days": len(sp.trade_dates),
        })
    return pd.DataFrame(rows)
