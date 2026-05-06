"""Annualisation conventions used across the project.

Background
----------
The Krauss et al. (2017) paper reports annualised returns using compounded
(CAGR-style) annualisation, not arithmetic-mean annualisation.  Earlier
notebooks used ``r.mean() * 252`` which is arithmetic and over-states or
under-states the paper's number depending on the variance drag of the
series.  This module provides the canonical helpers so every notebook,
script, and downstream artefact uses the same convention.

Key formulas
------------
- CAGR (compound annual growth rate) on a daily return series ``r``::

      CAGR = ((1 + r).prod()) ** (252 / n) - 1

  where ``n = len(r)``.  This matches paper Tables 3 and 5.

- Sharpe ratio (annualised, paper convention)::

      Sharpe = (CAGR - rf) / (r.std() * sqrt(252))

  where ``rf`` is the annual risk-free rate (paper uses ~0.02).

Cost convention
---------------
Two cost regimes are supported:

- **Realistic / project default**: charge realised turnover per day at
  ``5 bps`` per half-turn.  This is the implementation in
  ``costs.apply_transaction_costs``.  Computed turnover for a paper-style
  k=10 long-short portfolio averages ~2.5/day in the H2O reproduction
  (vs the paper's implied ~4.0).

- **Paper convention**: charge a flat ``20 bps/day`` (= 5 bps × 4
  half-turns).  Equivalent to assuming turnover is 4 every day, i.e. a
  full daily round-trip on both legs.  This is what the paper does --
  the paper's pre-vs-post mean-return gap of 0.0020 implies a flat
  daily charge, not a turnover-scaled one.  Use this convention when
  comparing to paper Tables 2/3/5 numbers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PAPER_BPS_PER_DAY: float = 20.0
"""Paper-convention transaction cost: 20 bps/day flat (= 5 bps × 4 half-turns)."""

DEFAULT_PERIODS_PER_YEAR: int = 252
"""US trading days per year, used by the paper."""


def cagr(daily_returns, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate of a daily return series.

    Parameters
    ----------
    daily_returns : array-like of daily simple returns (e.g. 0.001 = 10 bps).
    periods_per_year : int, default 252.

    Returns
    -------
    float : annualised compound return.  Returns NaN for empty input.
    """
    r = pd.Series(daily_returns).dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    total = float((1.0 + r).prod())
    if total <= 0:
        # Wealth went to zero or negative -- CAGR is technically undefined.
        # Return -100% as a sentinel for "totally lost capital".
        return -1.0
    return total ** (periods_per_year / n) - 1.0


def annualised_vol(daily_returns,
                   periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """Annualised standard deviation = daily_std * sqrt(periods_per_year)."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(periods_per_year))


def sharpe_annual(daily_returns,
                  rf_annual: float = 0.0,
                  periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe = (CAGR - rf) / (daily_std * sqrt(periods_per_year)).

    Notes
    -----
    Matches the paper's convention (compound numerator, daily-std times
    sqrt(252) denominator, fixed annual risk-free rate).  Set ``rf_annual=0``
    for the "no excess return" variant used in some Phase 2 notebooks.
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return float("nan")
    ann_ret = cagr(r, periods_per_year)
    ann_std = annualised_vol(r, periods_per_year)
    if ann_std <= 0:
        return float("nan")
    return (ann_ret - rf_annual) / ann_std


def sortino_annual(daily_returns,
                   rf_annual: float = 0.0,
                   periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """Annualised Sortino, using only downside deviation in the denominator."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return float("nan")
    ann_ret = cagr(r, periods_per_year)
    downside = r[r < 0]
    if len(downside) < 2:
        return float("nan")
    ann_downside = float(downside.std() * np.sqrt(periods_per_year))
    if ann_downside <= 0:
        return float("nan")
    return (ann_ret - rf_annual) / ann_downside


def apply_paper_cost(gross_returns,
                     bps_per_day: float = PAPER_BPS_PER_DAY) -> pd.Series:
    """Subtract a flat per-day cost (paper convention).

    Parameters
    ----------
    gross_returns : array-like of daily simple gross returns.
    bps_per_day : flat daily cost in basis points.  Default 20 bps =
                  5 bps/half-turn × 4 half-turns/day (full round-trip on
                  both legs of the long-short book).
    """
    r = pd.Series(gross_returns)
    return r - (bps_per_day / 1e4)


def calmar_ratio(daily_returns,
                 periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    """CAGR / |max drawdown| using compounded equity curve."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return float("nan")
    ann_ret = cagr(r, periods_per_year)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    mdd = abs(float(drawdown.min()))
    if mdd <= 0:
        return float("nan")
    return ann_ret / mdd
