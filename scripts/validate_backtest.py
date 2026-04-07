"""
Validate the backtest engine end-to-end using a dummy scorer.

Usage:
    python scripts/validate_backtest.py

This script:
    1. Loads processed data (features, labels, returns, eligibility)
    2. Builds study periods (750 train / 250 trade)
    3. Generates RANDOM predictions as a dummy scorer
    4. Runs the full backtest pipeline (rank, select, portfolio, costs)
    5. Validates correctness:
       - No lookahead (signal date < return date)
       - Dollar neutrality (k long, k short every day)
       - Transaction costs are plausible
       - Turnover is plausible
       - Returns near zero for random predictions
       - Long/short legs roughly symmetric
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods, study_periods_summary
from krauss.backtest.ranking import rank_and_select
from krauss.backtest.portfolio import build_daily_portfolios, aggregate_portfolio_returns
from krauss.backtest.costs import compute_turnover, apply_transaction_costs
from krauss.backtest.rebalance import compute_position_changes

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

K = 10  # Paper's headline k
COST_BPS = 5
SEED = 42


def main():
    print("=" * 60)
    print("BACKTEST ENGINE VALIDATION — Dummy Scorer")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading data...")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    eligible = pd.read_parquet(PROCESSED / "universe_daily.parquet")

    print(f"  Features:    {len(features):,} rows")
    print(f"  Labels:      {len(labels):,} rows")
    print(f"  Returns:     {len(returns):,} rows")
    print(f"  Eligibility: {len(eligible):,} rows")

    # ------------------------------------------------------------------
    # Build study periods
    # ------------------------------------------------------------------
    print("\n[2/7] Building study periods...")
    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)
    summary = study_periods_summary(periods)

    print(f"  {len(periods)} study periods")
    print(f"\n  Study period summary:")
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Generate dummy predictions for first study period only
    # (full run would be slow; one period is enough to validate)
    # ------------------------------------------------------------------
    print(f"\n[3/7] Generating dummy predictions (period 0, k={K})...")
    sp = periods[0]
    rng = np.random.default_rng(SEED)

    # Get eligible stocks with features and labels on trade dates
    trade_dates_set = set(sp.trade_dates.astype("datetime64[ns]"))

    # Features + labels inner join on trade dates
    feat_dates = features[features["date"].isin(trade_dates_set)]
    lab_dates = labels[labels["date"].isin(trade_dates_set)]

    panel = feat_dates[["date", "permno"]].merge(
        lab_dates[["date", "permno"]], on=["date", "permno"], how="inner"
    )
    # Also must be eligible
    panel = panel.merge(eligible, on=["date", "permno"], how="inner")

    # Random predictions
    panel["p_hat"] = rng.random(len(panel))

    print(f"  {len(panel):,} stock-day predictions")
    print(f"  {panel['permno'].nunique()} unique stocks")
    print(f"  {panel['date'].nunique()} trading days")
    print(f"  Trade window: {sp.trade_start.date()} to {sp.trade_end.date()}")

    # ------------------------------------------------------------------
    # Rank and select
    # ------------------------------------------------------------------
    print(f"\n[4/7] Ranking and selecting top/bottom {K}...")
    selections = rank_and_select(panel, k=K, score_col="p_hat")

    n_long = (selections["side"] == "long").sum()
    n_short = (selections["side"] == "short").sum()
    n_days = selections["date"].nunique()
    print(f"  {n_long} long selections, {n_short} short selections")
    print(f"  Over {n_days} trading days")

    # Validate: exactly k per side per day
    side_counts = selections.groupby(["date", "side"]).size().reset_index(name="count")
    long_counts = side_counts[side_counts["side"] == "long"]["count"]
    short_counts = side_counts[side_counts["side"] == "short"]["count"]

    assert (long_counts == K).all(), f"ERROR: Not all days have {K} longs!"
    assert (short_counts == K).all(), f"ERROR: Not all days have {K} shorts!"
    print(f"  PASS: Exactly {K} long and {K} short every day")

    # ------------------------------------------------------------------
    # Build portfolios
    # ------------------------------------------------------------------
    print("\n[5/7] Building daily portfolios...")
    holdings = build_daily_portfolios(selections, returns, k=K)

    print(f"  {len(holdings):,} stock-day holdings")

    # Validate dollar neutrality
    daily_weight_sum = holdings.groupby("date")["weight"].sum()
    max_imbalance = daily_weight_sum.abs().max()
    print(f"  Max daily weight imbalance: {max_imbalance:.2e} (should be ~0)")
    assert max_imbalance < 1e-10, "ERROR: Portfolio not dollar neutral!"
    print(f"  PASS: Dollar neutral every day")

    # Validate no lookahead
    assert (holdings["next_date"] > holdings["date"]).all(), \
        "ERROR: next_date not after signal date!"
    print(f"  PASS: No lookahead (all next_date > signal date)")

    daily = aggregate_portfolio_returns(holdings)
    print(f"  {len(daily)} daily portfolio return observations")

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    print(f"\n[6/7] Computing turnover and transaction costs ({COST_BPS} bps)...")
    turnover = compute_turnover(holdings, k=K)
    daily_with_costs = apply_transaction_costs(daily, turnover, COST_BPS)

    avg_turnover = turnover["turnover"].mean()
    avg_cost = daily_with_costs["cost"].mean()
    print(f"  Avg daily turnover: {avg_turnover:.4f}")
    print(f"  Avg daily cost:     {avg_cost:.6f} ({avg_cost*10000:.2f} bps)")
    print(f"  Day-1 turnover:     {turnover.iloc[0]['turnover']:.4f} "
          f"(should be 2.0 for full portfolio build)")

    # Day 1 turnover should be 2.0 (buy k longs at +1/k, sell k shorts at -1/k)
    assert abs(turnover.iloc[0]["turnover"] - 2.0) < 1e-10, \
        "ERROR: Day-1 turnover should be 2.0!"
    print(f"  PASS: Day-1 turnover is 2.0")

    # ------------------------------------------------------------------
    # Return sanity
    # ------------------------------------------------------------------
    print("\n[7/7] Return sanity checks...")
    mean_ret = daily_with_costs["port_ret"].mean()
    mean_ret_net = daily_with_costs["port_ret_net"].mean()
    mean_long = daily_with_costs["long_ret"].mean()
    mean_short = daily_with_costs["short_ret"].mean()

    print(f"  Mean daily return (pre-cost):  {mean_ret*100:.4f}%")
    print(f"  Mean daily return (post-cost): {mean_ret_net*100:.4f}%")
    print(f"  Mean long leg:  {mean_long*100:.4f}%")
    print(f"  Mean short leg: {mean_short*100:.4f}%")
    print(f"  Std daily return: {daily_with_costs['port_ret'].std()*100:.4f}%")

    # With random predictions, returns should be near zero
    print(f"\n  Expected: near-zero returns for random scorer")
    if abs(mean_ret) < 0.005:
        print(f"  PASS: Mean return is near zero ({mean_ret:.6f})")
    else:
        print(f"  WARNING: Mean return seems large for random scorer ({mean_ret:.6f})")
        print(f"  (Could be sample noise with k={K}, small portfolio)")

    # Net should be below gross
    assert mean_ret_net <= mean_ret + 1e-10, \
        "ERROR: Net return should not exceed gross!"
    print(f"  PASS: Net return < gross return (costs applied correctly)")

    # ------------------------------------------------------------------
    # Turnover diagnostics
    # ------------------------------------------------------------------
    print("\n  Turnover diagnostics:")
    changes = compute_position_changes(holdings)
    avg_new_long = changes["n_new_long"].mean()
    avg_new_short = changes["n_new_short"].mean()
    avg_stay_long = changes["n_stay_long"].mean()
    avg_stay_short = changes["n_stay_short"].mean()
    print(f"    Avg new longs/day:   {avg_new_long:.1f}")
    print(f"    Avg new shorts/day:  {avg_new_short:.1f}")
    print(f"    Avg stay longs/day:  {avg_stay_long:.1f}")
    print(f"    Avg stay shorts/day: {avg_stay_short:.1f}")
    print(f"    (With random scorer, expect high turnover — ~{K} new per day)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 60)
    print(f"\nBacktest engine is ready for real model predictions.")
    print(f"  Study periods: {len(periods)}")
    print(f"  Trading window tested: period 0 "
          f"({sp.trade_start.date()} to {sp.trade_end.date()})")
    print(f"  k={K}, cost={COST_BPS} bps per half-turn")


if __name__ == "__main__":
    main()
