"""Page 9 — Conclusions.

Three sections: what worked, what didn't, what I'd build next.  Written in
plain English for any reader who has been through the Background primer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Conclusion", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_summary_table, load_cost_bands, load_regime_leg_decomp,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
bands = load_cost_bands()
regime_leg = load_regime_leg_decomp()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains "
    "the vocabulary used below — statistical arbitrage, Sharpe, alpha decay, "
    "post-cost."
)

st.title("Conclusion")
st.caption(
    "A plain-English wrap-up of what the project found, what it didn't, "
    "and what a follow-up would try."
)

# --- What worked -----------------------------------------------------------
st.header("What worked")

st.markdown(
    """
1. **The paper reproduces.** Our 1992-2015 ENS1 baseline earns a post-cost
   Sharpe of **2.18** against the paper's reported **1.81** — a difference
   of 0.37, mainly because our walk-forward turnover (~2.5 half-turns/day)
   is lower than the paper's implied ~4.0, so the same 5 bps/half-turn
   convention bites less hard on our runs. We reproduce the sign of every
   headline result on that era:
   P-only beats U-only, product composite destroys value, the top-k portfolio
   concentrates its Sharpe at k = 10.
"""
)

ens1_crsp = summary.query(
    "model == 'ENS1' and era == '1992-2015 (CRSP)' "
    "and cost_regime == '5bps_half_turn'"
).sort_values("sharpe", ascending=False)
st.dataframe(
    ens1_crsp[["scheme", "daily_return", "sharpe", "matched_days_return"]]
    .rename(columns={
        "scheme": "Scheme",
        "daily_return": "Daily ret",
        "sharpe": "Post-cost Sharpe",
        "matched_days_return": "Matched-days ret",
    }),
    use_container_width=True, hide_index=True,
    column_config={
        "Daily ret": st.column_config.NumberColumn(format="%.4f"),
        "Post-cost Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Matched-days ret": st.column_config.NumberColumn(format="%.4f"),
    },
)

st.markdown(
    """
2. **The z-score composite lifts daily return in the training era.** Ranking
   by `0.5·z(P̂) + 0.5·z(Û)` — the equally-weighted average of cross-sectional
   z-scores of the direction and magnitude predictions — lifts daily return
   from 0.28 % to 0.31 %. Sharpe is slightly lower (1.97 vs P-only's 2.18)
   because Z-comp runs at higher vol; the daily-return lift is consistent
   across sub-periods and survives the cost charge. That's the strongest
   unconditional improvement to the paper's ranking rule on daily return,
   though not on risk-adjusted return.

3. **The no-trade band helps XGB, hurts DNN and ENS1.** A 10 bps band
   (skip trades smaller than 10 bps of NAV) gives positive Sharpe deltas
   in every XGB cell (the only model where this is true), small mixed
   effects on RF (3 of 4 cells slightly negative, one marginally positive),
   and negative deltas across the board for DNN and ENS1. XGB *improves*
   under the band because its daily rank flips are churn more than signal;
   see the cost-analysis page for the full 16-row matrix and the
   explanation of why XGB is the only model that benefits.

4. **The directional-disagreement analysis explains the product
   composite's failure.** Multiplying `(2P̂ − 1)` by `Û` only works when the
   two heads agree on sign. In the CRSP era they disagree on ~49 % of
   stocks per day — essentially independent — so the product flips sign
   constantly and negative-ally contaminates the tails. The z-score
   composite side-steps this by adding instead of multiplying, so
   disagreement just attenuates the score toward zero rather than flipping
   it.
"""
)

# --- What didn't work ------------------------------------------------------
st.header("What didn't work")

st.markdown(
    """
1. **Three independent regime frameworks all produced null results.** The
   VIX regime grid (shown on the *What didn't work* page) is representative:
   k = 10 dominates in every regime, and the best regime-conditional k
   fit ties rather than beats the fixed baseline. Two earlier regime
   frameworks (moving-average/drawdown regimes and a leading-indicator
   cross-validation regime classifier) came to the same conclusion through
   different routes. Whatever is dragging on the strategy post-2015 is
   **not concentrated in a particular vol regime** — it's regime-universal.

2. **Post-2015 baseline alpha has collapsed.** On our 2015-2025 extension,
   ENS1's four non-gated schemes all land at negative post-cost
   Sharpe: P-only at −0.39, U-only at −0.66, Z-comp at −0.69, and Product
   at −1.41. The signal in the *baseline ranking rule* has decayed across
   every magnitude-aware variant. The directional-disagreement rate drops
   from ~49 % to ~27 % across the break, but that isn't good news. Û has
   compressed by ~1.5× (the top-10/bottom-10 spread of |Û| drops from
   0.63 % to 0.43 %) and its sign has become more aligned with
   P̂ − 0.5. The magnitude head has lost most of its independent
   information.

3. **Only the gated schemes retain post-cost signal in the extension.**
   ENS1 P-gate(0.05) earns +0.94 Sharpe on the 32 % of days it trades
   — a result that lands at the 99.8th percentile of a null distribution
   built from 10,000 random 792-day subsamples of P-only's extension
   returns (`notebooks/null_test_pgate.ipynb`, documented in
   `docs/reproduction_deviations.md` §17). The gate is identifying days
   the model is informative on, not picking a lucky subsample.
   P-gate(0.03), which trades more often (91 % of days), earns a smaller
   +0.39 Sharpe. Two caveats remain: (a) the gate trades 68 % less often
   than the baseline, so the same edge is expressed over a much shorter
   window; (b) the null test rules out sampling-fluke explanations but
   doesn't distinguish "rare strong edge isolated by the gate" from
   "small persistent edge scaled by trading less often."
"""
)

# Little cross-check table summarising era 2 post-cost.
ens1_ext = summary.query(
    "model == 'ENS1' and era == '2015-2025 (extension)' "
    "and cost_regime == '5bps_half_turn'"
).sort_values("sharpe", ascending=False)
if not ens1_ext.empty:
    st.dataframe(
        ens1_ext[["scheme", "daily_return", "sharpe"]].rename(columns={
            "scheme": "Scheme",
            "daily_return": "Daily ret (extension)",
            "sharpe": "Post-cost Sharpe (extension)",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Daily ret (extension)": st.column_config.NumberColumn(format="%.4f"),
            "Post-cost Sharpe (extension)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

# --- What I'd build next ---------------------------------------------------
st.header("What I'd build next")

st.markdown(
    """
1. **Features beyond lagged returns.** The entire feature set in the
   paper — and in this reproduction — is 31 transformations of past daily
   returns. It's a striking demonstration that shallow data can carry
   alpha, but it's also almost certainly where the decay lives. A serious
   follow-up would add: (a) **intraday features** such as realised
   volatility and overnight gaps, (b) **flow features** such as short
   interest changes and ETF-rebalance footprints, and (c) **fundamental
   features** such as earnings surprise and analyst-revision deltas.
   Any of those would give the model something to learn beyond pure
   short-horizon mean reversion, which is what 31 lagged returns really
   capture.

2. **A realistic short-borrow cost model.** The 5-bps-per-half-turn
   assumption is fine for longs but cheap for shorts, especially on hard-
   to-borrow names. A follow-up would model short-borrow fees per stock
   per day (roughly 25-50 bps annualised for easy names, multiples of
   that for hard-to-borrow), and would penalise the short leg
   accordingly. My expectation is that this would eat ~20-40 % of the
   CRSP-era Sharpe and push the extension-era Sharpe further negative —
   useful because it tells you which alpha sources still clear the
   realistic cost bar.

3. **A live paper-trading feed.** The simplest way to tell whether the
   post-2015 collapse is genuine signal death or an artefact of the
   Datastream data pipeline is to run the same model live on a
   point-in-time survivorship-bias-free feed (Polygon / IEX / LSEG
   intraday) for six to twelve months and compare the live P&L against
   both the backtest-replay on the same dates and the live SPY return.
   If live Sharpe tracks the backtest-replay, the decay is real; if it
   tracks the CRSP-era number, we have a data-contamination story in the
   extension.
"""
)

st.divider()
st.info(
    "**Bottom line.** The paper's headline ranking rule reproduces "
    "cleanly, the magnitude-aware extensions add a modest but honest "
    "lift in the training era on daily return (though not Sharpe), and "
    "the 2015-2025 baseline collapse is the result worth taking most "
    "seriously. The gated schemes retain extension-era signal but only "
    "on a small subset of days; our follow-up work should focus on "
    "understanding *why* the gates work — whether they isolate a real "
    "residual edge or scale a tiny persistent one."
)
