"""Page 9 — Conclusions.

Three sections: what worked, what didn't, what I'd build next.  Written in
plain English for any reader who has been through the Background primer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Conclusion", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete,
    load_cost_bands,
    load_regime_leg_decomp,
    load_summary_table,
    missing_build_warning,
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
   Sharpe of **2.18** against the paper's reported 1.81. The gap is directionally
   explained by lower realised turnover in our reproduction, which makes the
   same 5 bps / half-turn charge bite less. We reproduce the key ordering in
   the paper baseline: ENS1 is the strongest P-only model, and the k = 10
   portfolio is the concentrated headline setting.
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
2. **The z-score composite lifts daily return but not Sharpe.** Ranking
   by `0.5·z(P̂) + 0.5·z(Û)` — the equally-weighted average of cross-sectional
   z-scores of the direction and magnitude predictions — lifts daily return
   from 0.28 % to 0.31 %. It still trails P-only on Sharpe, so the honest
   conclusion is narrower than "better model": magnitude adds information,
   but not enough to dominate the paper baseline on risk-adjusted performance.

3. **The no-trade band is useful, but model-dependent.** A 10 bps band
   replaces an incumbent only when the new candidate improves predicted
   excess return by at least one round-trip cost. In the app's 16-row
   cost-band matrix, XGB is the consistent winner; RF is mixed; ENS1 and DNN
   do not automatically improve. The rule works only when Û is wide and
   reliable enough to distinguish true improvement from churn.

4. **The directional-disagreement analysis explains the product
   composite's failure.** Multiplying `(2P̂ − 1)` by `Û` only works when the
   two heads agree on sign. In the CRSP era they disagree on ~49 % of
   stocks per day — essentially independent — so the product flips sign
   constantly and negatively contaminates the tails. The z-score
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
   VIX regime grid on Results & Risk Diagnostics is representative:
   k = 10 dominates in every regime, and the best regime-conditional k
   fit ties rather than beats the fixed baseline. Two earlier regime
   frameworks (moving-average/drawdown regimes and a leading-indicator
   cross-validation regime classifier) came to the same conclusion through
   different routes. Whatever is dragging on the strategy post-2015 is
   **not concentrated in a particular vol regime** — it's regime-universal.

2. **Post-2015 alpha has collapsed for the broad daily rankings.** On our
   2015-live extension, ENS1 P-only, U-only, Z-comp, and Product all land at
   negative post-cost Sharpe. The P-gates are the exception in the materialized
   table: they stay positive by trading fewer, higher-conviction days. The
   directional-disagreement rate drops from ~49 % to ~27 % across the break,
   but that isn't cleanly good news: Û has compressed in magnitude, so the
   second head no longer supplies the same independent magnitude signal.

3. **No scheme survives the combination of cost drag and decay.** The
   post-cost CRSP Sharpe of 2.18 would have been tradeable; the post-cost
   extension Sharpe of the broad ENS1 P-only baseline is around −0.4. The gap
   between those two numbers is the story: a model that worked on the paper
   era stops looking like a broad daily trading edge in the next decade.
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
            "Post-cost Sharpe (extension)": st.column_config.NumberColumn(
                format="%.2f"
            ),
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
    "**Bottom line.** The paper's result reproduces cleanly, the "
    "magnitude-aware extensions add useful diagnostics and selective gated "
    "variants, and the 2015-live out-of-sample deterioration is the result "
    "worth taking most seriously. The next build should either bring new "
    "features to the problem or be honest about the decay and move on."
)
