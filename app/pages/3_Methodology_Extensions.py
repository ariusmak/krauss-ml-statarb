"""Page 3 — Extension methodology.

Short narrative on the multi-task DNN's dual P̂ / Û output, the six
scoring schemes that combine them, and the no-trade band execution overlay.
Detail on individual models and schemes lives in the appendix pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Methodology & Extensions", page_icon=":bar_chart:",
                   layout="wide")

st.info(
    ":bulb: **New to this?** Read the [Background primer](Background) "
    "first — P̂ and Û are introduced there."
)

st.title("Methodology & Extensions")
st.caption(
    "What we kept from the paper, what we added, and how the new ranking "
    "rules combine direction with magnitude."
)

# --- What we kept ----------------------------------------------------------

st.markdown(
    """
**What we kept identical to the paper.** Same S&P 500 constituent idea with
month-end no-lookahead membership. Same 31 lagged-return features. Same
750-train / 250-trade rolling walk-forward. Same k = 10 long-short
construction at 5 bps / half-turn cost. Random seed 1, fixed across runs.

**What we added.** A second prediction target alongside the paper's
binary direction label: **Û = next-day return − next-day cross-sectional
median**. Each of the three model families now produces two outputs per
stock per day: **P̂ = P(stock beats the median)** and **Û = predicted
excess return vs the median**. RF and XGB get separate classifier and
regressor heads; the DNN becomes a **multi-task network with a shared
maxout trunk and two heads** trained jointly on `0.5·BCE + 0.5·Huber`.
Walk-forward extends ten new periods through 2025-09-24, taking total
out-of-sample coverage from 5,750 to 8,250 days.
    """
)

# --- The six scoring schemes ----------------------------------------------

st.subheader("Six scoring schemes")

st.markdown(
    """
Each scheme turns the (P̂, Û) pair into one rankable score per stock per
day. P-only is the paper baseline.

- **P-only** — rank by P̂ alone. Uses direction; ignores magnitude.
- **U-only** — rank by signed Û. Uses magnitude; ignores direction
  confidence.
- **Z-comp** — rank by `0.5·z(P̂) + 0.5·z(Û)`. Day-by-day cross-sectional
  z-scores; combines the two heads additively.
- **Product** — rank by `(2P̂ − 1) · Û`. Multiplicative combination —
  fails because of directional disagreement (see appendix).
- **P-gate(c)** — keep only stocks with `|P̂ − 0.5| ≥ c`, then rank by
  signed Û within each side. Tested at c = 0.03 and c = 0.05.
    """
)

st.subheader("No-trade band execution rule")

st.markdown(
    """
The no-trade band is an execution overlay, not a new forecasting model. It is
available only for schemes that use **Û**: U-only, Z-comp, P-gate(0.03), and
P-gate(0.05).

The baseline strategy rebuilds the top-k long and bottom-k short books every
day. The band instead asks whether a replacement is worth paying costs. With a
10 bps round-trip band, an incumbent long is replaced only when the candidate's
predicted excess return is at least 10 bps higher. On the short side, the new
candidate must be at least 10 bps lower in predicted excess return. If the
improvement is smaller than the round-trip cost, the old position stays.

This matters because high-turnover strategies can look attractive before costs
and then disappear after costs. In our materialized cost-band table the rule is
useful but not automatic: XGB improves consistently, RF is mixed, and ENS1 keeps
more return than many single models but does not universally gain Sharpe from
the band. That is why the Simulator exposes the band as a separate cost regime
rather than treating it as the default.
    """
)

st.divider()
st.markdown(
    "→ Per-model hyperparameters and ensemble formulas: "
    "**[Models explained](Models_Explained)** (appendix). "
    "Per-scheme deep-dive with the directional-disagreement scatter and "
    "live Û-magnitude histogram: **[Scoring schemes](Scoring_Schemes)** "
    "(appendix). End-to-end data flow: "
    "**[System Architecture](System_Architecture)**."
)
