"""Page 0 — Background / Before you start.

A plain-English primer for readers who are new to statistical arbitrage,
long-short portfolios, or ML trading research.  No finance jargon goes
unexplained on this page; every other page assumes you've read it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Background", page_icon=":bar_chart:",
                   layout="wide")

st.title("Before you start")
st.caption(
    "A short primer for anyone who isn't already fluent in equity trading "
    "jargon. Six or seven minutes of reading will make the rest of the app "
    "much easier to follow."
)


def card(title: str, body: str) -> None:
    """Render one explainer card with a light border and consistent spacing."""
    with st.container(border=True):
        st.subheader(title)
        st.markdown(body)


# --- 1. Statistical arbitrage ----------------------------------------------
card(
    "1 — What is 'statistical arbitrage'?",
    """
Every trading day we rank a few hundred stocks by a model's score. We
**buy the 10 names the model says will do best** the next day, and
**short-sell the 10 names it says will do worst**. A short sale is a
borrowed-and-sold position that profits when the stock falls.

If the model is right on average, the longs go up a bit more than the
shorts, and the gap between them is our daily profit — regardless of
whether the whole market went up or down that day. That "regardless of
the market" property is the point of statistical arbitrage: we try to
harvest a tiny but *consistent* edge in relative performance, not a
directional bet on the S&P 500.
    """,
)

# --- 2. Dollar-neutral long-short ------------------------------------------
card(
    "2 — What does 'dollar-neutral long-short' mean?",
    """
For every **$1 we invest long** we put **$1 short**. Net market exposure
is zero, so a rally or a crash in the overall index leaves the book
essentially unchanged; only the spread between our longs and shorts
moves the P&L.
    """,
)

# Tiny bar visualisation of dollar-neutrality.
fig = go.Figure()
fig.add_trace(go.Bar(
    y=["Long leg", "Short leg", "Net exposure"],
    x=[1.0, -1.0, 0.0],
    orientation="h",
    marker_color=["#2e7d32", "#c62828", "#555"],
    text=["+$1.00", "−$1.00", "$0.00"],
    textposition="auto",
))
fig.update_layout(
    height=220,
    xaxis=dict(range=[-1.3, 1.3], title="Dollar exposure"),
    margin=dict(l=80, r=20, t=10, b=40),
    showlegend=False,
)
st.plotly_chart(fig, width="stretch")

# --- 3. Sharpe ratio --------------------------------------------------------
card(
    "3 — What is a Sharpe ratio?",
    """
Sharpe divides the **average return** by the **standard deviation of
return**. Higher = more return per unit of risk. A rough intuition
scale, annualised:

| Sharpe | What it means |
|---|---|
| **0** | Breakeven on a risk-adjusted basis |
| **0.5** | Useful but noisy |
| **1.0** | Good |
| **2.0** | Great |
| **3.0+** | Exceptional — or possibly overfit or otherwise too good to be true |

When we say *"Sharpe 1.92 post-cost"* we mean after subtracting trading
frictions, the strategy earned 1.92 units of annualised excess return
for every 1 unit of annualised volatility.
    """,
)

# --- 4. k = 10 --------------------------------------------------------------
card(
    "4 — What is k = 10?",
    """
`k` is how many stocks we hold on each side. **k = 10** means every
trading day we pick the top 10 predicted winners and the bottom 10
predicted losers. Smaller `k` concentrates the portfolio into the
highest-conviction names; larger `k` diversifies but dilutes the
signal. The paper reports k = 10 because that's the Sharpe-maximising
choice across the models they tested — we reproduce the same grid on
our data (see Results & Risk Diagnostics).
    """,
)

# --- 5. Post-cost -----------------------------------------------------------
card(
    "5 — What does 'post-cost' mean?",
    """
Every time we buy or sell, a real desk pays spread, commission, and
impact. The project uses the paper's convention: **5 basis points per
half-turn**, where a half-turn is either a buy or a sell. So a full
round trip (buy then later sell) costs **10 bps** on the traded volume.

A **basis point** is 1/100 of 1% (0.01%). 10 bps on $100 is $0.10.
That sounds trivial, but a high-turnover daily long-short strategy
typically trades ~200 % of its book per day, which compounds to a
meaningful drag. Every "post-cost" number in the app has that 5-bps
tax already subtracted.
    """,
)

# --- 6. P-hat and U-hat -----------------------------------------------------
card(
    "6 — What are P̂ and Û?",
    """
The ML models output two numbers per stock per day:

- **P̂** (P-hat) is the **probability** that the stock's next-day return
  will beat the day's **cross-sectional median** — that is, beat the
  median of all the other stocks in the universe on that same day.
  P̂ ≈ 0.5 means "no view"; P̂ > 0.5 means the model leans bullish
  relative to the median, P̂ < 0.5 means bearish.
- **Û** (U-hat) is the model's **point estimate of the excess return**
  the stock will post over the median. It's a signed magnitude —
  positive means predicted outperformance, negative means
  underperformance.

P̂ answers *"which side of the median?"*; Û answers *"by how much?"*
The scoring-schemes page (page 4) shows what happens when you combine
them different ways.
    """,
)

# --- 7. Alpha decay ---------------------------------------------------------
card(
    "7 — What does 'alpha decay' mean?",
    """
**Alpha** is risk-adjusted excess return — the bit of performance the
model generates *beyond* what you'd get from simple market exposure.
**Alpha decay** means an edge that used to work stops working, usually
because either (i) the market structure changes, (ii) other traders
spot the same signal and compete it away, or (iii) the training data
was a favourable sample.

In this project, pre-2015 the strategy earned a big post-cost Sharpe
(1.5–2.6 depending on scheme). Post-2015 the broad, unconditional rankings
mostly decay after costs. The selective P-gates can stay positive by trading
fewer days, but they no longer look like the broad daily edge in the paper.
That's alpha decay: similar recipe, harder market, much weaker signal. The
app's 2015-live extension is designed to make that deterioration visible
rather than hide it.
    """,
)

st.divider()
st.info(
    "With those seven ideas in hand, every other page of the app should "
    "read without extra jargon. If a term trips you up later, the shared "
    "terms are all defined here."
)
