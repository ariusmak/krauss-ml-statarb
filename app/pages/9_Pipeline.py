"""Page 2 — Pipeline.

Graphviz flow diagram with clickable nodes for each stage explanation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Pipeline", page_icon=":bar_chart:", layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning, load_pipeline_metadata,
)

if not data_build_is_complete():
    missing_build_warning()

meta = load_pipeline_metadata()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) explains the "
    "jargon (statistical arbitrage, k = 10, P̂/Û, basis points) before you "
    "dive into the pipeline."
)

st.caption(
    ":books: **Appendix** — deep-dive page. The main nine-section story is "
    "Background → Paper overview → Reproduction → Methodology → Results → "
    "Regimes → Cost-aware execution → Simulator → Conclusion."
)

st.title("Pipeline")
st.caption(
    "From raw vendor panel to a daily dollar-neutral long-short portfolio. "
    "Each node below is explained in the accompanying dropdown."
)

GRAPH = """
digraph pipeline {
    rankdir=LR;
    graph [splines=ortho, nodesep=0.4, ranksep=0.8, bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fillcolor="#f5f7fa",
           color="#4a5b6d", fontname="Helvetica", fontsize=11,
           margin="0.15,0.1"];
    edge  [color="#8a9aab", arrowsize=0.7];

    subgraph cluster_data {
        label="Data"; color="#c9d3dd"; fontname="Helvetica"; fontsize=12;
        crsp    [label="CRSP daily file\n(1992-2015)"];
        ds      [label="Datastream US\n(2015-2025)"];
        sp500   [label="S&P 500 index\n(VIX proxy)"];
    }

    universe [label="Universe build\nno-lookahead S&P 500\nmembership"];
    features [label="Features\n31 lagged returns\nR1, R2, ..., R240"];
    labels   [label="Labels\ny_binary (direction)\nu_excess (magnitude)"];

    subgraph cluster_models {
        label="Models"; color="#c9d3dd"; fontname="Helvetica"; fontsize=12;
        rf     [label="Random Forest"];
        xgb    [label="XGBoost"];
        dnn    [label="Feed-forward DNN"];
        mtdnn  [label="Multi-task DNN"];
        ens    [label="Ensembles\nENS1 / ENS2 / ENS3"];
    }

    subgraph cluster_portfolio {
        label="Portfolio construction"; color="#c9d3dd";
        fontname="Helvetica"; fontsize=12;
        score   [label="Scoring scheme\nP / U / Z-comp /\nProduct / P-gate"];
        rank    [label="Cross-sectional rank\ntop-k long\nbottom-k short"];
        port    [label="Equal-weight\ndollar-neutral"];
        costs   [label="Transaction costs\n5 bps / half-turn"];
    }

    crsp    -> universe;
    ds      -> universe;
    universe -> features -> rf;
    universe -> features -> xgb;
    universe -> features -> dnn;
    universe -> features -> mtdnn;
    universe -> labels   -> rf;
    labels   -> xgb;
    labels   -> dnn;
    labels   -> mtdnn;
    rf      -> ens;
    xgb     -> ens;
    dnn     -> ens;
    ens     -> score;
    mtdnn   -> score;
    sp500   -> score [style=dashed, label="regime overlay"];
    score   -> rank  -> port -> costs;
}
"""

st.graphviz_chart(GRAPH, use_container_width=True)

st.markdown("### Stage explanations")

with st.expander("Data sources"):
    st.markdown(
        f"""
- **CRSP daily file** — data source for Pipeline B's 1992-2015 era
  (Pipeline B's 2015-2025 extension uses Datastream; see the
  [Reproduction](Reproduction) page for the full two-pipeline breakdown).
  Coverage {meta['eras']['crsp']['start']} to {meta['eras']['crsp']['end']}.
  Source: {meta['data_sources']['crsp']}.
- **Datastream US** — data source for the 2015-2025 out-of-sample extension,
  and for the paper-conditions reproduction (Pipeline A) on the
  [Reproduction](Reproduction) page. Coverage {meta['eras']['extension']['start']} to
  {meta['eras']['extension']['end']}. Source: {meta['data_sources']['datastream']}.
- **S&P 500 index** — daily close used to derive VIX-style volatility
  regimes on the [Regimes](Regimes) page. Source: {meta['data_sources']['sp500_index']}.
        """
    )

with st.expander("Universe build"):
    st.markdown(
        """
For every trading day t we reconstruct the set of S&P 500 constituents
**as they were known at the close of day t-1** (point-in-time membership).
This avoids the classic survivor-bias trap where delisted names are dropped
from the panel.

Only constituents with at least 240 consecutive trading days of prior returns
are eligible — the longest feature lookback in the 31-feature set.
        """
    )

with st.expander("Features and labels"):
    st.markdown(
        """
- **Features**: 31 lagged return transforms — `R1, R2, ..., R20` (20 short-term
  daily lags) and `R40, R60, R80, R100, R120, R140, R160, R180, R200, R220, R240`
  (11 longer-horizon lags in 20-day steps), per Krauss et al. equation 1.
  Features are raw simple returns; no standardization or scaling is applied
  before training.
- **Label y_binary**: 1 if the stock's next-day return exceeds the
  cross-sectional median, 0 otherwise.
- **Label u_excess**: the stock's next-day return minus the cross-sectional
  median — the Phase-2 magnitude target.
        """
    )

with st.expander("Models"):
    st.markdown(
        """
See the **Models explained** page for the full treatment. Summary:

- **RF** — 1000 trees, depth 20.
- **XGB** — 500 trees, depth 6, lr 0.05.
- **DNN** — 31-31-10 feed-forward with dropout 0.1.
- **Multi-task DNN** — shared trunk with a classification head (direction)
  and a regression head (magnitude).
- **RF/XGB cls+reg pairs** — two independent models per algorithm: one
  classifier for `y_binary` and one regressor for `u_excess`, used to get
  both P_hat and U_hat from tree models.
- **ENS1 / ENS2 / ENS3** — simple means over model subsets. ENS1 =
  mean(RF, XGB, DNN); ENS2 = mean(RF, XGB); ENS3 = mean(DNN, XGB).
        """
    )

with st.expander("Scoring schemes"):
    st.markdown(
        """
Scoring turns model outputs into a single number per (stock, day) that we
rank on.

| Scheme | Formula | Intent |
|---|---|---|
| P-only | `P_hat` | Paper baseline: direction only |
| U-only | `U_hat` | Magnitude only |
| Z-comp | `0.5 z(P_hat) + 0.5 z(U_hat)` | Averaged cross-sectional z-scores |
| Product | `(2 P_hat - 1) * U_hat` | Signed magnitude |
| P-gate(c) | longs: `P_hat > 0.5 + c`, ranked by `U_hat` desc; shorts: `P_hat < 0.5 - c`, ranked by `U_hat` asc | Gate out uncertain predictions; rank confident candidates by signed U |
        """
    )

with st.expander("Portfolio construction and costs"):
    st.markdown(
        f"""
- **Ranking**: sort each day's scores descending; top-{meta['backtest']['k_long']}
  become the long leg, bottom-{meta['backtest']['k_short']} the short leg.
- **Weights**: equal-weight within each leg, dollar-neutral across legs.
- **Rebalance**: daily, close-to-close.
- **Costs**: `{meta['backtest']['cost_bps_per_half_turn']}` basis points per
  half-turn — the paper convention. A full round trip therefore costs
  `{2 * meta['backtest']['cost_bps_per_half_turn']}` bps on volume.

Cost regimes surfaced in the app:
- *no_cost* — gross returns, for like-for-like comparison with the paper's
  pre-cost tables.
- *5bps_half_turn* — the realistic cost assumption used in every
  post-cost claim.
        """
    )
