"""Entry point for the Krauss ML stat-arb research demo app.

Streamlit auto-discovers the pages under ``app/pages/``.  The landing view
simply renders the Overview page content so the app opens on the headline
narrative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="Krauss ML stat-arb — research demo",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.data import data_build_is_complete, missing_build_warning  # noqa: E402

if not data_build_is_complete():
    missing_build_warning()

st.title("Overview")
st.caption(
    "A reproduction and extension of Krauss, Do & Huck (2017). "
    "This app summarises findings across the main, vix-regime-analysis and "
    "cost-modeling branches of the research repo."
)

st.markdown(
    """
Use the **sidebar** to navigate through the pages.

The deck is structured in nine sections; the appendix at the bottom (pages 9-12) holds deep-dive material referenced from the main flow.

**Main flow**

0. **Background** — plain-English primer on statistical arbitrage, Sharpe, k = 10, P̂ / Û, alpha decay. **Start here** if any of those terms are unfamiliar.
1. **Paper overview** — what Krauss et al. (2017) did, in two paragraphs.
2. **Reproduction** — how closely our re-runs match the paper's published numbers.
3. **Methodology** — what we changed: dual P̂ / Û output and five new scoring schemes.
4. **Results** — equity curve 1992-2025 with SPY overlay, plus headline KPIs and findings.
5. **Regime attempts** — three regime frameworks, all null. Why the alpha decay isn't a vol-regime artefact.
6. **Cost-aware execution** — turnover vs Sharpe, plus the 16-row no-trade-band matrix.
7. **Simulator** — pick a date window and parameters; replays historical daily returns into equity-curve and risk metrics.
8. **Conclusion** — what worked, what didn't, what to build next.

**Appendix**

9. **Pipeline** — Graphviz of the data flow with stage explanations.
10. **Models explained** — RF, XGB, DNN, multi-task DNN, cls+reg pairs, three ensembles.
11. **Scoring schemes** — interactive disagreement scatter and Û-magnitude histogram.
12. **Results matrix** — filterable grid of every (era, scheme, model, cost regime) we ran.

---

Every figure is historical. When the app says *"on 2008-10-10 the model said Y"*
that means Y is exactly the output of the fitted model for that date — not a
prediction for the future.
"""
)

st.info("All data shown here was pre-computed by `scripts/build_app_data.py`. "
        "No backtests run at page-render time.")
