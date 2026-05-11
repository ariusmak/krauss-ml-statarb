"""Page 2 — Reproduction results.

Two-pipeline framing:
  * Pipeline A (Datastream + H2O) is the apples-to-apples reproduction of the
    paper's setup. Numbers below come from notebooks/reproduction_h2o.ipynb
    and are documented in docs/reproduction_deviations.md Section 15.
  * Pipeline B (CRSP + PyTorch/sklearn/XGBoost) is the stack the project's
    extensions are built on. Numbers below come from summary_table.parquet
    and are documented in docs/reproduction_deviations.md Sections 1b, 6b, 8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Reproduction", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning, load_summary_table,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()

st.info(
    ":bulb: **New to this?** The [Background primer](Background) defines "
    "Sharpe, basis points, post-cost and the rest of the vocabulary used "
    "here."
)

st.title("Reproduction results")
st.caption(
    "How closely our re-runs match the paper's published numbers, broken "
    "out across two pipelines."
)

# --- Why two pipelines? ----------------------------------------------------

st.markdown(
    """
We ran the reproduction twice, on two stacks. **Pipeline A** uses the
paper's exact data source (Thomson Reuters Datastream) and ML framework
(H2O) — this is the apples-to-apples reproduction. **Pipeline B** uses
a Python stack (PyTorch / scikit-learn / XGBoost) throughout. Its data
source changes at the era boundary: **CRSP for the 1992-2015 era,
Datastream for the 2015-2025 extension.** The Python stack — not CRSP
per se — is what defines Pipeline B as separate from Pipeline A. We
built Pipeline B because Phase 2 work (the multi-task DNN, magnitude
regressors, and five scoring schemes on Page 3) requires libraries that
don't exist in H2O.

Every chart and table elsewhere in the app reads from Pipeline B.
Pipeline A appears only on this page, as the parity benchmark.
    """
)

# --- Pipeline A — paper-conditions reproduction ----------------------------

st.divider()
st.header("Pipeline A — paper-conditions reproduction")
st.caption(
    "Same data vendor (Datastream via WRDS) and same ML framework (H2O) "
    "as the paper. 23 walk-forward periods, k = 10, 5,750 OOS trading days."
)

# Pipeline A numbers come from docs/reproduction_deviations.md Section 15.
# These are hardcoded constants because Pipeline A is not part of the live
# data pipeline — its purpose is the parity statement, not the extensions.
PAPER_VALUES_TABLE2 = {
    "DNN":  0.0033,
    "GBT":  0.0037,
    "RAF":  0.0043,
    "ENS1": 0.0045,
}
PIPELINE_A_PRECOST = {
    "DNN":  0.0028,
    "GBT":  0.0039,
    "RAF":  0.0040,
    "ENS1": 0.0042,
}

st.subheader("Daily return parity — pre-cost, k = 10")

rows_a = []
for m, paper_val in PAPER_VALUES_TABLE2.items():
    ours = PIPELINE_A_PRECOST[m]
    rows_a.append({
        "Model": m,
        "Paper (Table 2)": f"{paper_val * 100:.2f}%",
        "Pipeline A":      f"{ours * 100:.2f}%",
        "Ratio":           f"{ours / paper_val * 100:.0f}%",
    })

st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)

st.markdown(
    """
**ENS1 reproduces at 94 % of the paper's published pre-cost daily return.**
GBT and RAF land within 7 % of their published values; DNN at 85 % is
the weakest base learner, which is consistent with the paper's own
observation that the DNN is the hardest model to configure (Krauss
Section 5.2, p. 697).

Annualised post-cost Sharpe on Pipeline A is **2.23** versus the paper's
**1.81**. The Sharpe gap in our favour comes from lower realised
turnover — ~2.5 sides rebalanced per day versus the ~4.0 implied by the
paper's cost arithmetic — so the same 5 bps / half-turn convention bites
less hard on our runs than on the paper's. Possible cause: H2O has
shipped ten years of internal changes between the paper's version
(~v3.8-3.10, 2016) and our v3.46, and the prediction noise profile is
plausibly lower in the newer release.
    """
)

# --- Pipeline A sub-period detail -------------------------------------------

with st.expander("Sub-period parity (Pipeline A vs paper Table 5)"):
    sub_period_table = pd.DataFrame(
        [
            ("12/92–03/01 (pre-popularization)", "233.5%", "140.3%", "60%"),
            ("04/01–08/08 (moderation)",         "22.3%",  "55.3%",  "248%"),
            ("09/08–12/09 (financial crisis)",   "405.2%", "115.6%", "29%"),
            ("01/10–10/15 (deterioration)",      "-18.0%", "-1.5%",  "8%"),
        ],
        columns=["Sub-period", "Paper ann. ret", "Pipeline A ann. ret",
                  "Ratio"],
    )
    st.dataframe(sub_period_table, use_container_width=True, hide_index=True)
    st.markdown(
        """
The reproduction holds in direction across every sub-period. The two
gaps worth noting are the crisis sub-period (29 %) and the late
deterioration sub-period (8 %). Both are driven by the same mechanic:
when absolute returns are extreme or near zero, small differences in
which exact top-10 / bottom-10 stocks get picked on a handful of
high-impact days compound into large percentage gaps. The paper's
~405 % crisis return is largely earned in October 2008 alone (Krauss
reports >100 % in that single month, Section 5.3) — so the 29 % ratio
is best read as "we capture the regime, not the path."
        """
    )

st.caption(
    "Sources: paper values from Krauss, Do & Huck (2017) Tables 2 and 5; "
    "Pipeline A values from `docs/reproduction_deviations.md` Section 15."
)

# --- Pipeline B — extension pipeline ---------------------------------------

st.divider()
st.header("Pipeline B — extension pipeline")
st.caption(
    "Python stack throughout. The CRSP-era half loads from CRSP feeds; "
    "the extension-era half loads from Datastream feeds. All app pages "
    "other than this one read from Pipeline B's data. Two known sources "
    "of deviation from the paper's CRSP-era results: (i) CRSP identifies "
    "614 more 'ever-member' stocks than the paper's Datastream "
    "constituency, which dilutes the training signal; (ii) the DNN is "
    "ported from H2O to PyTorch, which introduces a documented ADADELTA "
    "batch-size constraint (see docs/reproduction_deviations.md Section 8)."
)

st.subheader("Daily return parity — pre-cost, k = 10")

DISPLAY_NAME = {"DNN": "DNN", "XGB": "GBT", "RF": "RAF", "ENS1": "ENS1"}
rows_b = []
for model_code, paper_label in DISPLAY_NAME.items():
    sub = summary.query(
        "model == @model_code and scheme == 'P-only' "
        "and era == '1992-2015 (CRSP)' and cost_regime == 'no_cost'"
    )
    if sub.empty:
        continue
    ours = float(sub["full_sample_return"].iloc[0])
    paper_val = PAPER_VALUES_TABLE2[paper_label]
    rows_b.append({
        "Model": paper_label,
        "Paper (Table 2)": f"{paper_val * 100:.2f}%",
        "Pipeline B":      f"{ours * 100:.2f}%",
        "Ratio":           f"{ours / paper_val * 100:.0f}%",
    })

st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)

st.markdown(
    """
**ENS1 on Pipeline B reproduces at 90 % of paper.** GBT and RAF are
within ~5 % of their published values — comparable to Pipeline A. The
gap is concentrated in the **DNN at 63 %**.

We characterise the DNN gap precisely in
`docs/reproduction_deviations.md` Section 8. Short version: H2O's
ADADELTA implementation runs with `mini_batch_size = 1` (pure SGD) and
internal Java native gradient handling. PyTorch's `torch.optim.Adadelta`
diverges or collapses at small batch sizes — we ran the full grid
(batch ∈ {1, 32, 1024}, lr ∈ {0.005, 1.0}) and `batch_size = 1024,
lr = 1.0` is the only configuration that produces stable, meaningful
gradients. The cost of that constraint is reduced prediction spread and
a weaker DNN, which propagates into the ENS1 mean.

This is an *inherent* framework deviation — fixing it would require
porting the multi-task DNN back to H2O, which would break Phase 2.
We chose to keep the extensions on PyTorch and disclose the DNN gap.

Post-cost ENS1 Sharpe on Pipeline B is **2.18** (similar to Pipeline A's
2.23), versus the paper's **1.81**. The same low-turnover mechanic from
Pipeline A applies here.
    """
)

# --- What this means for the rest of the app -------------------------------

st.divider()
st.header("What this means for the rest of the app")

st.markdown(
    """
- The **headline parity number is 94 %** — ENS1 pre-cost on Pipeline A,
  the apples-to-apples reproduction.
- **Every chart and table from Page 3 onwards reads from Pipeline B**
  (`summary_table.parquet` includes both eras of Pipeline B — CRSP-sourced
  1992-2015 and Datastream-sourced 2015-2025).
- The 4-percentage-point gap between Pipeline A and B is split roughly
  equally between (i) CRSP vs Datastream constituent differences and
  (ii) the DNN's PyTorch ADADELTA constraint.
- The DNN under-reproduction in Pipeline B is the most important
  caveat to keep in mind when reading the extension results. ENS1, RF
  and XGB are well-reproduced; the multi-task DNN inherits the
  Phase-1 DNN's framework constraints and shouldn't be expected to
  recover them.
- The **multi-task DNN was trained on CRSP only**, not on the
  2015-2025 extension. It does not appear in any post-2015 chart in
  the app.
    """
)
