"""Page 3 — Models explained.

Seven model families with hyperparameters copied verbatim from the source of
truth files inside the repo. No plausible-sounding defaults — every value
shown here has a file:line reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="Models explained", page_icon=":bar_chart:",
                   layout="wide")

from lib.data import (  # noqa: E402
    data_build_is_complete, missing_build_warning,
    load_summary_table, load_daily_holdings, load_equity_curves,
)

if not data_build_is_complete():
    missing_build_warning()

summary = load_summary_table()
equity = load_equity_curves()
holdings = load_daily_holdings()

st.title("Models explained")
st.caption(
    "Seven model families. All hyperparameters below were copied verbatim from "
    "the repo — each section cites the source file and line range. Values not "
    "present in the code are marked 'not part of the headline run'."
)

# Shared source-of-truth hyperparameter tables ------------------------------

HYPERPARAMS = {
    "RF": dict(
        source="src/krauss/models/rf_phase1.py:29-49",
        rows=[
            ("n_estimators", 1000),
            ("max_depth", 20),
            ("max_features", "floor(sqrt(31)) = 5"),
            ("bootstrap", True),
            ("max_samples", 0.632),
            ("criterion", "entropy"),
            ("random_state", 1),
            ("n_jobs", "-1"),
        ],
    ),
    "XGB": dict(
        source="src/krauss/models/xgb_phase1.py:32-51",
        rows=[
            ("n_estimators", 100),
            ("max_depth", 3),
            ("learning_rate", 0.1),
            ("colsample_bynode", "15/31 ≈ 0.484"),
            ("min_child_weight", 10),
            ("reg_lambda", 0),
            ("reg_alpha", 0),
            ("gamma", 1e-5),
            ("max_bin", 20),
            ("objective", "binary:logistic"),
            ("eval_metric", "logloss"),
        ],
    ),
    "DNN": dict(
        source="src/krauss/models/dnn_phase1.py:63-304",
        rows=[
            ("architecture", "31 → Maxout(31) → Maxout(10) → Maxout(5) → Linear(2)"),
            ("maxout_channels", 2),
            ("dropout_input", 0.1),
            ("dropout_hidden", 0.5),
            ("optimizer", "Adadelta(lr=1.0, rho=0.99, eps=1e-8)"),
            ("loss", "CrossEntropyLoss (2-class softmax)"),
            ("epochs (max)", 400),
            ("batch_size", 1024),
            ("l1_lambda", 1e-5),
            ("score_every_n_samples", 750_000),
            ("early-stop window / patience", "5 / 5"),
        ],
    ),
    "MT-DNN": dict(
        source="src/krauss/models/dnn_multitask.py:55-275",
        rows=[
            ("shared trunk", "31 → Maxout(31) → Maxout(10) → Maxout(5)"),
            ("classification head", "Linear(5, 1) + sigmoid"),
            ("regression head", "Linear(5, 1)"),
            ("loss", "0.5 · BCE + 0.5 · Huber"),
            ("other training schedule",
              "identical to Phase-1 DNN (Adadelta, L1 1e-5, batch 1024)"),
            ("trained on",
              "CRSP Phase 2 only — not trained on the Datastream extension"),
        ],
    ),
    "RF cls+reg": dict(
        source="src/krauss/models/rf_extension.py:22-49",
        rows=[
            ("classifier", "identical to RF Phase 1 above"),
            ("regressor", "same tree params, criterion = 'squared_error'"),
            ("target cls/reg", "y_binary / u_excess"),
        ],
    ),
    "XGB cls+reg": dict(
        source="src/krauss/models/xgb_extension.py:22-58",
        rows=[
            ("classifier", "identical to XGB Phase 1 above"),
            ("regressor", "same params, objective = 'reg:pseudohubererror'"),
            ("target cls/reg", "y_binary / u_excess"),
        ],
    ),
    "ENS1": dict(
        source="src/krauss/models/ensembles_phase1.py:14-24 (Phase 1);  "
                "ensembles_phase2.py:16-32 (Phase 2)",
        rows=[
            ("Phase 1", "P = (p_dnn + p_xgb + p_rf) / 3"),
            ("Phase 2",
              "P = (p_dnn + p_xgb + p_rf) / 3;  U = (u_dnn + u_xgb + u_rf) / 3"),
            ("weights", "1/3, 1/3, 1/3"),
        ],
    ),
    "ENS2": dict(
        source="src/krauss/models/ensembles_phase1.py:37-70",
        rows=[
            ("composition", "DNN + XGB + RF, Phase 1 only"),
            ("weights",
              "proportional to training-period Gini = 2·AUC − 1"),
            ("fallback", "reverts to ENS1 if all Gini ≤ 0"),
        ],
    ),
    "ENS3": dict(
        source="src/krauss/models/ensembles_phase1.py:73-101",
        rows=[
            ("composition", "DNN + XGB + RF, Phase 1 only"),
            ("weights",
              "w_i = (1/R_i) / Σ_j(1/R_j), R_i = Gini rank, 1 = best"),
            ("citation", "Aiolfi & Timmermann (2006) harmonic-rank weighting"),
        ],
    ),
}


def _hp_table(key: str) -> None:
    spec = HYPERPARAMS[key]
    st.caption(f"Source: `{spec['source']}`")
    df = pd.DataFrame(spec["rows"], columns=["Hyperparameter", "Value"])
    df["Value"] = df["Value"].astype(str)
    st.dataframe(df, width="stretch", hide_index=True)


def _oos_block(model: str, schemes_to_show: list[str] | None = None) -> None:
    """Out-of-sample panel for a single model across eras and cost regimes."""
    sub = summary[summary["model"] == model]
    if schemes_to_show:
        sub = sub[sub["scheme"].isin(schemes_to_show)]
    if sub.empty:
        st.info(f"No post-run rows for {model} in the app data.")
        return
    cols = ["era", "cost_regime", "scheme", "daily_return", "sharpe",
            "matched_days_return", "avg_turnover", "trading_days"]
    view = (sub[cols]
             .sort_values(["era", "cost_regime", "scheme"])
             .reset_index(drop=True))
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "daily_return": st.column_config.NumberColumn(format="%.4f"),
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "matched_days_return": st.column_config.NumberColumn(format="%.4f"),
            "avg_turnover": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _example_top5(model: str, scheme: str = "P-only") -> None:
    """Show a concrete example day's top-5 longs and their next-day returns.

    Uses the daily holdings table; limits to a date where the model actually
    had a clear signal (we pick a sample date inside a known crisis, 2008-10-13,
    falling back to the first available date if the exact one is missing).
    """
    if holdings is None or holdings.empty:
        st.caption("Daily holdings were not produced — skipping concrete example.")
        return

    targets = [pd.Timestamp("2008-10-13"),
               pd.Timestamp("2020-03-16"),
               pd.Timestamp("2009-03-09")]
    sub = holdings[(holdings["scheme"] == scheme)]
    sub = sub[sub["side"] == "long"]
    if sub.empty:
        return
    for d in targets:
        day_rows = sub[sub["date"] == d]
        if not day_rows.empty:
            break
    else:
        d = sub["date"].min()
        day_rows = sub[sub["date"] == d]

    top5 = day_rows.nsmallest(5, "rank")[[
        "date", "rank", "stock_id", "p_hat", "u_hat", "next_day_ret",
    ]]
    st.caption(
        f"Concrete example — top-5 longs on {d.date()} under {scheme} "
        "scoring (ENS1 demo scheme, period covering that date)."
    )
    st.dataframe(top5, width="stretch", hide_index=True,
                 column_config={
                     "p_hat": st.column_config.NumberColumn(format="%.3f"),
                     "u_hat": st.column_config.NumberColumn(format="%.5f"),
                     "next_day_ret": st.column_config.NumberColumn(format="%.4f"),
                 })


# Section: RF ----------------------------------------------------------------
st.header("1 — Random Forest (RF)")
st.markdown(
    """
The Phase-1 direction classifier. A bag of 1,000 deep trees trained on the
31-feature lagged-return panel to predict whether each stock's next-day
return will be above or below the cross-sectional median. Entropy split
criterion (matching the H2O DRF reproduction) and per-tree 63.2 % bootstrap
sampling keep the model close to the Krauss paper's tree specification.

Strengths. High-capacity, stable under walk-forward retraining, tolerant of
the standardised return features. The sqrt feature subsampling at each split
handles the correlation between adjacent return lags.

Weaknesses. Calibration drifts. Probability outputs cluster away from 0 and
1 because deep trees vote over discrete splits, so the |P − 0.5| scale is
not comparable across training windows, which is one reason the P-gate
schemes behave unevenly when switched between RF and XGB.
"""
)
_hp_table("RF")
_oos_block("RF", schemes_to_show=["P-only", "U-only", "Z-comp"])

# Section: XGB ---------------------------------------------------------------
st.header("2 — XGBoost (XGB)")
st.markdown(
    """
Shallow boosted trees. 100 rounds, depth 3, learning rate 0.1, with
column-subsampling-per-split of 15/31. The model intentionally matches the
H2O GBM configuration from the paper rather than a modern tuned XGBoost —
the goal is reproduction, not raw accuracy.

Strengths. Faster to train than the RF and the DNN; calibration is noticeably
cleaner so the P-gate family works more predictably on XGB than on RF.

Weaknesses. Under-regularised on long training windows: variance across
walk-forward periods is the highest of the three base learners.
"""
)
_hp_table("XGB")
_oos_block("XGB", schemes_to_show=["P-only", "U-only", "Z-comp"])

# Section: DNN ---------------------------------------------------------------
st.header("3 — Feed-forward DNN")
st.markdown(
    """
The paper's direction classifier. A narrow 31 → 31 → 10 → 5 → 2 maxout
network with heavy dropout (0.1 input, 0.5 hidden) and ADADELTA — again,
this is a faithful reproduction of the H2O DeepLearning pipeline rather
than a modern design. Training is capped at 400 epochs but is dominated
by the early-stopping rule, which watches a 5-sample moving average of
the validation score with patience 5.

Strengths. Lowest-variance of the three base learners. The L1 penalty
(1e-5) keeps the feature importances spread, which is what gives ENS1
its diversification benefit.

Weaknesses. Very slow to train (dominant cost in the Phase-1 pipeline).
Maxout activation without batch-norm means hidden units occasionally
collapse to one channel, which shows up as dead features in the
training-set Gini.
"""
)
_hp_table("DNN")
_oos_block("DNN", schemes_to_show=["P-only", "U-only", "Z-comp"])

# Section: MT-DNN ------------------------------------------------------------
st.header("4 — Multi-task DNN (MT-DNN)")
st.markdown(
    """
A Phase-2 extension. Same shared trunk as the Phase-1 DNN, with two output
heads: a classification head trained with binary cross-entropy against
y_binary, and a regression head trained with Huber loss against u_excess.
The loss is 0.5·BCE + 0.5·Huber — not tuned, just equal weight.

Strengths. Unifies P and U into one fit, saving roughly half the compute
versus the separate cls+reg pair.

Weaknesses. On CRSP the single shared trunk under-fits the regression
head relative to a dedicated regressor, so in the results matrix the
classic RF/XGB cls+reg pair tends to beat MT-DNN on schemes that lean on U
(Z-comp, the gated family). And crucially — it was not trained on the
Datastream extension: MT-DNN is not part of the post-2015 headline run.
"""
)
_hp_table("MT-DNN")
st.info(
    "No MT-DNN rows appear in the results matrix or trading demo. "
    "The `data/models_p2_ds/period_*/` directories omit `mt_dnn.pt`."
)

# Section: RF / XGB cls+reg pairs --------------------------------------------
st.header("5 — RF and XGB classifier + regressor pairs")
st.markdown(
    """
Phase 2. Two parallel fits per algorithm: the usual classifier predicts
P_hat (y_binary), a sibling regressor predicts U_hat (u_excess). The
classifier hyperparameters are identical to the Phase-1 RF/XGB above; the
regressor switches to the correct loss but otherwise inherits every other
setting.

Strengths. Because the heads are independent they don't compete for
capacity, so U_hat is sharper here than under the shared-trunk MT-DNN.

Weaknesses. Double the training cost. And P_hat and U_hat are calibrated
on different loss surfaces, which is part of the story behind the
product-composite failure on the scoring-schemes page.
"""
)
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("RF cls + reg")
    _hp_table("RF cls+reg")
with col_b:
    st.subheader("XGB cls + reg")
    _hp_table("XGB cls+reg")
_oos_block("RF", schemes_to_show=["P-gate(0.03)", "P-gate(0.05)"])

# Section: ENS1 --------------------------------------------------------------
st.header("6 — ENS1")
st.markdown(
    """
ENS1 is the plain arithmetic mean of the three base learners. It is the
ensemble used in every headline chart in the project because its
behaviour is easy to reason about: any improvement comes from averaging
away idiosyncratic errors, not from weighting schemes that can leak
test-set information.

The Phase-2 ENS1 averages both P_hat and U_hat across the same three
models, so the Z-score composite and P-gate schemes always ride on
equally-weighted signals.
"""
)
_hp_table("ENS1")
_oos_block("ENS1")
_example_top5("ENS1")

# Section: ENS2 + ENS3 -------------------------------------------------------
st.header("7 — ENS2 and ENS3")
st.markdown(
    """
Two weighted variants of ENS1. Both reweight the three base learners by
training-period Gini coefficient. ENS2 uses the Gini itself as the weight,
with a fallback to ENS1 when every Gini is non-positive. ENS3 uses the
inverse-rank (1/R) harmonic weighting from Aiolfi & Timmermann (2006),
which dampens the best-model weight compared to ENS2.

Both are Phase-1 only — they are computed on direction probabilities
and do not have a magnitude (U) equivalent in the codebase. In
practice ENS2 and ENS3 track ENS1 closely on the CRSP era because the
three base Ginis are usually within a few percentage points of each
other, so the weighting has limited effect.
"""
)
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("ENS2 — Gini-weighted")
    _hp_table("ENS2")
with col_b:
    st.subheader("ENS3 — Rank-weighted")
    _hp_table("ENS3")

col_a, col_b = st.columns(2)
with col_a:
    st.caption("ENS2 results")
    _oos_block("ENS2", schemes_to_show=["P-only"])
with col_b:
    st.caption("ENS3 results")
    _oos_block("ENS3", schemes_to_show=["P-only"])

# Closing note about missing pieces ------------------------------------------
st.divider()
st.caption(
    "**Reading this page.** Every hyperparameter table above cites a specific "
    "file and line range. Where a value is missing from the repo (for example, "
    "MT-DNN is not run on the Datastream extension) the page says so explicitly "
    "rather than inferring a plausible default."
)
