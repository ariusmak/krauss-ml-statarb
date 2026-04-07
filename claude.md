# CLAUDE.md — Krauss Project Agent Specification

## 0) Purpose

You are helping implement a three-phase algorithmic trading project based on:

1. **Krauss et al. (2017)** — the source of truth for the original methodology.
2. **The project deck** — the motivation for the extensions, but **not** the source of truth when it conflicts with the paper.
3. **The IEOR 4733 course guidelines** — the source of truth for deliverables, system requirements, and evaluation expectations.

This project is **not** “build a cool ML strategy inspired by Krauss.”
It is:

- **Phase 1:** reproduce the original paper as faithfully as possible in Python using WRDS-accessible data,
- **Phase 2:** implement the agreed extensions on top of the reproduction pipeline,
- **Phase 3:** build a proper sequential trading simulation / deployed research system that satisfies the course requirements.

The top-level principle is:

> **Parity first, extension second, systemization third.**

Do not reorder those priorities.

---

## 1) Non-negotiable working rules

### 1.1 Never make weak assumptions
If a material implementation choice is unclear, ambiguous, or has multiple plausible interpretations, **stop and ask the user** before proceeding.

Examples:
- ambiguous WRDS table choice,
- ambiguous identifier mapping,
- paper-vs-library mismatch that materially changes methodology,
- unresolved regime thresholds or portfolio sizing map,
- any design change that would materially affect parity or evaluation.

Do **not** silently “pick something reasonable.”

### 1.2 Use the paper as source of truth when paper and deck differ
The deck is a planning document. The paper is the reproduction target.

Examples of likely conflicts:
- the deck’s high-level parameter summaries may simplify or slightly misstate the paper,
- the deck’s `P * U` composite is known to be flawed and has been replaced by a clarified formula.

### 1.3 User preference: consult before branching
The user explicitly prefers that **implicit decisions are not made without consulting**.
If you hit a decision fork, present the options and ask.

### 1.4 User workflow preference
- Put reusable code in a `utils/` folder.
- One-off data-building or exploratory scripts may live in notebooks, but the final project must **not** depend on notebooks for core functionality.

### 1.5 Do not over-optimize early
Before Phase 1 parity is working, do **not**:
- launch wide hyperparameter searches,
- redesign models for convenience,
- add extra factors/features,
- add new data sources beyond what is needed for parity,
- change execution assumptions “for realism.”

First get a faithful baseline.

---

## 2) Locked project decisions

These decisions are already made and should be treated as fixed unless the user explicitly changes them.

### 2.1 Stack
- Language: **Python**
- Tree boosting: **XGBoost**
- Random forest: **scikit-learn**
- DNN / multi-task DNN: **PyTorch**
- Data access: **WRDS**
- App/UI: **Streamlit**
- Reusable helpers: `utils/`

### 2.2 Data and reproduction philosophy
- The original paper uses Datastream data.
- This project will use **WRDS-accessible data** and aim for the **closest possible methodological/data parity**.
- All reproduction differences vs the paper must be logged explicitly.

### 2.3 No-lookahead universe construction
Use the **end-of-month S&P 500 constituency list to define the universe for the next month**.

This is a hard rule.
Do not use future membership information.

### 2.4 Phase structure
- **Phase 1:** reproduce original models/results.
- **Phase 2:** implement extensions on the same core pipeline and compare fairly on the same data / framework.
- **Phase 3:** build a sequential trading simulator / deployed research system.

### 2.5 Extension target definitions
Both extension targets are defined **relative to the next-day cross-sectional median**:
- `P_t = P(U_t > 0)`
- `U_t = next_day_stock_return - next_day_cross_sectional_median_return`

### 2.6 DNN extension architecture
- The extension DNN is **shared trunk + two heads**
- One classification head for `P_t`
- One regression head for `U_t`

### 2.7 DNN extension loss
Use a single joint objective:
- classification loss: binary / cross-entropy-type loss on sign of `U_t`
- regression loss: **Huber**
- combined as a weighted sum

Do not invent a more exotic unified loss unless the user explicitly asks.

### 2.8 Composite score for the extension
The deck’s `P_t * U_t` composite is **not** to be used.

Use:
- **Default composite:** `(2 * P_t - 1) * U_t`

Do not silently switch to the piecewise alternative unless the user asks for that as an ablation.

### 2.9 No-trade band
Apply the no-trade band **only** to strategies that explicitly predict `U_t`:
- `U`-only strategies,
- composite-score strategies.

Do **not** apply the no-trade band to the `P`-only baseline unless the user explicitly asks for a separate hysteresis design.

### 2.10 Regime adaptation
- Regime adaptation method is **VIX-only**
- Do **not** use HMMs

Important: the exact VIX thresholds and exact `(k_long, k_short)` mapping are **not fully locked**. If implementation reaches that point, ask the user before hard-coding the thresholds.

### 2.11 Extension ensemble scope
For Phase 2, implement **ENS1 only** for the extension.
Do not extend ENS2/ENS3 to the new multi-objective framework unless the user explicitly asks.

### 2.12 Phase 3 simulation philosophy
Phase 3 must be a **sequential walk-forward trading simulator**, not just a passive backtest dashboard.

However:
- the **baseline retraining schedule** should remain aligned with the paper’s block structure,
- “true online learning” is **not** assumed to be required by default.

If you want to add more frequent refits / quasi-online updates later, ask first.

### 2.13 Execution convention for parity
For the headline paper-parity reproduction, use the paper-style **daily close-based alignment**, i.e. the next-day daily return convention implied by the paper, not an invented next-open execution convention.

If you want to add an optional “realism mode” with next-open execution, ask before making it the default.

---

## 3) High-level spirit of the project

The spirit of the project is:

1. **Faithful reproduction**
   - Understand what the paper actually did.
   - Build the closest possible Python/WRDS analogue.
   - Measure and explain differences.

2. **Economically motivated extension**
   - The extension exists because the paper trains on a binary objective but monetizes ranked tail opportunities.
   - The extension should therefore align scoring more closely with expected trading opportunity, not just add complexity.

3. **Production-level research system**
   - The final system must be runnable, modular, reproducible, and sequential.
   - The project guidelines explicitly reject notebook-only work.

Always preserve this spirit.

---

## 4) Source hierarchy

When implementing, use this hierarchy:

### Source of truth #1: Krauss paper
Use the paper for:
- methodology,
- training/trading structure,
- feature definitions,
- model architecture/parameters,
- ensemble definitions,
- trading rule,
- evaluation metrics,
- subperiod analysis,
- robustness/sensitivity checks.

### Source of truth #2: user-locked decisions
Use the user discussion for:
- Python stack,
- WRDS usage,
- clarified composite score,
- no-trade band scope,
- ENS1-only extension,
- VIX-only regime logic,
- simulator direction,
- ask-before-assume rule.

### Source of truth #3: course guidelines
Use the guidelines for:
- required deliverables,
- reproducibility requirements,
- no-lookahead requirement,
- realistic execution assumptions,
- need for backtest engine + dashboard + ability to run simulations.

### Source of truth #4: project deck
Use the deck mainly for:
- motivation,
- extension intent,
- high-level milestone framing.

If the deck disagrees with the paper, **do not follow the deck blindly**.

---

## 5) Phase 1 — Reproduce the original paper

## 5.1 Goal
Build the closest possible Python/WRDS replication of Krauss et al. (2017).

This includes:
- data construction,
- feature generation,
- original classification models,
- original ensembles,
- original ranking/trading rule,
- core evaluation tables/figures.

## 5.2 Required parity targets from the paper

You should aim to reproduce, as closely as possible, the **pattern** and **magnitude** of the original findings:

- base models: DNN, GBT, RAF
- ensembles: ENS1, ENS2, ENS3
- baseline trading rule: long top `k`, short bottom `k`, dollar neutral, daily rebalancing
- key headline result for `k=10`:
  - ENS1 around **0.45%/day pre-cost**
  - ENS1 around **0.25%/day post-cost**
  - annualized Sharpe around **1.81 post-cost**
- returns decline after 2001 and spike during crisis periods
- RAF strongest individual base model
- ENS variants similar, ENS1 hard to beat
- post-cost profitability deteriorates in later years

Do not treat exact numerical parity as guaranteed, because:
- Datastream != WRDS
- H2O models != Python/XGBoost/PyTorch clones
- implementation details may differ

But do treat these as the reproduction target.

## 5.3 Data construction requirements

### 5.3.1 Universe
Use historical S&P 500 membership with **month-end membership deciding the next month’s eligible universe**.

Do not use current constituents.
Do not use survivor-only lists.

### 5.3.2 WRDS handling
Use WRDS-accessible historical data.
However:

- **Do not assume exact WRDS table names or column names without verifying them.**
- Different WRDS accounts/views can expose slightly different schemas.
- If table names or join paths are unclear, inspect or ask.

### 5.3.3 Canonical security identifier
Preferred canonical daily-security key should be the one that is most stable in the daily return data source (typically a security-level identifier such as PERMNO if available).

But:
- if membership comes at a different identifier level,
- or mapping to multiple securities/share classes is ambiguous,
- or the join logic is materially unclear,

**stop and ask the user**.

### 5.3.4 Total-return consistency
The paper uses daily total return indices.
Your WRDS/Python analogue should be as close as possible to a total-return-consistent daily return series.

Do not just use raw close prices without checking return consistency.

### 5.3.5 Delistings / corporate actions
Handle these carefully.
Do not hand-wave them away.
If the chosen WRDS dataset requires special treatment of delisting returns or corporate-action-adjusted fields, implement it explicitly.

If the correct handling is unclear in the chosen schema, ask.

## 5.4 Study-period logic

The paper’s logic:
- build study periods from a large daily panel,
- each study period consists of:
  - **750 trading days train**
  - **250 trading days trade**
- the first **240 days** are consumed by feature lookbacks
- effectively:
  - training window contributes about **510 daily training observations per stock**
  - trading window remains **250 daily observations**

You must preserve this logic.

### Important ambiguity guard
There is potential tension between:
- monthly membership-driven universe eligibility,
- and the paper’s “study-period batch” language.

Implement the membership matrix first.
If the exact per-row/per-period eligibility logic remains ambiguous after inspection, ask the user before hard-coding a fixed interpretation.

## 5.5 Feature generation

Use the paper’s **31 lagged return features** exactly:

- `R(1)` to `R(20)` daily returns
- `R(40), R(60), ..., R(240)` multi-period returns

Definitions:
- simple return over `m` periods:
  - `R_t,m = P_t / P_(t-m) - 1`
- features are built from daily prices / return-consistent series

Do not add extra predictors in Phase 1.

## 5.6 Original target definition

Binary response:
- `Y_{i,t+1} = 1` if stock `i`’s next-day return exceeds the next-day **cross-sectional median return**
- otherwise `0`

This is the original paper’s classification target.

## 5.7 Original models

### 5.7.1 DNN parity target
Paper target architecture:
- `31-31-10-5-2`
- maxout hidden activations
- dropout:
  - hidden dropout ratio `0.5`
  - input dropout ratio `0.1`
- slight L1 regularization: `1e-5`
- ADADELTA optimizer
- up to `400` epochs
- early stopping
- seed fixed to `1`
- deterministic behavior where feasible

#### Important parity requirement
Do not replace the paper’s DNN with a generic MLP and call it reproduced.
If an exact PyTorch maxout translation is ambiguous, ask.

Preferred interpretation:
- preserve the paper’s effective maxout-layer logic rather than collapsing to ReLU.

#### Output format
For Phase 1 parity, prefer a **two-class softmax output** to mirror the paper rather than silently switching to a single-logit sigmoid.

If you want to simplify this, ask first.

### 5.7.2 GBT parity target
The paper’s tree model is H2O-based, not XGBoost.
In Python, use XGBoost as the closest agreed analogue, but explicitly document this as a reproduction deviation.

Do not silently trust the deck’s parameter summary if it conflicts with the paper.

Important:
- the deck simplifies the tree description,
- the paper should be used to choose the intended depth/interaction structure.

If there is uncertainty in mapping H2O GBM/AdaBoost semantics to XGBoost parameters, ask before finalizing.

### 5.7.3 RAF parity target
Use sklearn random forest as the agreed Python analogue.

Paper target:
- `1000` trees
- deep trees (paper discusses max depth around `20`)
- feature subsampling around `sqrt(p)`
- seed fixed

Do not aggressively tune Phase 1 RF before baseline parity is complete.

## 5.8 Original ensembles

### ENS1
Equal-weight average of DNN / GBT / RAF probabilities.

### ENS2
Training-period performance-based weighting using Gini/AUC-derived weights.

### ENS3
Training-period rank-based weighting using Gini/AUC ranks.

These belong in **Phase 1** reproduction.

## 5.9 Trading rule

For each day:
1. generate one-day-ahead probability forecasts,
2. rank stocks cross-sectionally descending,
3. long top `k`,
4. short bottom `k`,
5. equal-weight within each side,
6. dollar neutral,
7. daily rebalance,
8. transaction costs at **5 bps per half-turn**.

### Phase 1 `k` values
At minimum, implement:
- `k = 10, 50, 100, 150, 200`

The paper reports general performance across that grid and then focuses heavily on `k=10`.

## 5.10 Evaluation requirements

Implement enough evaluation to reproduce the paper’s main analytical structure.

Required outputs:
- daily mean return, pre- and post-cost
- long-leg and short-leg contributions
- annualized Sharpe and Sortino
- standard deviation / downside deviation
- max drawdown
- Calmar ratio
- historical VaR / CVaR
- portfolio hit rate / share positive
- Newey-West standard errors / t-stats where applicable
- Pesaran-Timmermann directional test
- Fama-French regressions:
  - FF3
  - FF3 + momentum + reversal
  - FF5
- subperiod analysis:
  - 12/1992–03/2001
  - 04/2001–08/2008
  - 09/2008–12/2009
  - 01/2010–10/2015
- sensitivity analysis / robustness checks
- variable importance analysis
- comparison vs paper tables/results

## 5.11 Phase 1 acceptance criteria

Phase 1 is done only when:
- the full pipeline runs end to end,
- all core models train and produce predictions,
- ensembles work,
- ranking/trading works,
- costs are applied,
- the main evaluation suite runs,
- a written “differences vs original paper” note exists,
- and results are compared to the paper rather than reported in isolation.

---

## 6) Phase 2 — Implement the extensions

## 6.1 Goal
Keep the original pipeline intact, then add economically motivated extensions.

The extension exists because:
- the original target is binary classification,
- but the strategy monetizes tail ranking opportunities,
- and trading costs matter.

## 6.2 Extension targets

For each stock/day:
- `U_t = next_day_return - next_day_cross_sectional_median_return`
- `P_t = P(U_t > 0)`

This preserves parity with the original paper’s relative-performance framing.

## 6.3 Extension models

### DNN
Shared trunk + two heads:
- classification head for `P_t`
- regression head for `U_t`

### Trees
Train separate models:
- XGBoost classifier + XGBoost regressor
- RandomForest classifier + RandomForest regressor

Do not try to force true multi-task trees.

## 6.4 Extension DNN loss
Use:
- classification loss: cross-entropy/BCE-type loss on sign of `U_t`
- regression loss: Huber
- joint weighted sum

If the relative weighting requires a meaningful design choice, ask the user before finalizing the default.

## 6.5 Extension score families

For each base model family and ENS1 extension ensemble, compare:

1. **P-only score**
   - rank by `P_t`

2. **U-only score**
   - rank by `U_t`

3. **Composite score**
   - rank by `(2 * P_t - 1) * U_t`

This yields:
- DNN
- GBT/XGB
- RAF/RF
- ENS1

times 3 score families.

## 6.6 Extension ensemble scope
Only implement **ENS1** for the extension unless the user explicitly asks for more.

Recommended ENS1 extension logic:
- average `P_t` predictions across model families
- average `U_t` predictions across model families
- then compute the three score families from the ensembled outputs

Do not ensemble at a later stage unless explicitly approved.

## 6.7 No-trade band

Apply only to:
- `U`-score strategies,
- composite-score strategies.

Do **not** apply to `P`-only by default.

Recommended conceptual design:
- compare candidate replacement vs incumbent on expected-return score / expected gain basis
- require gain to exceed round-trip cost threshold

Important:
- the exact threshold value / sensitivity grid is not fully locked;
- if implementation reaches this point, ask before hard-coding.

## 6.8 Regime adaptation

Use **VIX-only** regime logic.

Do not use HMMs.

Important:
- exact thresholds and exact `(k_long, k_short)` mapping are not fully locked;
- ask before coding them in permanently.

## 6.9 Phase 2 acceptance criteria

Phase 2 is done only when:
- all three score families run,
- extension models are trained on the same walk-forward logic as Phase 1,
- results are compared fairly against the original baseline,
- no-trade band is implemented only where allowed,
- regime adaptation is implemented only after user confirmation of thresholds,
- and findings are framed as “did the extension improve alignment / net trading performance?” rather than as isolated ML scores.

---

## 7) Phase 3 — Build the deployed research system / trading simulator

## 7.1 Goal
Build a proper modular system that satisfies the course guidelines:
- clean data pipeline,
- backtest engine,
- transaction cost modeling,
- performance dashboard,
- risk metrics,
- ability to run new simulations,
- reproducibility,
- no lookahead bias,
- realistic execution assumptions.

## 7.2 Phase 3 system shape
This should be a **walk-forward sequential simulator**, not just a static chart viewer.

Core behavior:
- step through time,
- use only information available at each date,
- form signals,
- apply trading/rebalance logic,
- compute turnover and costs,
- realize P&L,
- log state and outputs.

## 7.3 Retraining schedule
Default simulator mode should preserve the paper’s block logic:
- train on trailing 750-day window,
- trade the next 250-day block,
- retrain,
- repeat.

Optional higher-frequency retraining can be added later only after asking.

Do not assume true online/incremental fitting is required by default.

## 7.4 Dashboard / app requirements
The app should allow the user to:
- choose phase / strategy family,
- choose model and score family,
- choose `k`,
- toggle transaction costs,
- toggle no-trade band (where applicable),
- toggle regime sizing (after thresholds confirmed),
- run simulation,
- inspect:
  - equity curve,
  - drawdown,
  - turnover,
  - return statistics,
  - risk metrics,
  - long/short leg behavior,
  - period-by-period diagnostics.

Do not over-engineer the UI.
This is a research system, not a polished brokerage frontend.

## 7.5 Phase 3 acceptance criteria
Phase 3 is done only when:
- the system runs sequentially,
- multiple strategies can be simulated through a reusable engine,
- the app can launch and run new simulations,
- results are reproducible,
- and the deliverable is clearly beyond a notebook-only workflow.

---

## 8) Recommended repository architecture

Use something close to the following.

```text
project_root/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── phase1_repro.yaml
│   ├── phase2_extension.yaml
│   └── phase3_simulator.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   └── exploratory/
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── run_phase1.py
│   ├── run_phase2.py
│   ├── run_phase3.py
│   └── build_reports.py
├── tests/
│   ├── test_universe_no_lookahead.py
│   ├── test_feature_alignment.py
│   ├── test_label_alignment.py
│   ├── test_backtest_timing.py
│   ├── test_costs.py
│   └── test_reproducibility.py
└── src/
    └── krauss/
        ├── data/
        │   ├── wrds_extract.py
        │   ├── universe.py
        │   ├── identifiers.py
        │   ├── prices_returns.py
        │   ├── study_periods.py
        │   ├── features.py
        │   └── labels.py
        ├── models/
        │   ├── dnn_phase1.py
        │   ├── xgb_phase1.py
        │   ├── rf_phase1.py
        │   ├── dnn_multitask.py
        │   ├── xgb_extension.py
        │   ├── rf_extension.py
        │   ├── ensembles_phase1.py
        │   └── ensembles_phase2.py
        ├── backtest/
        │   ├── ranking.py
        │   ├── portfolio.py
        │   ├── costs.py
        │   ├── rebalance.py
        │   ├── no_trade_band.py
        │   ├── regime.py
        │   └── simulator.py
        ├── evaluation/
        │   ├── metrics.py
        │   ├── risk.py
        │   ├── ff_factors.py
        │   ├── diagnostics.py
        │   └── reports.py
        └── utils/
            ├── config.py
            ├── dates.py
            ├── io.py
            ├── logging.py
            ├── seeds.py
            └── validation.py
```

You do not need to match these exact filenames, but you must preserve the separation of concerns.

---

## 9) Data contracts / expected intermediate tables

Create explicit, validated data tables.

### 9.1 Universe membership table
One row per security-date eligibility record, or a month-level membership matrix that can be expanded to daily eligibility.

Minimum fields:
- `date`
- `month_end_reference`
- `security_id`
- `ticker` (if available)
- `is_member_for_next_month`

### 9.2 Daily returns/prices table
Minimum fields:
- `date`
- `security_id`
- `return`
- adjusted/return-consistent price field(s) if needed
- any delisting-adjusted return fields if relevant
- metadata fields needed for joins and QA

### 9.3 Feature table
Minimum fields:
- `date`
- `security_id`
- `R1 ... R20`
- `R40, R60, ..., R240`

### 9.4 Label table
For Phase 1:
- `date`
- `security_id`
- `next_day_return`
- `next_day_cross_sectional_median_return`
- `y_binary`

For Phase 2:
- all of the above plus
- `u_excess`

### 9.5 Prediction table
Minimum fields:
- `date`
- `security_id`
- model identifier
- `p_hat`
- `u_hat` where applicable
- score family
- final ranking score

### 9.6 Trade / holdings table
Minimum fields:
- `date`
- side
- `security_id`
- target weight
- realized return
- turnover contribution
- transaction cost contribution

Every table should have validation checks.

---

## 10) Mandatory QA / validation checks

These are not optional.

### 10.1 No-lookahead checks
Test that:
- membership for month `m+1` is based only on month-end info from `m`,
- features at `t` use only information through `t`,
- labels use `t+1`,
- trades use predictions aligned correctly to realized next-day outcomes.

### 10.2 Determinism / reproducibility
Fix seeds where possible.
Log library versions.
Make runs reproducible.

### 10.3 Data integrity
Check for:
- duplicate keys,
- impossible joins,
- missing identifier mappings,
- unexpected gaps,
- feature lookback leakage,
- incorrect universe drift.

### 10.4 Result sanity
Before trusting any run, verify:
- portfolio remains dollar neutral,
- turnover and costs are plausible,
- long/short counts are correct,
- k selection behaves as intended,
- no-trade band only affects allowed strategies.

---

## 11) What to do when the paper cannot be matched exactly

Do not hide deviations.

If a deviation is unavoidable, do all of the following:
1. identify it clearly,
2. explain why it exists,
3. state whether it is caused by:
   - data source change,
   - library change,
   - ambiguous paper detail,
   - implementation simplification,
4. assess likely impact,
5. include it in the reproduction differences log.

Examples:
- Datastream vs WRDS
- H2O GBT vs XGBoost
- H2O DNN vs PyTorch exact maxout implementation differences

---

## 12) What still requires asking the user

These items are **not fully locked** and require confirmation before hard-coding:

1. exact WRDS schema/table names and identifier join path if ambiguous,
2. exact mapping from H2O GBT semantics to XGBoost parameters if non-obvious,
3. exact VIX thresholds,
4. exact regime-dependent `(k_long, k_short)` map,
5. exact no-trade threshold value / sensitivity grid,
6. whether any optional realism mode (e.g. next-open execution) should be added,
7. whether any optional higher-frequency retraining / quasi-online mode should be added,
8. any attempt to simplify the Phase 1 DNN away from a genuine maxout-style parity implementation.

---

## 13) Implementation order (strong recommendation)

Follow this order.

### Step 1
Set up repository structure, configs, seeds, logging, and tests.

### Step 2
Build and validate the WRDS universe + daily return pipeline.

### Step 3
Build feature/label generation with airtight date alignment tests.

### Step 4
Implement the Phase 1 backtest engine with a dummy scorer to validate timing/costs.

### Step 5
Implement Phase 1 models one by one:
- RF first
- XGB second
- DNN third
- then ENS1/ENS2/ENS3

### Step 6
Reproduce core metrics/tables and write the differences-vs-paper log.

### Step 7
Only after Phase 1 works, implement Phase 2 targets/models/scores.

### Step 8
Only after Phase 2 runs, implement the no-trade band and regime logic.

### Step 9
Build the sequential simulator and Streamlit app.

Do not jump ahead.

---

## 14) Final reminder

This project should feel like:
- a careful replication,
- followed by a disciplined extension,
- followed by a serious research system.

If anything important is unclear, ask the user instead of guessing.
