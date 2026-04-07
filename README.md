# README.md — Krauss Project Roadmap

## Before you read: major decisions, deck deviations, and project direction

This project started from a short proposal deck, but it is **not** being implemented exactly as the deck wrote it. 

> We are trying to:
> 1. reproduce the original paper as faithfully as possible,
> 2. extend it in a way that is economically motivated and internally coherent,
> 3. package it as a real, sequential trading research system.

That leads to several important choices.

### The biggest decisions already made

| Topic | Final project direction | Why |
|---|---|---|

| Data source | **WRDS-accessible data**, built to be as close as possible to the paper’s Datastream setup | We have WRDS access, not Datastream. |
| Universe construction | **End-of-month constituency determines the next month’s universe** | This preserves no-lookahead and matches the intended survivor-bias-free setup. |
| Reproduction goal | **Best-faith Python/WRDS replication**, with explicit logging of differences vs the paper | Exact vendor/library parity is impossible, so deviations must be documented instead of hidden. |
| Python stack | **PyTorch + XGBoost + sklearn + Streamlit** | Locked by project choice. |
| Extension DNN | **Shared trunk + two heads** | Best fit for the dual-objective idea without abandoning the paper’s architecture spirit. |
| Extension targets | Both `P` and `U` are defined **relative to the next-day cross-sectional median** | Keeps the extension aligned with the original paper’s target framing. |
| Composite score | **`(2P - 1) * U`**, not the deck’s `P * U` | The deck formula is directionally flawed; the new formula better combines sign confidence and expected magnitude. |
| No-trade band | Apply it **only where `U` is predicted** | It needs a return-like quantity; it should not be forced onto the `P`-only baseline. |
| Regime logic | **VIX-only**, not HMM | Simpler, more interpretable, and consistent with the crisis-sensitivity story in the paper. |
| Extension ensemble | **ENS1 only** | Keeps Phase 2 manageable and focused. |
| Deployed system | **Sequential walk-forward simulator + Streamlit interface**, not just a static dashboard | Closer to the course guidelines and the spirit of a real trading system. |
| Baseline retraining schedule | **Keep the paper’s block structure first** | Parity comes before “live-ish” experimentation. |

### Important places where the project intentionally departs from the original deck

1. **The deck’s `P * U` composite is not being used.**  
   That formula looks simple, but it breaks the directional logic when `U` is negative. The project will instead use `(2P - 1) * U`.

2. **The no-trade band is narrower in scope than the deck suggests.**  
   We are not applying it to the `P`-only strategies. It only makes clean economic sense where the model predicts a return-like quantity.

3. **The extension ensemble is simpler than the original paper’s ensemble comparison.**  
   Phase 1 reproduces ENS1 / ENS2 / ENS3 because the paper did. Phase 2 uses **ENS1 only** so the extension does not explode in complexity.

4. **The app is not “just a dashboard.”**  
   The final system should behave like a sequential simulator that walks through time and produces trading decisions, not just a pretty wrapper around stored backtests.



---

## 1) What the original paper actually does

Krauss et al. (2017) builds a daily statistical arbitrage strategy on the S&P 500 using three machine-learning model families:

- Deep neural networks
- Gradient-boosted trees
- Random forests

It then combines them into three ensembles:
- ENS1: equal-weighted
- ENS2: performance-weighted
- ENS3: rank-weighted

### Core setup
- Universe: survivor-bias-free S&P 500 constituents
- Frequency: daily
- Features: 31 lagged return features
  - `R(1)` through `R(20)`
  - `R(40), R(60), ..., R(240)`
- Target: whether a stock’s next-day return beats the **next-day cross-sectional median return**
- Train/trade schedule:
  - 750-day training window
  - 250-day trading window
  - repeated in non-overlapping trading blocks
- Trading rule:
  - rank all stocks by predicted probability of outperforming the median,
  - long the top `k`,
  - short the bottom `k`,
  - equal weight,
  - dollar neutral,
  - daily rebalance,
  - 5 bps transaction cost per half-turn

### Why the paper matters
The paper is interesting for two reasons at once:
1. it is a strong applied ML paper in finance,
2. it exposes a tension between the training objective and the trading objective.

The models are trained as binary classifiers, but the strategy makes money by trading the **tails** of a cross-sectional ranking. That gap is exactly where the extension becomes interesting.

---

## 2) Why the extension makes sense

The original model says:
- “Will this stock beat the cross-sectional median tomorrow?”

But the strategy really cares about:
- “How attractive is this stock relative to the rest of the cross-section?”
- “How strong is the opportunity?”
- “Is the opportunity strong enough to justify turnover and costs?”

That creates three economically sensible extension ideas:

### A. Predict both direction and magnitude
Instead of only predicting a binary event, also predict an expected excess return relative to the next-day cross-sectional median.

This gives two outputs:
- `P_t = P(U_t > 0)`
- `U_t = E[next-day return - next-day median return]`

### B. Rank using a better score
Instead of ranking only by the probability of “winning,” compare:
- `P` only,
- `U` only,
- composite score `(2P - 1) * U`

This lets us test whether the extension better aligns model training with actual trading P&L.

### C. Trade more intelligently
If a signal barely changes, replacing a position every day may be too expensive.
That motivates:
- a no-trade band,
- and regime-adaptive sizing.

The point of the extension is not novelty for novelty’s sake.  
It is to fix the most obvious mismatch between the original paper’s learning problem and its trading problem.

---

## 3) What “reproduction” means in this project

This is probably the most important framing decision in the whole project.

Because the paper used **Datastream + H2O + R**, while this project will use **WRDS + Python + PyTorch/XGBoost/sklearn**, we should think of Phase 1 as:

> **A best-faith methodological replication with explicit deviation accounting**

not:
- a literal byte-for-byte vendor/library clone,
- and not a loose “same idea, different system” reimplementation.

### What must be preserved
The following are the real non-negotiables:

- no lookahead bias,
- survivor-bias-aware universe construction,
- end-of-month constituency driving the next month’s tradable universe,
- the paper’s feature set,
- the paper’s label definition,
- the paper’s rolling train/trade structure,
- the paper’s ranking and trading rule,
- the paper’s evaluation style.

### What will inevitably differ
At least three things will differ:

1. **Data vendor**
   - Datastream in the paper,
   - WRDS-accessible data here.

2. **Tree library**
   - H2O in the paper,
   - XGBoost / sklearn here.

3. **DNN library**
   - H2O in the paper,
   - PyTorch here.

These differences are acceptable only if they are:
- recognized,
- documented,
- and discussed when interpreting reproduction gaps.

---

## 4) The three project phases

## Phase 1 — Reproduce the original paper

### Goal
Build the full original pipeline in Python and get as close as possible to the original results and result patterns.

### Deliverables
- WRDS-based historical S&P 500 universe construction
- daily return / price pipeline
- feature generation
- original binary classification label
- original DNN, GBT, RAF analogues
- ENS1 / ENS2 / ENS3
- ranking + trading engine
- transaction cost model
- reproduction metrics and comparison to paper

### What success looks like
Success is **not** “exactly match every number.”
Success is:
- broad parity in method,
- broad parity in result magnitudes and patterns,
- and a convincing explanation of where gaps come from.

### The models in Phase 1

#### DNN
The paper’s DNN is a relatively small but nontrivial architecture:
- `31-31-10-5-2`
- maxout hidden layers
- dropout
- ADADELTA
- slight L1 regularization
- early stopping

This matters because the paper’s DNN is not just a generic feedforward net. If the PyTorch version collapses into a convenience MLP, the reproduction stops being faithful.

#### GBT
The original tree boosting model is H2O-based, not XGBoost. That means the Phase 1 “GBT” is really a Python analogue, not a perfect clone. This is one of the most important reproduction caveats.

#### RAF
The random forest is conceptually the easiest model to translate into sklearn and may end up being the cleanest baseline among the three base learners.

#### Ensembles
Phase 1 must include:
- ENS1,
- ENS2,
- ENS3,

because one of the paper’s important findings is that the simple ensemble is hard to beat.

### Phase 1 evaluation
Phase 1 should reproduce as much of the paper’s evaluation structure as practical:

- returns pre/post cost
- long and short contributions
- Sharpe and Sortino
- drawdown and Calmar
- VaR/CVaR
- directional accuracy style measures
- Fama-French regressions
- subperiod analysis
- robustness checks

This is important because the course guidelines do not allow a “just train models and show accuracy” project.

---

## Phase 2 — Implement the extensions

### Goal
Keep the original pipeline intact, then test whether the extensions improve the strategy in a fair comparison.

### Extension 1: Dual-objective training
Each model family will now produce:
- a directional probability `P`,
- and an expected excess return `U`.

The key design choice here is that both are defined relative to the **next-day cross-sectional median**, so the extension stays aligned with the original paper’s framing rather than drifting into a different problem.

### Extension 2: Alternative scoring systems
For each model family plus ENS1:
- rank by `P`,
- rank by `U`,
- rank by `(2P - 1) * U`.

This is the core comparison in Phase 2.

### Extension 3: No-trade band
Only use this where it makes economic sense:
- `U` strategies,
- composite strategies.

Do not try to shoehorn it onto the `P`-only baseline.

### Extension 4: Regime-adaptive portfolio sizing
Use VIX-based regimes only.
The idea is not to build a complicated macro-state model, but to test whether long/short breadth should vary in calm vs stressed environments.

### What success looks like in Phase 2
The right question is:
- “Did the new scoring / execution logic improve trading outcomes net of costs?”
not:
- “Did the new model get a slightly better loss value?”

---

## Phase 3 — Build the full trading system

### Goal
Satisfy the course requirement for a real deployed application / trading system pipeline.

The guidelines clearly require more than a notebook:
- clean data pipeline,
- backtest engine,
- transaction cost modeling,
- performance dashboard,
- risk metrics,
- ability to run new simulations,
- reproducibility.

### What the Phase 3 system should feel like
The right mental model is:

> a **research-grade trading simulator**

not:
- a toy dashboard,
- and not necessarily a production brokerage system.

The simulator should move through time sequentially, generate signals, rebalance, charge costs, and store state.

### Why this matters
This bridges the academic and systems parts of the assignment:
- Phase 1 and 2 answer the research question,
- Phase 3 proves that the strategy lives inside a proper trading pipeline.

---

## 5) Suggested repository architecture

The cleanest architecture is a modular Python repo with separated concerns.

### Suggested layout
- `src/data/` — WRDS extraction, universe construction, identifiers, prices/returns, study periods, features, labels
- `src/models/` — Phase 1 models, extension models, ensembles
- `src/backtest/` — ranking, positions, costs, no-trade logic, regime logic, simulator
- `src/evaluation/` — performance metrics, risk, factor regressions, diagnostics, reports
- `src/app/` — Streamlit app
- `src/utils/` — reusable helpers
- `configs/` — reproducible configs by phase
- `tests/` — timing, leakage, reproducibility, and cost tests

### Why modularity matters
The project has three natural layers:
1. **data layer**
2. **model/signal layer**
3. **trading/evaluation layer**

If those are mixed together, it becomes much harder to:
- verify no lookahead,
- compare Phase 1 vs Phase 2 fairly,
- or build a clean Phase 3 app.

---

## 6) The hardest implementation risks

These are the places most likely to cause trouble.

### A. Universe construction and identifier mapping
This is probably the single most important engineering risk.
If the historical universe is wrong, everything downstream is wrong.

### B. Timing alignment
Daily features, next-day labels, monthly membership updates, and block retraining all have to line up exactly.

### C. “Almost the same” model substitutions
It is easy to tell yourself that:
- XGBoost is “basically the same” as the paper’s GBT,
- or a generic PyTorch MLP is “basically the same” as the paper’s DNN.

That is not safe.
Those substitutions need to be treated as explicit deviations, not invisible implementation shortcuts.

### D. Scope creep in Phase 2 and 3
The project can easily sprawl if:
- hyperparameter tuning gets too broad,
- too many regime rules are added,
- the app becomes a product-design project,
- or “online learning” becomes a rabbit hole.

The project should stay disciplined.

---

## 7) What still deserves careful confirmation during implementation

Even after the decisions already made, there are a few things that should still be treated carefully rather than assumed:

1. **Exact WRDS schema / join path**  
   Which membership and daily return tables are available and how they map.

2. **Exact parity mapping for the paper’s GBT in XGBoost**  
   This is an approximation problem, not a direct translation.

3. **Exact VIX thresholds and the exact `(k_long, k_short)` map**  
   The method is locked in, but the precise rule still deserves confirmation.

4. **Exact no-trade threshold value(s)**  
   The logic is clear, but the threshold still needs explicit confirmation before hard-coding.

5. **Whether to add optional higher-frequency refits / quasi-online updates in Phase 3**  
   The baseline should stay aligned with the paper first.

These are not reasons to delay the documentation or the architecture. They are just the places where implementation should pause and confirm rather than improvise.

---

## 8) Recommended implementation order

A sensible build order is:

1. set up repo structure, configs, tests, and reproducibility utilities,
2. build the WRDS universe + daily return pipeline,
3. build features and labels,
4. validate timing and no-lookahead with tests,
5. implement a barebones backtest engine first,
6. implement Phase 1 models and ensembles,
7. reproduce the main paper evaluation,
8. only then add Phase 2 extension targets/models/scores,
9. only then add no-trade and regime logic,
10. build the Phase 3 Streamlit simulator last.

This order matters because it forces the project to earn complexity only after the baseline is correct.

---

## 9) What the final project should feel like

If the project is done well, the final result should feel like this:

- **Phase 1:** “We understand the original paper deeply and reproduced it in a disciplined way.”
- **Phase 2:** “We extended it for a reason, not just because we could.”
- **Phase 3:** “We turned it into a serious trading research system.”

That combination — paper understanding, extension discipline, and system design — is exactly the spirit of the course guidelines.
