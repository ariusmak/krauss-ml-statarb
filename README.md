# ML-Driven Statistical Arbitrage: Signal Construction and Cost-Aware Execution

This repository reproduces and extends **Krauss, Do & Huck (2017)**, *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500*.

The main extension is to move beyond the paper's binary probability-ranking setup and test a more explicit signal-construction problem:

```text
Given estimates of directional edge P̂ and magnitude edge Û,
how should a daily long-short portfolio trade after costs?
```

The repo evaluates this through:

- dual-output models estimating both `P̂ = P(U > 0)` and `Û = E[U]`, where `U` is next-day excess return relative to the cross-sectional median;
- score-construction experiments: probability-only, magnitude-only, product composite, rank composite, z-score composite, and probability-gated magnitude ranking;
- a transaction-cost-aware no-trade band that only replaces incumbent positions when predicted incremental `Û` clears a cost hurdle;
- walk-forward long-short backtests with daily turnover, 5 bps per half-turn costs, and post-cost Sharpe/return diagnostics;
- a Streamlit research app for equity curves, cost/turnover comparisons, regime checks, and result matrices.

The reproduction is used as a baseline and validity check. The more relevant research question is whether model outputs can be transformed into better portfolio decisions after accounting for scaling, disagreement between signals, turnover, and trading frictions.

## Main findings

### 1. Magnitude forecasts help only when the score construction is stable

The Phase 2 extension adds a continuous target:

```text
U = next-day stock return - next-day cross-sectional median return
```

Each model estimates:

```text
P̂ = P(U > 0)
Û = predicted excess-return magnitude
```

The first test is whether `Û` improves the ranking rule relative to the original probability-only baseline.

**ENS1 post-cost daily return across score constructions**

| Score construction | k = 10 | k = 50 | Interpretation |
|---|---:|---:|---|
| Phase 1 baseline `P̂` | 0.2788% | 0.1208% | Strong probability-ranking benchmark |
| U-only `Û` | 0.2567% | 0.1012% | Magnitude alone does not dominate direction |
| Product composite `(2P̂ - 1) * Û` | -0.0489% | -0.0616% | Bad combination rule; sign/scale disagreement matters |
| Rank composite `0.5r(P̂) + 0.5r(Û)` | 0.2723% | 0.1208% | Recovers most of baseline performance |
| Z-score composite `0.5z(P̂) + 0.5z(Û)` | 0.3058% | 0.1301% | Best non-gated result in saved notebook outputs |

The useful result is not simply that one composite works. The useful result is that raw `Û` is noisy enough that naive arithmetic can destroy the signal. The product score looks appealing because it has an expected-edge interpretation, but empirically it is unstable when `P̂` and `Û` disagree or have incompatible cross-sectional scales.

The rank and z-score composites are more robust because they force both signals onto comparable cross-sectional scales before combining them. That is a reasonable outcome: in a noisy one-day equity prediction problem, the relative ordering of signals may be more reliable than raw predicted return units.

### 2. Probability gating improves the use of Û, but it changes the trading sample

The second extension uses `P̂` as a directional filter and `Û` as the ranking variable only inside the filtered long and short pools:

```text
Long candidates:  P̂ > 0.5 + threshold, then rank by Û descending
Short candidates: P̂ < 0.5 - threshold, then rank by Û ascending
```

**ENS1 gated U-ranking variants**

| Score construction | k = 10 | k = 50 | Trading days | Interpretation |
|---|---:|---:|---:|---|
| Phase 1 baseline | 0.2788% | 0.1208% | 5,750 | Always-trading probability baseline |
| U-only, no gate | 0.2567% | 0.1012% | 5,750 | Magnitude without direction filtering underperforms |
| P-gate(0.02) + U | 0.2621% | 0.1547% | 5,667 | Modest gate helps k=50 |
| P-gate(0.03) + U | 0.3475% | 0.2556% | 5,142 | Better average return, lower coverage |
| P-gate(0.05) + U | 0.7193% | 0.6295% | 2,703 | High average return, materially fewer active days |

This is a positive result, but not a free lunch. A tighter gate improves average return partly by restricting the trading sample. That may be exactly what a live strategy should do if there are many low-conviction days, but it also makes the comparison to always-trading baselines less direct.

The interpretation is that `Û` is more useful conditional on directional confidence. It is not enough to ask which names have the largest predicted magnitude; the model also needs to be confident about the sign of that magnitude.

### 3. Cost-aware no-trade bands can improve Sharpe when turnover is mostly low-edge churn

The baseline strategy re-ranks and rebalances daily. That matches the paper, but it can overtrade small rank changes. The extension adds a no-trade band:

```text
Only replace an incumbent if the challenger's predicted Û improvement exceeds the cost threshold.
```

The current band uses a 10 bps swap hurdle, motivated by the paper's 5 bps per half-turn assumption. A replacement requires paying to exit one position and enter another, so a round-trip hurdle is the natural baseline.

The implementation is position-aware rather than rank-only:

- keep names that remain in the candidate set;
- compare expiring incumbents to new candidates using `Û`;
- swap only when the predicted improvement clears the threshold;
- force-evict names that leave the valid universe;
- track holdings explicitly because the portfolio can differ from the daily top-k/bottom-k list.

The most informative result is that the band helped **XGB** more than the other model families. That pattern is plausible rather than accidental:

- boosted trees can produce sharper daily rank changes from small feature movements;
- those rank changes raise turnover even when the economic edge difference is small;
- the band removes trades whose estimated incremental alpha is not large enough to pay the cost;
- smoother models and ensembles have less low-edge churn to remove, so the same threshold can also suppress useful trades.

The result is therefore conditional: the no-trade band is not universally good. It helps when the model creates turnover that is not compensated by expected alpha. For models with better-calibrated magnitude forecasts, this framework is more interesting because the trade/no-trade decision can be stated directly in return units:

```text
Trade if expected incremental alpha > transaction cost + uncertainty buffer.
```

That is the practical reason to model `Û`: not necessarily to rank the entire book by raw expected return, but to decide whether a marginal rebalance is worth paying for.

## Research design

The baseline follows the original paper's one-day cross-sectional setup.

For each eligible stock on signal date `t`:

```text
y_binary = 1 if next-day stock return > next-day cross-sectional median return
           0 otherwise
```

The daily strategy:

- ranks the eligible universe by model score;
- goes long the top `k` names;
- shorts the bottom `k` names;
- uses equal weights on each side;
- realizes next-day close-to-close returns;
- subtracts transaction costs based on daily turnover.

The extension changes the score, not the trading environment. This keeps the comparison clean: the question is whether different transformations of `P̂` and `Û` improve the same long-short construction.

## Data pipeline

### Primary paper-parity pipeline

- **Data source:** Thomson Reuters Datastream via WRDS.
- **Constituents:** S&P 500 Datastream monthly constituent table.
- **Returns:** Datastream daily total-return index (`RI`), converted to returns as `RI_t / RI_{t-1} - 1`.
- **Calendar:** Datastream observations are filtered to U.S. trading days to remove stale non-U.S.-holiday rows.
- **Identifier:** Datastream `infocode`.

### Alternative comparison pipeline

- **Data source:** WRDS CRSP.
- **Constituents:** `crsp.dsp500list`.
- **Returns:** `crsp.dsf`, including delisting-return adjustment.
- **Identifier:** CRSP `PERMNO`.

The CRSP pipeline is retained for comparison. The Datastream + H2O pipeline is the closest match to the original paper.

### Universe construction

The main universe rule is monthly and no-lookahead:

```text
S&P 500 membership at month-end M determines eligibility for month M+1.
```

The effective modeling panel is then filtered by:

- return availability;
- 240-day feature lookback availability;
- label availability;
- next-day eligibility for the median calculation.

This avoids treating “S&P 500 constituent” as a single static concept. The actual tradable panel is the intersection of membership, data availability, feature completeness, and label construction.

### Features

The feature set follows Krauss et al.'s lagged-return specification:

```text
R1, R2, ..., R20, R40, R60, ..., R240
```

Each feature is computed from a cumulative total-return index:

```text
R_{t,m} = P_t / P_{t-m} - 1
```

No feature normalization is applied in the paper-parity setup.

### Walk-forward periods

Each study period contains:

- 750 trading days for training;
- 250 trading days for trading;
- 240 initial training days consumed by the longest feature lookback;
- a 250-trading-day roll forward.

This produces 23 rolling study periods in the original reproduction window.

## Models

### Reproduction models

- **DNN:** H2O Deep Learning for paper parity; PyTorch comparison implementation with the same maxout architecture.
- **GBT / XGB:** H2O GBM for paper parity; XGBoost for Python and extension work.
- **RAF / RF:** H2O Distributed Random Forest for paper parity; scikit-learn Random Forest for Python and extension work.
- **ENS1:** equal-weight average of DNN, GBT/XGB, and RAF/RF probabilities.
- **ENS2:** training-period Gini/AUC-weighted ensemble.
- **ENS3:** inverse-rank-weighted ensemble based on training-period Gini rankings.

### Extension models

- **RF extension:** classifier + regressor pair.
- **XGB extension:** classifier + regressor pair.
- **Multitask DNN:** shared maxout trunk with a classification head for `P̂` and a regression head for `Û`.
- **ENS1 extension:** equal-weight average of model-family `P̂` outputs and model-family `Û` outputs.

## Reproduction result

The closest reproduction uses Datastream + H2O. At `k = 10`, full-sample pre-cost returns are close to the published results.

| Model | Paper pre-cost daily return | This repo pre-cost daily return | Ratio to paper |
|---|---:|---:|---:|
| DNN | 0.33% | 0.28% | 85% |
| GBT | 0.37% | 0.39% | 106% |
| RAF | 0.43% | 0.40% | 93% |
| ENS1 | 0.45% | 0.42% | 94% |

This is close enough to validate the data and backtest pipeline. It is not exact, and exact equality is not expected. The original paper used a 2016-era Datastream pull and 2016-era H2O implementation; current WRDS data and current H2O internals can differ.

## Research app

The Streamlit app is a precomputed research demo. It summarizes:

- reproduction results;
- Phase 2 signal extensions;
- equity curves and result matrices;
- regime-analysis attempts;
- turnover and cost-aware execution;
- model and scoring-scheme explanations;
- pipeline diagrams.

Run after building app data:

```bash
streamlit run app/streamlit_app.py
```

The app reads precomputed artifacts. It does not rerun models or backtests at page-render time.

## Repository structure

```text
app/                 Streamlit research demo and app data builders
configs/             Experiment settings
docs/                Build log, reproduction process, deviation log, universe notes
notebooks/           Research notebooks and exploratory result tables
scripts/             Pipeline entry points for data, models, backtests, app artifacts
src/krauss/          Core reusable data, model, backtest, and evaluation code
tests/               Placeholder test structure; not yet a complete test suite
```

## Reproducible workflow

Install the package:

```bash
pip install -e ".[dev]"
```

Build CRSP data and labels:

```bash
python scripts/build_data.py
python scripts/build_features_labels.py
```

Run Phase 1 and Phase 2 Python pipelines:

```bash
python scripts/run_phase1.py
python scripts/run_phase2.py
```

Build Datastream paper-parity data:

```bash
python scripts/build_data_datastream.py
python scripts/build_features_labels_datastream.py
```

Run H2O / Datastream reproduction and extension scripts where available:

```bash
python scripts/run_phase1_h2o.py
python scripts/run_phase2_datastream.py
```

Build app artifacts:

```bash
python app/scripts/build_app_data.py
```

Notes:

- WRDS access is required for CRSP and Datastream extraction.
- H2O reproduction scripts require H2O in the environment.
- Some app artifacts come from precomputed research outputs rather than live app-time backtests.

## Limitations

### Not a production trading system

The backtests use close-to-close daily returns and simplified daily rebalancing. A production implementation would need execution-time assumptions, spread estimates, borrow costs, short-sale constraints, financing, margin, capacity analysis, liquidity filters, and portfolio-level exposure controls.

### Simplified transaction costs

The paper-parity cost model uses a flat 5 bps per half-turn. This is useful for comparison, but real costs vary by name, date, liquidity, volatility, participation rate, and short availability. The no-trade band is a step toward cost-aware execution, not a full transaction-cost model.

### Magnitude forecasts need calibration work

The extension shows that `Û` can improve signal construction when normalized or gated, but raw magnitude forecasts are noisy. A stronger version would evaluate calibration in return units and include uncertainty-aware thresholds.

### Gated strategies change exposure and sample selection

P-gated strategies trade fewer days. Higher average returns under tight gates should be evaluated with matched-day analysis, turnover-adjusted metrics, and out-of-sample threshold selection.

### Multiple-testing risk

The repo tests several models, score rules, thresholds, eras, and cost regimes. Some results may reflect researcher degrees of freedom. A stricter validation protocol would select gates, bands, and score constructions before final evaluation.

### Exact replication is not fully controllable

The original paper used a 2016-era Datastream pull and 2016-era H2O internals. Modern WRDS data and modern H2O can differ in constituents, revised return histories, tree-splitting behavior, DNN training internals, and stochastic behavior.

### Alpha decay is part of the result

The original strategy weakens over time. That should be treated as a core finding, not as an implementation nuisance. The project is best read as a replication and signal-construction study, not as evidence that the original strategy remains directly deployable.

### Engineering work remains

The research pipeline is functional, but automated tests are incomplete. The next engineering step is to convert more notebook results into parameterized scripts with unit tests for universe construction, label alignment, turnover accounting, no-trade-band behavior, and app-data generation.

## Next steps

Natural extensions:

1. **Magnitude calibration:** test whether `Û` is reliable in return units, not just useful for ranking.
2. **Liquidity-aware costs:** replace the flat 5 bps assumption with spread, ADV, volatility, borrow, and participation-rate estimates.
3. **Uncertainty-aware execution:** trade only when expected incremental alpha exceeds both cost and forecast uncertainty.
4. **Threshold validation:** select gates and no-trade bands using a formal validation protocol.
5. **Portfolio constraints:** add beta, sector, volatility, and gross/net exposure controls.

## Bottom line

The repo starts from a published ML stat-arb strategy and extends it into a signal-construction and execution problem. The main research finding is that model outputs should not be treated as rankings mechanically. Direction, magnitude, scale, turnover, and costs interact. The best-performing rules are the ones that respect that interaction.