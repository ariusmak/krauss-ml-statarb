# Krauss: ML-Driven Statistical Arbitrage

Reproduction and extension of the Krauss et al. (2017) statistical arbitrage strategy using a Python stack built around WRDS data, PyTorch, XGBoost, scikit-learn, H2O, and parquet-backed research pipelines.

## Overview

This project studies whether machine learning models can generate tradable cross-sectional equity signals in a realistic daily long-short setting.

The repo has two goals:
1. **Reproduce** the core pipeline from Krauss et al. (2017) as faithfully as possible in Python.
2. **Extend** the original setup with better signal construction and a more complete research/trading workflow.

## Current status

**Working research pipeline with unfinished presentation and validation layers**

### Implemented
- CRSP-based data pipeline from WRDS with monthly no-lookahead S&P 500 eligibility, delisting-adjusted returns, 31 lagged-return features, and Phase 1/Phase 2 labels
- Datastream-based pipeline matching the paper more closely, including saved US-trading-calendar-filtered datasets and H2O reproduction outputs
- Rolling 23-period walk-forward study construction (750 train / 250 trade) with saved predictions covering 1992-12-17 through 2015-10-15
- Phase 1 model runners for RF, XGB, PyTorch DNN, H2O RF/GBT/DNN, and ENS1/ENS2/ENS3 ensembles
- Phase 2 extension runner with multitask DNN plus RF/XGB classifier-regressor pairs and multiple score families
- Backtest engine for ranking, portfolio formation, turnover, and transaction-cost application
- Robustness and frozen-universe experiment scripts
- Saved research artifacts under `data/` for CRSP, Datastream, H2O, robustness, and Phase 2 runs

### Not yet complete
- Automated test coverage: `tests/` exists, but the files are currently placeholders and `pytest` does not run any real tests
- Reporting/evaluation package: `src/krauss/evaluation/` is scaffolded but currently empty
- Utility helpers in `src/krauss/utils/` are scaffolded but mostly empty
- Streamlit app layer: `app/streamlit_app.py` is currently an empty stub
- Phase 3 simulator/reporting workflow: `scripts/run_phase3.py`, `scripts/build_reports.py`, and `configs/phase3_simulator.yaml` are placeholders

## What this project demonstrates

- Applied machine learning for financial prediction
- Careful leakage-aware and survivor-bias-aware research design
- Translating an academic paper into a structured Python research pipeline
- Model comparison across neural nets, boosted trees, random forests, and ensembles
- Connecting model outputs to trading rules rather than stopping at prediction accuracy

## Method summary

The original paper predicts whether a stock’s next-day return will beat the next-day cross-sectional median, then forms a daily dollar-neutral long-short portfolio by ranking stocks on predicted signal strength.

This project keeps that framing and adds a more explicit research extension: predicting both **direction** and **magnitude** of excess return relative to the next-day cross-sectional median, then testing alternative ranking rules and execution logic.

## Preliminary results

### ENS1 post-cost daily return (%) across score constructions

| Score construction | k = 10 | k = 50 | vs. Phase 1 baseline at k = 50 |
|---|---:|---:|---|
| Phase 1 baseline | 0.2788 | 0.1208 | Baseline |
| U-only | 0.2567 | 0.1012 | Underperforms |
| Product composite `(2P̂ - 1) * Ũ` | -0.0489 | -0.0616 | Underperforms |
| Rank composite `0.5r(P̂) + 0.5r(Ũ)` | 0.2723 | 0.1208 | Matches baseline |
| Z-score composite `0.5z(P̂) + 0.5z(Ũ)` | 0.3058 | 0.1301 | Outperforms |

### ENS1 gated U-ranking variants

| Score construction | k = 10 | k = 50 | vs. Phase 1 baseline at k = 50 | Trading days |
|---|---:|---:|---|---:|
| Phase 1 baseline | 0.2788 | 0.1208 | Baseline | 5750 |
| U-only (no gate) | 0.2567 | 0.1012 | Underperforms | 5750 |
| P-gate(0.02) + U | 0.2621 | 0.1547 | Outperforms | 5667 |
| P-gate(0.03) + U | 0.3475 | 0.2556 | Outperforms | 5142 |
| P-gate(0.05) + U | 0.7193 | 0.6295 | Outperforms* | 2703 |

\* Higher gates improve average return in the saved notebook outputs but materially reduce trading frequency, so these results should be treated as exploratory rather than final.

Across the current Phase 2 extensions, pure excess-return ranking alone does not beat the Phase 1 baseline, and the direct product composite performs worst suggesting that the U and P predictions often directionally disagree. The strongest non-gated extension so far is the z-score composite, while the rank composite roughly matches the baseline. Gated U-ranking variants are promising, but tighter gates also reduce the number of active trading days substantially.

## Repo structure

```text
app/          Planned app layer (currently stubbed)
configs/      Reproducible experiment settings
docs/         Project notes and supporting documentation
notebooks/    Research and exploratory analysis
scripts/      Runnable pipeline entry points
src/krauss/   Core project code
tests/        Placeholder test files (not yet implemented)
