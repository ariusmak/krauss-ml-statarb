# Krauss: ML-Driven Statistical Arbitrage

Reproduction and extension of the Krauss et al. (2017) statistical arbitrage strategy using a Python stack built around WRDS data, PyTorch, XGBoost, scikit-learn, and Streamlit.

## Overview

This project studies whether machine learning models can generate tradable cross-sectional equity signals in a realistic daily long-short setting.

The repo has two goals:
1. **Reproduce** the core pipeline from Krauss et al. (2017) as faithfully as possible in Python.
2. **Extend** the original setup with better signal construction and a more complete research/trading workflow.

## Current status

**In progress**

### Completed
- Python project structure with configs, scripts, tests, and modular source code
- Core model-building and training workflow
- Reproduction framing for the original paper’s DNN, gradient-boosted tree, random forest, and ensemble setup
- Clear methodology for no-lookahead, survivor-bias-aware universe construction, and walk-forward evaluation

### In progress
- Full historical data pipeline from WRDS
- End-to-end backtesting and transaction cost evaluation
- Final comparison of reproduction results versus the paper
- Extension experiments and Streamlit research interface

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
app/          Streamlit interface and presentation layer
configs/      Reproducible experiment settings
docs/         Project notes and supporting documentation
notebooks/    Research and exploratory analysis
scripts/      Runnable pipeline entry points
src/krauss/   Core project code
tests/        Validation and regression checks
