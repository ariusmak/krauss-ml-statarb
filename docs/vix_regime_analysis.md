# VIX Regime Analysis

This directory's addition to the repo. Five notebooks, one module, one fetch script, tests. Built to answer the regime question from our group chat: does VIX-based volatility regime detection change what k we pick, and should longs and shorts have different k?

Answer: no, on both. See notebooks for why.

## What's here

### Module
- `src/krauss/regimes/vix_regimes.py` — no-lookahead VIX regime labeling. 5-day trailing mean of VIX, shifted by 1 day before the rolling window. Three regimes under paper-matched thresholds (VIX <20 = low, 20-30 = mid, >30 = high). Coverage: 59% low / 31% mid / 9.5% high.
- `src/krauss/regimes/analysis.py` — helpers used across notebooks. `sharpe()`, `cross_sectional_zscore()`, `run_backtest()`, `backtest_k10()`, `regime_stats()`, `regime_stats_df()`, `bootstrap_sharpe_ci()`, `rule_cash_in_regime()`.

### Data
- `data/raw/vix_daily.parquet` — VIXCLS from FRED, 1990-2015. Fetched by `scripts/fetch_vix.py`. Already gitignored under the `*.parquet` rule.

### Tests
- `tests/test_vix_regimes.py` — 8 tests including no-lookahead invariants (perturb day-t VIX, label unchanged; append future spike, historical labels unchanged).
- `tests/test_regime_analysis.py` — 10 tests for helper primitives.

Run: `pytest tests/test_vix_regimes.py tests/test_regime_analysis.py -v`

### Notebooks (under `notebooks/`)

All five notebooks are self-contained. They read the committed data parquets and Arius's backtest modules. Nothing needs to be retrained.

1. `regime_analysis.ipynb` — k-sensitivity by regime. The original question. If you read only one, read this.
2. `regime_leg_decomp.ipynb` — leg decomposition, 2008 robustness, cash-on-high-vol rule.
3. `regime_cross_model.ipynb` — same analysis across DNN, XGB, RF, ENS1/2/3.
4. `regime_scoring_subperiod.ipynb` — four scoring schemes + Krauss-style sub-period breakdown. Section 5 has the matched-days selection-effect analysis.
5. `regime_asymmetric_k.ipynb` — closes Arius's "individual for long and short" ask. Includes cross-scheme robustness.

## How to use

### Labeling regimes on your own data

```python
from krauss.regimes.vix_regimes import RegimeConfig, label_vix_regimes, attach_regime
import pandas as pd

vix = pd.read_parquet('data/raw/vix_daily.parquet')
regimes = label_vix_regimes(vix, RegimeConfig())
# regimes is (date, regime) where regime in {'low_vol', 'mid_vol', 'high_vol'}

# Merge regime labels onto any daily series
daily_returns_with_regime = attach_regime(daily_returns, regimes)
```

`RegimeConfig()` defaults to `low_threshold=20, high_threshold=30, smoothing_window=5`. Override if you want to experiment with different thresholds.

### Running a regime-sliced backtest

```python
from krauss.regimes.analysis import backtest_k10, regime_stats_df
from krauss.regimes.vix_regimes import label_vix_regimes, RegimeConfig

# Any predictions DataFrame with columns [date, permno, <score_col>]
daily = backtest_k10(predictions, rets, score_col='p_ens1')
# daily is a post-cost daily return series with port_ret_net column

regimes = label_vix_regimes(vix, RegimeConfig())
table = regime_stats_df(daily, regimes)
# Returns a DataFrame with overall/low/mid/high Sharpe, means, turnover
```

### Bootstrap confidence intervals on Sharpe

```python
from krauss.regimes.analysis import bootstrap_sharpe_ci

point, lo, hi = bootstrap_sharpe_ci(daily['port_ret_net'], n_boot=2000, seed=42)
# 95% CI by default. Set seed for reproducibility.
```

### Reproducing specific findings

The headline results in the group-chat message come from these cells:

- **k=10 wins in every regime** → `regime_analysis.ipynb`, grid table in "K-sensitivity by regime" section.
- **Short leg breaks in high-vol, specifically 2008** → `regime_leg_decomp.ipynb`, "2008 concentration" section, where the drop-window argument drops Sharpe from 1.31 to 2.18.
- **Sharpe ranking across scoring schemes** → `regime_scoring_subperiod.ipynb`, Section 1 (scheme × regime table).
- **Matched-days selection effect (1.46x not 2.6x)** → `regime_scoring_subperiod.ipynb`, Section 5.
- **No asymmetric rule warranted** → `regime_asymmetric_k.ipynb`, both the main grid and the cross-scheme robustness section.
- **Sub-period decay** → `regime_scoring_subperiod.ipynb`, Section 3 and 4.

## Common follow-ups

### "I want to try different regime thresholds"

```python
cfg = RegimeConfig(low_threshold=15, high_threshold=25)  # tighter
regimes = label_vix_regimes(vix, cfg)
```

This breaks comparability with the headline results but lets you test sensitivity. Bootstrap CIs get thinner in "high" regime as your threshold drops; keep that in mind.

### "I want to add a new scoring scheme"

The pattern is in Part 4. Construct your score column on `pred2`, then pipe it through `run_backtest()` or `backtest_k10()`. If it's a gated scheme, copy the `gated_rank_and_select` function from `composite_gate_test.ipynb` or from Part 4's first code cell.

### "I want to rerun everything"

Each notebook is self-contained and runs in ~1-2 minutes on the committed parquets. To rerun all five:

```bash
cd notebooks/
for nb in regime_analysis regime_leg_decomp regime_cross_model regime_scoring_subperiod regime_asymmetric_k; do
  jupyter nbconvert --to notebook --execute "${nb}.ipynb" --inplace \
    --ExecutePreprocessor.timeout=600
done
```

No model retraining required — all backtests run on frozen predictions from `predictions_phase1.parquet` and `predictions_phase2.parquet`.

## Design choices worth noting

**Thresholds match paper Table 4.** VIX <20 / 20-30 / >30, not terciles. An independent review tried expanding terciles and got ~40% of days in "high" — incompatible with how the paper and the team talk about high vol. Paper-matched thresholds keep the regime work comparable to Krauss's own regime breakdowns.

**No-lookahead by construction.** The 5-day trailing mean of VIX is computed as `vix.shift(1).rolling(5).mean()`. Day-t's label uses VIX data strictly from before day t. Tested adversarially in `test_vix_regimes.py`.

**5 bps per half-turn on costs.** Matches the project-wide convention in `apply_transaction_costs`. Nothing changed here.

**Regime labels are attached to signal dates, not return dates.** When we slice `port_ret_net` by regime, the regime is the VIX regime on the signal/rebalance date, not the day the return is realized. This matches how the rest of the project aligns returns.

## Contact

Ping me on the group chat if something doesn't behave. The notebooks have enough prose that you should be able to read through them top-to-bottom — the "Reading the table" sections after each output explain what to look at.
