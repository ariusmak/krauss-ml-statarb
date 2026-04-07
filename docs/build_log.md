# Krauss Project Build Log

## Session 1 — 2026-04-06

### Step 1: Repository setup
- Created `pyproject.toml` with all dependencies (numpy, pandas, scipy, wrds, sklearn, xgboost, torch, etc.)
- Dev dependencies (pytest, ruff, black, ipykernel) under `[project.optional-dependencies.dev]`
- Fixed build backend from incorrect `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`
- Created full directory structure per CLAUDE.md Section 8
- Installed with `pip install -e ".[dev]"` in conda env `ML` (Python 3.13)

### Step 2: WRDS data extraction
**Decisions made:**
- **Data source:** CRSP daily stock file (`crsp.dsf`) — closest WRDS analogue to the paper's Datastream total return indices
- **Canonical identifier:** PERMNO (security-level, stable across time)
- **S&P 500 membership:** `crsp.dsp500list` (start/ending spell table, keyed on PERMNO)
- **Authentication:** WRDS prompts interactively; credentials cached in `~/.pgpass`

**Scripts/modules built:**
- `src/krauss/data/wrds_extract.py` — three query functions:
  - `fetch_sp500_membership()` — full `crsp.dsp500list` table
  - `fetch_daily_stock_data()` — `crsp.dsf` (ret, prc, shrout, cfacpr, cfacshr)
  - `fetch_delisting_returns()` — `crsp.dsedelist` (dlstdt, dlret, dlstcd)

- `src/krauss/data/universe.py` — no-lookahead monthly membership:
  - `build_membership_matrix()` — for each month-end, checks which PERMNOs have an active S&P 500 spell (`start <= month_end AND (ending >= month_end OR ending IS NULL)`), assigns them as eligible for the NEXT month
  - `get_eligible_universe(trade_date)` — lookup function mapping any trading date to its eligible PERMNOs
  - `build_daily_eligibility()` — expands monthly panel to (date, permno) rows
  - **No-lookahead rule:** Jan 31 membership → February eligible, Feb 28 → March eligible, etc.
  - Universe updates month-by-month; NOT frozen per study period

- `src/krauss/data/prices_returns.py` — delisting-adjusted returns:
  - Standard CRSP convention: `adj_ret = (1 + ret) * (1 + dlret) - 1` on delisting date
  - If ret missing but dlret exists, uses dlret alone

- `scripts/build_data.py` — orchestration script (ran successfully)

- `configs/phase1_repro.yaml` — all dates/parameters:
  - raw_start_date: 1989-01-01 (need lookback room for 240-day features)
  - raw_end_date: 2015-12-31
  - study_start_date: 1992-01-01
  - study_end_date: 2015-10-30

**Data pull results (2026-04-06):**
- 2,064 membership spells, 1,936 unique PERMNOs ever in S&P 500
- 1,343 PERMNOs with CRSP daily data in 1989–2015 range (593 were members outside this window)
- 5,776,538 daily rows pulled; 5,764,478 after return cleaning
- 578 delisting events
- 162,012 stock-month eligibility rows, 324 effective months, avg 500 stocks/month
- 3,392,164 stock-day eligibility rows

**Output files:**
- `data/raw/sp500_membership.parquet` (0.0 MB)
- `data/raw/crsp_daily.parquet` (39.4 MB)
- `data/raw/crsp_delist.parquet` (0.0 MB)
- `data/processed/membership_monthly.parquet` (0.1 MB)
- `data/processed/universe_daily.parquet` (0.7 MB)
- `data/processed/daily_returns.parquet` (26.9 MB)

### Step 3: Feature and label generation
**Features (31 total per the paper):**
- R1–R20: simple returns over 1–20 day lookbacks
- R40, R60, R80, R100, R120, R140, R160, R180, R200, R220, R240: multi-period returns
- Definition: `R_{t,m} = P_t / P_{t-m} - 1` where P is a cumulative total-return price index reconstructed from daily holding-period returns
- Rows with < 240 trading days of history are dropped (feature lookback requirement)

**Labels:**
- `y_binary`: 1 if stock's next-day return > next-day cross-sectional median, else 0 (Phase 1 target)
- `u_excess`: next-day return minus next-day cross-sectional median (Phase 2 target)
- Cross-sectional median computed only over eligible stocks on the next day
- Features at date t use info through t; labels use return realized on t+1

**Scripts/modules:**
- `src/krauss/data/features.py` — `compute_lagged_returns()`
- `src/krauss/data/labels.py` — `compute_labels()`
- `scripts/build_features_labels.py` — orchestration with sanity checks

**Results (2026-04-06):**
- Features: 5,446,588 rows, 1,308 stocks, date range 1989-12-13 to 2015-12-31
- Labels: 3,386,718 rows, 1,135 stocks, date range 1989-02-01 to 2015-12-30
- `y_binary=1` rate: 0.4924 (expected ~0.50 — slight asymmetry is normal due to right-skewed returns)
- `u_excess` mean: 0.000434 (near zero as expected — small positive skew)
- `u_excess` median: 0.000000 (exactly zero by construction at the cross-sectional level)
- `u_excess` std: 0.020934

**Output files:**
- `data/processed/features.parquet` (1,391.5 MB)
- `data/processed/labels.parquet` (35.2 MB)

**Sanity check notes:**
- y_binary at 49.24% rather than exactly 50%: this is expected. The median splits the cross-section into two halves, but when the count is even, stocks exactly at the median get y_binary=0 (since we use strict >). Also, the eligible set can have odd counts.
- u_excess mean slightly positive: consistent with the fact that median < mean for right-skewed return distributions.
- Features table is larger than labels because features are computed for ALL ever-members (needed for lookback), while labels are restricted to eligible (S&P 500 member) stock-days only.

### Step 4: Backtest engine with dummy scorer validation
**Purpose:** Build the full backtest pipeline and validate correctness before plugging in real models.

**Modules built:**
- `src/krauss/data/study_periods.py` — partitions trading dates into rolling 750-train / 250-trade blocks
  - First 240 days of each training window consumed by feature lookback → 510 usable training days
  - 24 study periods total covering 1989–2015 (rolling windows, advance by 250 trade days)
- `src/krauss/backtest/ranking.py` — cross-sectional ranking, top-k / bottom-k selection
- `src/krauss/backtest/portfolio.py` — equal-weight dollar-neutral portfolio construction, daily rebalance
  - Long k stocks at +1/k, short k at -1/k
  - Portfolio return = mean(long returns) - mean(short returns)
- `src/krauss/backtest/costs.py` — turnover computation and transaction cost application (5 bps/half-turn)
  - Turnover = sum of |weight changes| day-over-day
  - Day-1 turnover = 2.0 (full build: k buys + k sells)
- `src/krauss/backtest/rebalance.py` — position change tracking (entries, exits, side-switches)
- `scripts/validate_backtest.py` — end-to-end validation with random dummy scorer

**Study periods generated:**

| Period | Train             | Trade             |
|--------|-------------------|-------------------|
| 0      | 1989-01-03 → 1991-12-18 | 1991-12-19 → 1992-12-14 |
| 1      | 1992-12-15 → 1995-12-01 | 1995-12-04 → 1996-11-26 |
| 2      | 1996-11-27 → 1999-11-17 | 1999-11-18 → 2000-11-13 |
| 3      | 2000-11-14 → 2003-11-11 | 2003-11-12 → 2004-11-09 |
| 4      | 2004-11-10 → 2007-11-01 | 2007-11-02 → 2008-10-29 |
| 5      | 1993-12-10 → 1996-11-26 | 1996-11-27 → 1997-11-21 |
| ...    | ...                     | ...                     |
| 23     | 2011-10-21 → 2014-10-15 | 2014-10-16 → 2015-10-13 |

Training windows overlap (each advances by 250 trade days). Last trade window ends 2015-10-13, matching the paper's Oct 2015 endpoint.

**Validation results (period 0, k=10, random scorer):**
- 124,704 stock-day predictions, 508 stocks, 250 trading days
- Exactly 10 long + 10 short every day: PASS
- Dollar neutral every day (max imbalance: 0.00): PASS
- No lookahead (all return dates > signal dates): PASS
- Day-1 turnover = 2.0 (correct for full portfolio build): PASS
- Net return < gross return (costs applied correctly): PASS
- Mean daily return: -0.034% pre-cost (near zero as expected for random): PASS
- Avg daily turnover: 3.91 (high, as expected for random — ~96% daily replacement)
- Avg new stocks per day: 9.6/10 on each side (random reshuffles almost entirely)

**Key design notes:**
- The backtest uses signal date t for ranking, realizes returns on t+1. This matches the paper's convention.
- Study periods use rolling windows: advance by 250 (trade_days), training windows overlap. This matches the paper's coverage through Oct 2015.
- Initially implemented as non-overlapping (advance by 1000), which only produced 6 periods ending Oct 2012. Fixed to rolling after identifying the coverage gap.

### Step 5: Phase 1 models
**Models implemented:**
- `src/krauss/models/rf_phase1.py` — RandomForestClassifier (sklearn)
  - 1000 trees, depth 20, sqrt(31)=5 features/split, seed=1, n_jobs=-1
  - Deviation: sklearn vs H2O
- `src/krauss/models/xgb_phase1.py` — XGBClassifier (xgboost)
  - 100 trees, depth 3, lr=0.1, 15/31 features/split, seed=1
  - Deviation: XGBoost vs H2O GBM/AdaBoost
- `src/krauss/models/dnn_phase1.py` — KraussDNN (PyTorch)
  - Architecture: 31-31-10-5-2 with maxout activation (2 channels)
  - Dropout: 0.5 hidden, 0.1 input
  - L1 reg: 1e-5, ADADELTA optimizer, 400 epochs max, early stopping
  - 2-class softmax output
  - Deviation: PyTorch vs H2O
- `src/krauss/models/ensembles_phase1.py` — ENS1/ENS2/ENS3
  - ENS1: equal-weight average of DNN/GBT/RAF probabilities
  - ENS2: Gini/AUC-weighted average (training period performance)
  - ENS3: rank-weighted average (training period rank)

**Orchestration:**
- `scripts/run_phase1.py` — trains all models across 23 study periods
  - Supports `--model rf/xgb/dnn/all` and `--periods 0 1 2 ...`
  - Saves trained models to `data/models/period_XX/`
  - Saves predictions to `data/processed/predictions_phase1.parquet`
  - Per-period metadata + ensemble weights saved as JSON

**Period 0 sanity checks (all passed):**
- RF: 0.70%/day pre-cost (paper: RF strongest base model) ✓
- XGB: 0.58%/day pre-cost ✓
- DNN: 0.49%/day pre-cost (paper: DNN weakest) ✓
- ENS1: 0.67%/day pre-cost ✓
- Classification accuracy: all >52% (>50% = better than random) ✓
- Model correlations: RF/XGB 0.82, RF/DNN 0.66, XGB/DNN 0.74 ✓
- Dollar neutral, exactly k per side, no lookahead: all PASS ✓

**Known issue:** macOS segfault when RF (n_jobs=-1, loky) followed by XGBoost (libomp) in same process. libomp is not fork-safe on macOS. Running on Windows resolves this.
