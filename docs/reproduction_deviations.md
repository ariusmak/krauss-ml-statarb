# Reproduction Deviations Log

## Krauss et al. (2017) EJOR — Phase 1 Reproduction

### Resolved Deviations (parity achieved)

| # | Item | Original deviation | Fix applied | Date |
|---|------|--------------------|-------------|------|
| 1 | ADADELTA rho/eps | PyTorch defaults (rho=0.9, eps=1e-6) vs H2O defaults (rho=0.99, eps=1e-8) | Explicitly set `rho=0.99, eps=1e-8` in `torch.optim.Adadelta` | 2026-04-07 |
| 2 | DNN bias initialization | PyTorch uniform init vs H2O zero init | Added `nn.init.zeros_` for all bias parameters in `build_dnn_model()` | 2026-04-07 |
| 3 | XGB min_child_weight | XGBoost default=1 vs H2O default min_rows=10 | Set `min_child_weight=10` in `XGBClassifier` | 2026-04-07 |
| 4 | DNN early stopping data | Was using 10% val holdout; H2O scores on training data | Changed to score on full training data with H2O-style time-based scoring | 2026-04-07 |
| 5 | DNN early stopping mechanism | Was epoch-level patience; H2O uses time-based (~5s) scoring events | Implemented wall-clock-based scoring every `scoring_interval_sec=5.0` seconds | 2026-04-07 |
| 6 | XGB depth convention | Confirmed H2O and XGBoost use same `max_depth` convention. No deviation. | N/A — was already correct | 2026-04-07 |
| 7 | DNN weight initialization | Confirmed PyTorch Kaiming uniform matches H2O uniform(-1/√fan_in, 1/√fan_in). No deviation. | N/A — was already correct | 2026-04-07 |
| 8 | RF sampling method | sklearn `bootstrap=True` (n samples WITH replacement) vs H2O `sample_rate=0.6320` (63.2% WITHOUT replacement) | Set `bootstrap=True, max_samples=0.6320` (sklearn 1.7 disallows max_samples with bootstrap=False; with replacement from 63.2% subset is practically equivalent for large n) | 2026-04-07 |
| 9 | RF split criterion | sklearn default `gini` vs H2O DRF default `entropy` | Set `criterion='entropy'` | 2026-04-07 |
| 10 | XGB L2 regularization (`reg_lambda`) | XGBoost default `reg_lambda=1`; H2O GBM has no L2 regularization | Set `reg_lambda=0` in `XGBClassifier` | 2026-04-07 |
| 11 | XGB L1 regularization (`reg_alpha`) | XGBoost default `reg_alpha=0`; H2O GBM has no L1 | Explicitly set `reg_alpha=0` (was already default, now explicit) | 2026-04-07 |
| 12 | XGB min split improvement (`gamma`) | XGBoost default `gamma=0`; H2O GBM `min_split_improvement=1e-5` | Set `gamma=1e-5` in `XGBClassifier` | 2026-04-07 |
| 13 | XGB histogram bins (`max_bin`) | XGBoost default `max_bin=256`; H2O GBM `nbins=20` | Set `max_bin=20` in `XGBClassifier` | 2026-04-07 |
| 14 | DNN L1 applied to biases | L1 penalty was applied to ALL parameters including biases; H2O applies L1 to weights only | Filtered `model.named_parameters()` to exclude `'bias'` from L1 sum | 2026-04-07 |
| 15 | ENS3 weighting formula | Was using `R_i / sum(R_j)` with rank 1=worst; paper Eq. 7 uses `(1/R_i) / sum(1/R_j)` with rank 1=best | Corrected to paper formula with descending rank (rank 1 = best Gini) | 2026-04-07 |
| 16 | DNN scoring sample size | Was scoring on full training set (~255K rows); H2O default `score_training_samples=10000` | Set `score_samples=10000`. Scoring on random 10K subset matches H2O behavior and frees more time for training | 2026-04-08 |
| 17 | DNN scoring interval | Was wall-clock-based (5s); PyTorch processes ~30x fewer samples/sec than H2O Java, giving only ~0.15 epochs between scores vs H2O's ~3 epochs | Changed to sample-count-based: score every 750K samples (~3 epochs), matching H2O's effective throughput per scoring interval | 2026-04-08 |

### Open Deviations (retraining pending or inherent to library choice)

**18. H2O silent input standardization (2026-05-11).** H2O's
`H2ODeepLearningEstimator` defaults to `standardize=True`, which z-scores
features per-column before training. The paper text (Section 4.2,
Equation 1) does not disclose this. Our PyTorch port trains on raw
simple returns, matching the paper's prose but not its silent H2O
behavior. Likely contributes to the Phase-1 DNN under-reproduction in
Pipeline B (63% of paper vs Pipeline A's 85%). Verified against H2O
Python API docs and h2o-py source.

---

## 1. Data-Source Deviations

### 1a. Primary pipeline: Datastream (paper parity)

As of 2026-04-09, the primary reproduction uses **Datastream data via WRDS** (`tr_ds_equities`), matching the paper's data source exactly.

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| **Data vendor** | Thomson Reuters Datastream | Thomson Reuters Datastream via WRDS |
| **Constituency source** | Datastream month-end S&P 500 constituent lists (Dec 1989 – Sep 2015) | `tr_ds_equities.ds2constmth` (indexlistintcode=4408) — spell-based, reconstructed to monthly snapshots |
| **Return source** | Datastream daily total return index (RI) | `tr_ds_equities.ds2primqtri` — daily RI values, returns computed as `RI_t / RI_{t-1} - 1` |
| **Canonical identifier** | Datastream infocode | Datastream infocode (same system) |
| **Trading calendar** | US trading days | Datastream filtered to CRSP US trading calendar (238 non-US holiday dates removed) |
| **Ever-members** | 1,322 unique infocodes | 1,322 unique infocodes (exact match) |
| **Avg stocks/month** | 499.7 (Table 1) | 499.5 |

**Calendar filtering:** Datastream reports RI values on non-US holidays (weekends excluded, but holidays like July 4th, Thanksgiving, etc. included with stale prices). These 238 extra dates produce 0% returns and inflate the trading day count. We filter to CRSP's US trading calendar (6,805 dates vs Datastream's 7,043) to ensure study period boundaries match the paper's.

**Impact level: LOW** — Same data vendor and source as the paper.

### 1b. Alternative pipeline: CRSP (for comparison)

The original CRSP-based pipeline is preserved for comparison purposes.

| Aspect | Paper | CRSP Pipeline |
|--------|-------|-------------|
| **Data vendor** | Thomson Reuters Datastream | WRDS CRSP |
| **Constituency source** | Datastream month-end lists | `crsp.dsp500list` — spell-based |
| **Return source** | Datastream daily RI | `crsp.dsf` — daily holding-period return |
| **Ever-members** | 1,322 (Datastream) | 1,936 (CRSP) |

**Key finding (2026-04-09):** CRSP identifies 614 more "ever in S&P 500" stocks than Datastream. This is the single largest driver of result differences between CRSP and Datastream pipelines. The extra CRSP stocks dilute the training signal and change which stocks appear in the top/bottom-k selections.

**CRSP pipeline results:** ENS1 pre-cost 0.0041/day (91% of paper). Datastream pipeline: 0.0042/day (94%).

**Impact level: MEDIUM** — CRSP pipeline is a viable alternative but systematically underperforms the paper, especially during crisis periods where constituency differences are largest.

---

## 2. Universe-Construction Decisions

**Month-end constituent rule:**
- For each calendar month M, we evaluate which PERMNOs are S&P 500 members as of the last calendar day of month M, using `crsp.dsp500list` spell data: `start <= month_end AND (ending >= month_end OR ending IS NULL)`.
- That set becomes the eligible trading universe for month M+1.
- Example: January 31 membership → February eligibility. February 28 membership → March eligibility.

**Daily eligibility from monthly panel:**
- Each trading date is mapped to its calendar month. The eligibility set for that month (determined by the prior month-end) applies to all trading dates in that month.

**Membership updated within study periods:**
- Yes. Universe updates month-by-month across both training and trading periods. The universe is NOT frozen for an entire 1000-day study period.

**Data availability rule:**
- A stock must be in the S&P 500 per the monthly membership rule AND have a non-missing CRSP return on that day to appear in the modeling panel.
- Features require 240 trading days of return history. Rows with insufficient lookback are dropped (handled by `compute_lagged_returns` which drops any row where any of the 31 features is NaN).

**Extra exclusions:**
- None beyond membership + return availability + feature completeness.

**Impact level: LOW** — Logic matches the paper's description. Any difference is from CRSP vs Datastream constituency lists.

---

## 3. Study-Period Construction

**Number of study periods:** 23

**Construction rule:**
- Each study period: 750 training days + 250 trading days = 1000 total days.
- Advance by 250 trading days (rolling windows — training periods overlap).
- First training window starts at the first trading day >= January 1, 1990.
- This matches the paper's Section 4.1: "We move the training-trading set forward by 250 days in a sliding-window approach, resulting in 23 non-overlapping batches."
- "Non-overlapping" refers to the trading periods; training periods do overlap.

**Feature lookback consumption:**
- The first 240 days of each 750-day training window are consumed by the longest feature (R240).
- Effective usable training days per period: 510.

**Per-period dates:**

| Period | Train Start | Train End | Usable Train Start | Trade Start | Trade End |
|--------|-----------|----------|-------------------|-----------|---------|
| 0 | 1990-01-02 | 1992-12-16 | 1990-12-12 | 1992-12-17 | 1993-12-13 |
| 1 | 1990-12-27 | 1993-12-13 | 1991-12-09 | 1993-12-14 | 1994-12-08 |
| ... | ... | ... | ... | ... | ... |
| 22 | 2011-10-25 | 2014-10-17 | 2012-10-08 | 2014-10-20 | 2015-10-15 |

Full period table saved in `data/models/period_XX/meta.json` for each period.

**Impact level: LOW** — Matches paper exactly (23 periods, 750+250, advance by 250).

---

## 4. Feature-Generation Parity

**31 features generated:**
- R1, R2, ..., R20 (daily simple returns over 1–20 day lookbacks)
- R40, R60, R80, R100, R120, R140, R160, R180, R200, R220, R240 (multi-period returns)

**Return formula:**
- `R_{t,m} = P_t / P_{t-m} - 1` where `P` is a cumulative total-return price index reconstructed from daily holding-period returns.
- This matches the paper's Equation (1).

**Scaling/normalization:**
- None applied. Features are raw returns, not standardized. This matches the paper (no mention of normalization).

**Missing feature handling:**
- Rows where ANY of the 31 features is NaN (due to insufficient lookback history) are dropped entirely.
- This is the 240-day lookback consumption described in the paper.

**Rows dropped:**
- 5,764,478 daily return observations → 5,446,588 rows with complete features (317,890 dropped, ~5.5%).
- These are exclusively the first ~240 trading days of each stock's return history.

**Impact level: LOW** — Direct implementation of the paper's formula.

---

## 5. Label-Construction Parity

**Binary target definition:**
- `y_binary = 1` if stock's next-day return > next-day cross-sectional median return; else 0.
- This matches the paper's definition exactly.

**Cross-sectional median universe:**
- The median is computed over all stocks that are eligible (S&P 500 member per monthly rule) on the NEXT day (t+1).
- Both the stock and the stocks forming the median must be eligible on date t+1.

**Tie handling at the median:**
- Stocks with next-day return exactly equal to the median get `y_binary = 0` (strict inequality: `>`, not `>=`).
- This creates a slight class imbalance (~49.24% class 1 vs ~50.76% class 0) rather than exactly 50/50.
- The paper does not specify tie handling.

**Rows lost:**
- Any stock-day where the next-day return is unavailable (last trading day of the stock's history) is dropped.
- 3,386,718 labeled stock-day rows out of 3,392,164 eligible stock-days (~99.8% retained).

**Class balance:**
- y_binary=1 rate: 49.24% overall (slight asymmetry expected due to right-skewed returns and tie handling).

**Impact level: LOW** — Direct implementation.

---

## 6. Model-Family Substitutions

### 6a. Primary models: H2O (paper parity)

As of 2026-04-09, the primary reproduction uses **H2O** — the same framework as the paper.

| Model | Paper Implementation | Our Implementation | Match |
|-------|--------------------|--------------------|-------|
| DNN | H2O Deep Learning | H2O `H2ODeepLearningEstimator` (v3.46) | Same framework, different version (paper ~v3.8-3.10 from 2016) |
| GBT | H2O GBM | H2O `H2OGradientBoostingEstimator` (v3.46) | Same framework |
| RAF | H2O DRF | H2O `H2ORandomForestEstimator` (v3.46) | Same framework |

**Impact level: LOW** — Same framework. Version difference (3.46 vs ~3.8) may affect DNN results due to internal ADADELTA implementation changes.

### 6b. Alternative models: Python stack (for comparison)

The Python-based models are preserved for Phase 2 extensions.

| Model | Paper Implementation | Python Implementation | Reason |
|-------|--------------------|--------------------|--------|
| DNN | H2O Deep Learning | PyTorch `KraussDNN` | Phase 2 multi-task DNN requires PyTorch |
| GBT | H2O GBM | XGBoost `XGBClassifier` | Phase 2 regression requires XGBoost |
| RAF | H2O DRF | scikit-learn `RandomForestClassifier` | Phase 2 regression requires sklearn |

**Impact level:**
- RAF: LOW (sklearn RF closely matches H2O DRF)
- GBT: LOW-MEDIUM (XGBoost with H2O-matched defaults performs similarly)
- DNN: MEDIUM (PyTorch ADADELTA batch-size constraints limit parity; see Section 8)

---

## 7. Hyperparameter Parity

### DNN

| Parameter | Paper (H2O) | Reproduction (PyTorch) | Match |
|-----------|------------|----------------------|-------|
| Architecture | 31-31-10-5-2 | 31-31-10-5-2 | ✓ |
| Activation | Maxout (2 channels) | Maxout (2 channels) | ✓ |
| Output | 2-class softmax | 2-class softmax | ✓ |
| Parameters | 2,746 | 2,746 | ✓ |
| Hidden dropout | 0.5 | 0.5 | ✓ |
| Input dropout | 0.1 | 0.1 | ✓ |
| L1 lambda | 1e-5 | 1e-5 | ✓ |
| Optimizer | ADADELTA | ADADELTA | ✓ |
| ADADELTA rho | 0.99 (H2O default) | 0.99 (explicit) | ✓ |
| ADADELTA epsilon | 1e-8 (H2O default) | 1e-8 (explicit) | ✓ |
| ADADELTA lr (rate) | 0.005 (H2O default) | 1.0 (PyTorch default) | ⚠ See note below |
| Rate annealing | 1e-6 per sample (H2O default) | Not applied | ⚠ See note below |
| Max epochs | 400 | 400 | ✓ |
| Seed | 1 | 1 | ✓ |
| Cores | Single core | Single thread (`torch.set_num_threads(1)`) | ✓ |

### GBT

| Parameter | Paper (H2O) | Reproduction (XGBoost) | Match |
|-----------|------------|----------------------|-------|
| n_estimators (M_GBT) | 100 | 100 | ✓ |
| max_depth (J_GBT) | 3 | 3 (same convention) | ✓ |
| learning_rate (λ_GBT) | 0.1 | 0.1 | ✓ |
| Feature subsampling (m_GBT) | 15 of 31 per split | `colsample_bynode=15/31` | ✓ |
| Row subsampling | Not used (features only) | `subsample=1.0` (default) | ✓ |
| min_child_weight | H2O default min_rows=10 | `min_child_weight=10` (explicit) | Matched |
| L2 regularization (reg_lambda) | H2O: none (0) | `reg_lambda=0` (explicit) | Matched |
| L1 regularization (reg_alpha) | H2O: none (0) | `reg_alpha=0` (explicit) | Matched |
| Min split improvement (gamma) | H2O: 1e-5 | `gamma=1e-5` (explicit) | Matched |
| Histogram bins (max_bin) | H2O: nbins=20 | `max_bin=20` (explicit) | Matched |
| Seed | 1 | 1 | ✓ |

**XGB depth-mapping ambiguity:** Both H2O and XGBoost define `max_depth` as the number of splits from root to leaf. `max_depth=3` = two-way interactions in both frameworks. No ambiguity.

### RAF

| Parameter | Paper (H2O) | Reproduction (sklearn) | Match |
|-----------|------------|----------------------|-------|
| n_estimators (B_RAF) | 1000 | 1000 | ✓ |
| max_depth (J_RAF) | 20 | 20 | ✓ |
| Feature subsampling (m_RAF) | ⌊√p⌋ = ⌊√31⌋ = 5 | `max_features=5` | ✓ |
| Sampling | H2O: 63.2% without replacement | `bootstrap=True, max_samples=0.6320` (with-replacement from 63.2% subset; practically equivalent for large n) | ≈ |
| Seed | 1 | 1 | ✓ |

---

## 8. DNN-Specific Training Deviations

| Aspect | Paper (H2O) | Reproduction (PyTorch) | Status |
|--------|------------|----------------------|--------|
| Framework | H2O Deep Learning (Java/JNI) | PyTorch | Substitution |
| Maxout | H2O built-in maxout | Custom `MaxoutLayer`: Linear(in, out*2) → reshape → max | Equivalent |
| ADADELTA rho | 0.99 | 0.99 (explicit, not PyTorch default of 0.9) | Matched |
| ADADELTA epsilon | 1e-8 | 1e-8 (explicit, not PyTorch default of 1e-6) | Matched |
| Device | CPU single core | CPU, `torch.set_num_threads(1)` | Matched |
| Weight initialization | H2O uniform(-1/√fan_in, 1/√fan_in) | PyTorch Kaiming uniform (same formula) | ✓ |
| Bias initialization | H2O: all biases = 0 | Explicit `nn.init.zeros_` on all biases | Matched |
| Batch size | H2O `mini_batch_size=1` (pure SGD) | `batch_size=1024` | ⚠ See note below |
| Shuffling | H2O default | DataLoader shuffle=True with seeded generator | Equivalent intent |
| Early stopping trigger | Moving avg of last 5 scoring events, stop after 5 non-improvements | Moving avg of last 5 scoring events, stop after 5 non-improvements | Matched |
| Early stopping scoring | H2O scores every ~5 seconds (time-based) | Wall-clock-based scoring every 5.0 seconds | Matched |
| Early stopping data | H2O `score_training_samples=10000` (random 10K subset) | Random 10K subset (`score_samples=10000`) | Matched |
| L1 penalty | Applied to weights only (not biases) | `sum(p.abs().sum() for weights only)` per batch, biases excluded | Matched |
| Validation split | None mentioned (H2O uses training data) | None (scores on full training data) | Matched |
| Determinism | Single core, seed=1 | `torch.manual_seed(1)`, `np.random.seed(1)`, `set_num_threads(1)` | Best effort |

**DNN batch size / ADADELTA rate investigation (2026-04-08):**

H2O defaults to `mini_batch_size=1` (pure SGD) and `rate=0.005` with `rate_annealing=1e-6`. These parameters interact with H2O's internal ADADELTA implementation (Java native loops, internal state management) in ways that do not translate to PyTorch's `torch.optim.Adadelta`. Configurations tested:

| batch_size | lr | Result |
|---|---|---|
| 1024 | 1.0 | **Best**: std ~0.012-0.017, model learns, predictions spread |
| 1 | 1.0 | Collapsed: std ~0.0002, ADADELTA accumulation explodes with single-sample noise |
| 1 | 0.005 | Near-constant: std ~0.005-0.008, model barely learns (lr too small) |
| 32 | 0.005 | Loss stuck at 0.692 (random), no learning |
| 32 | 1.0 | Diverging: loss increasing, model unstable |

Conclusion: PyTorch's ADADELTA requires batch_size ≥ ~1024 to produce stable, meaningful gradients. H2O's Java implementation handles single-sample SGD through internal mechanisms (rate scaling, native gradient accumulation) that have no PyTorch equivalent. This is an inherent framework deviation — the dominant remaining factor in DNN parity.

**Impact level: MEDIUM** — Framework substitution is the dominant factor. Batch size mismatch (1024 vs 1) affects gradient noise characteristics and prediction spread. DNN pre-cost return ~65% of paper; GBT/RAF ~95%.

---

## 9. Training Reproducibility Controls

| Control | Setting |
|---------|---------|
| Random seed | 1 (all models) |
| Device | CPU only |
| DNN threads | 1 (`torch.set_num_threads(1)`) |
| RF n_jobs | -1 (all cores — sklearn RF is deterministic with fixed seed regardless of n_jobs) |
| XGB n_jobs | -1 |
| Python version | 3.13 (ML env) / 3.12 (krauss env) |
| Key package versions | scikit-learn, xgboost, torch (versions logged at runtime) |
| OS | macOS (Darwin 25.2.0, arm64) |
| Deterministic mode | PyTorch: not using `torch.use_deterministic_algorithms(True)` (would require CUBLAS workspace config). Best-effort determinism via seed + single thread. |

---

## 10. Prediction and Ensemble Construction

**Probabilities saved:** Yes, for both train and trade panels per period. Trade predictions saved in `data/models/period_XX/predictions.parquet` and consolidated in `data/processed/predictions_phase1.parquet`. Train predictions used for ensemble weight computation and discarded after.

**ENS1 formula:**
- `P_ENS1 = (P_DNN + P_GBT + P_RAF) / 3`
- Matches paper Equation (5).

**ENS2 formula:**
- `w_i = g_i / (g_DNN + g_GBT + g_RAF)` where `g_i = 2 * AUC_i - 1` (Gini index)
- AUC computed on the training period using `sklearn.metrics.roc_auc_score`.
- Matches paper Equation (6).

**ENS3 formula:**
- `w_i = (1/R_i) / (1/R_DNN + 1/R_GBT + 1/R_RAF)` where `R_i` is the Gini-based performance rank (1=best, 3=worst), following Aiolfi & Timmermann (2006).
- The best-performing model (rank 1) gets weight 1/1.833 ≈ 0.545, middle gets 0.273, worst gets 0.182.
- Matches paper Equation (7).

**Degenerate cases:**
- If all Gini indices are 0 or negative, ENS2 falls back to ENS1 (equal weights).
- Negative Gini values are clipped to 0 before computing ENS2 weights.

**Impact level: LOW** — Direct implementation of paper formulas.

---

## 11. Trading-Rule Parity

| Aspect | Paper | Reproduction | Match |
|--------|-------|-------------|-------|
| Ranking | Cross-sectional descending by P(y=1) | `rank(method='first', ascending=False)` | ✓ |
| k values | {10, 50, 100, 150, 200} | {10, 50, 100, 150, 200} | ✓ |
| Daily reranking | Yes | Yes (rankings recomputed each day) | ✓ |
| Long selection | Top k | `rank <= k` | ✓ |
| Short selection | Bottom k | `rank > (n_stocks - k)` | ✓ |
| Tie handling | Not specified | `method='first'` (deterministic by input order) | ⚠ Unspecified in paper |
| Weight scheme | Equal weight | Long: +1/k, Short: -1/k | ✓ |
| Dollar neutral | Yes | Sum of weights = 0 every day (verified) | ✓ |
| Daily rebalance | Yes | Full rebuild each day | ✓ |

**Impact level: LOW** — Matched.

---

## 12. Execution Convention

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| Return alignment | "one-day-ahead trading signals" — signal at t, return realized at t+1 | Signal date t, next-day return from t+1 close to t+1 close | ✓ |
| Execution price | Implied close-to-close | Close-to-close (CRSP `ret` is close-to-close) | ✓ |
| Phase 1 mode | Paper-parity close-based | Close-based | ✓ |
| Realistic execution | Not tested in Phase 1 | Not tested (Phase 3 item per CLAUDE.md) | N/A |

The paper's return convention: features at date t use information through close of t. The prediction ranks stocks at t. Returns are realized from close of t to close of t+1. Our implementation matches: `next_day_ret` is the return from t to t+1, assigned to signal date t.

**Impact level: LOW** — Matched.

---

## 13. Transaction-Cost Modeling

| Aspect | Paper | Reproduction | Match |
|--------|-------|-------------|-------|
| Cost level | 0.05% per share per half-turn | 5 bps per half-turn | ✓ |
| Convention | Per half-turn (one side of trade) | Per half-turn | ✓ |
| Turnover definition | Not explicitly defined | Sum of absolute weight changes day-over-day | Standard |
| Cost application | Applied to portfolio return | `cost = turnover * 0.0005`; `net_ret = gross_ret - cost` | ✓ |
| Day-1 cost | Full portfolio build | Turnover = 2.0 on day 1 (buy k longs + sell k shorts) → cost = 10 bps | ✓ |
| Entry and exit | Both charged | Yes (turnover captures both sides) | ✓ |

**Impact level: LOW** — Standard implementation matching paper.

---

## 14. Per-Period Data Diagnostics

Saved in `data/models/period_XX/meta.json` for each of the 23 periods. Fields:
- `period_id`, `train_start`, `train_end`, `usable_train_start`
- `trade_start`, `trade_end`
- `n_train_obs`, `n_trade_obs`
- `n_train_stocks`, `n_trade_stocks`
- `n_train_days`, `n_trade_days`

Ensemble training-period AUC/Gini per model saved in `data/models/period_XX/ensemble_meta.json`.

---

## 15. Reproduction-Result Gaps

### Primary pipeline: Datastream + H2O (23 periods, k=10)

**Table 2 — Daily return characteristics (pre/post cost):**

| Model | Paper Pre | Ours Pre | Ratio | Paper Post | Ours Post | Ratio |
|-------|-----------|----------|-------|------------|-----------|-------|
| DNN | 0.0033 | 0.0028 | 85% | 0.0013 | 0.0017 | 133% |
| GBT | 0.0037 | 0.0039 | **106%** | 0.0017 | 0.0026 | **155%** |
| RAF | 0.0043 | 0.0040 | 93% | 0.0023 | 0.0027 | **116%** |
| ENS1 | 0.0045 | 0.0042 | **94%** | 0.0025 | 0.0030 | **120%** |

**Table 3 — Annualized risk-return (post-cost):**

| Metric | Paper ENS1 | Ours ENS1 |
|--------|-----------|-----------|
| Ann. return | 73.0% | 75.4% |
| Sharpe ratio | 1.81 | 2.23 |

**Table 5 — Sub-period analysis (ENS1 k=10, post-cost annualized):**

| Sub-period | Paper | Ours | Ratio |
|------------|-------|------|-------|
| 12/92–03/01 | 233.5% | 140.3% | 60% |
| 04/01–08/08 | 22.3% | 55.3% | 248% |
| 09/08–12/09 | 405.2% | 115.6% | 29% |
| 01/10–10/15 | -18.0% | -1.5% | 8% |

**Key observations:**
- Overall ENS1 pre-cost at **94%** of paper — strong reproduction.
- Post-cost metrics **exceed** paper because our turnover (~2.5) is lower than the paper's implied ~4.0. This is the H2O version difference: v3.46 may produce less noisy predictions than the 2016 version.
- GBT exceeds paper pre-cost (106%) — H2O GBM in v3.46 may have improved internals.
- DNN at 85% — weakest model, consistent with paper's observation that DNN is hardest to configure.
- Early period (12/92–03/01) at 60% of paper — strongest deviation. Likely due to pre-1992 constituency differences between Datastream versions.
- Crisis period (09/08–12/09) at 29% of paper — the paper's ~405% crisis returns are extreme and driven by the exact top/bottom-10 stock selection on a few key days (Oct 2008 alone: >100%).
- Moderation period exceeds paper (248%) — models perform better in this period than the paper reports.
- Late period matches paper's direction (negative after costs).

**Remaining gap sources (from investigation):**
1. H2O version (3.46 vs ~3.8-3.10): internal algorithm changes over 10 years
2. DNN: H2O Deep Learning internals changed between versions (ADADELTA implementation, dropout mechanics, early stopping behavior)
3. Datastream version: the Datastream constituent lists on WRDS may differ slightly from the 2016 Datastream download the paper used
4. Stochastic training: even with seed=1, Java runtime differences can affect results

### Alternative pipeline: CRSP + Python stack

| Model | Paper Pre | CRSP Pre | Ratio |
|-------|-----------|----------|-------|
| ENS1 | 0.0045 | 0.0041 | 91% |
| RAF | 0.0043 | 0.0041 | 95% |
| GBT | 0.0037 | 0.0035 | 96% |
| DNN | 0.0033 | 0.0021 | 63% |

The CRSP pipeline is systematically lower due to: (1) 614 extra "ever-member" stocks diluting signal, (2) different constituency boundaries, (3) PyTorch ADADELTA batch-size constraint for DNN.

---

## 16. Ambiguities Resolved by Implementation Choice

| Ambiguity | Paper Text | Our Resolution | Rationale |
|-----------|-----------|---------------|-----------|
| GBT algorithm | "H2O's implementation of AdaBoost, deploying shallow decision trees" | XGBoost stochastic gradient boosting | Agreed project substitution. H2O's AdaBoost variant is not available in standard Python. XGBoost is closest widely-used analogue. |
| GBT depth convention | J_GBT = 3 "allowing for two-way interactions" | `max_depth=3` in XGBoost | Both H2O and XGBoost count depth as splits from root. Depth 3 = two-way interactions in both. |
| Universe scope | "stocks considered in each batch are time-varying depending on index constituency and full data availability" | Monthly-updated S&P 500 membership + feature completeness per day | Paper could mean frozen-per-period or monthly-updated. We chose monthly-updated per the stated "time-varying" language. |
| DNN early stopping | "simple average of the loss over the last five scoring events does not improve for a total of five scoring events" | Wall-clock-based scoring every 5.0 seconds, moving average of last 5 scores, patience=5 | Matches H2O's time-based scoring logic. |
| DNN batch size | Not specified in paper | `batch_size=1024` (H2O default `mini_batch_size=1`) | H2O's ADADELTA internals (rate=0.005, rate_annealing, Java native loops) interact with small batches differently than PyTorch's. batch=1 and batch=32 both failed in PyTorch (ADADELTA accumulation dynamics collapse or diverge with noisy gradients). batch=1024 is the smallest size that allows PyTorch ADADELTA to converge. Logged as inherent framework deviation. |
| DNN initialization | Not specified | PyTorch default (Kaiming uniform) | H2O uses uniform initialization. Minor difference. |
| Tie handling in ranking | Not specified | `method='first'` (preserves input order for ties) | Deterministic but arbitrary. Affects very few stocks per day. |
| Cross-sectional median ties | Not specified | `y_binary = 0` for stocks exactly at median (strict `>`) | Conservative. Creates slight class imbalance (~49.2% vs 50.8%). |
| Execution timing | "one-day-ahead trading signals" | Close-to-close: features at t, trade at t, realize return t→t+1 | Standard interpretation of the paper's setup. |
| RF n_jobs | Paper uses single core | `n_jobs=-1` | sklearn RF is deterministic with fixed seed regardless of n_jobs. Single core would be prohibitively slow (~25 min/period). |
| DNN bias init | Not specified in paper | Set to 0 (H2O default) | H2O initializes all biases to 0. PyTorch default is uniform. Explicit fix applied. |
| XGB min_child_weight | Not specified in paper | Set to 10 (H2O min_rows default) | H2O GBM default min_rows=10. XGBoost default min_child_weight=1. Explicit fix applied. |
| XGB L2 regularization | Not specified in paper | Set `reg_lambda=0` (H2O GBM has none; XGBoost default=1 was silently regularizing) | Significant: XGB default L2 shrinks leaf weights, makes predictions conservative, reduces turnover. Fix applied 2026-04-07. |
| XGB histogram bins | Not specified in paper | Set `max_bin=20` (H2O GBM `nbins=20`; XGBoost default=256) | Coarser H2O bins produce different tree structures. Fix applied 2026-04-07. |
| XGB min split improvement | Not specified in paper | Set `gamma=1e-5` (H2O `min_split_improvement=1e-5`; XGBoost default=0) | Minor effect on tree structure. Fix applied 2026-04-07. |
| DNN L1 scope | Not specified in paper | L1 now applied to weights only, not biases (matching H2O behavior) | H2O applies L1 to weight matrices only. Over-regularizing biases reduced expressiveness. Fix applied 2026-04-07. |
| ENS3 rank formula | Paper Eq. 7: `(1/R_i) / sum(1/R_j)` with rank 1 = best | Corrected from `R_i / sum(R_j)` to paper formula | Old code gave weights 50/33/17 to best/mid/worst; correct is 54.5/27.3/18.2. Fix applied 2026-04-07. |

---

## 17. Null Test — P-gate(0.05) Extension Sharpe

ENS1 P-gate(0.05) earns post-cost Sharpe +0.94 on its 792 trading days
in the 2015-2025 extension. To distinguish signal from selection, we
ran a null test: 10,000 random 792-day subsamples of ENS1 P-only
extension returns.

| Statistic | Value |
|---|---|
| Observed gate Sharpe | 0.9348 |
| Null distribution mean | −0.4055 |
| Null distribution median | −0.3951 |
| 95th percentile of null | 0.3735 |
| 99th percentile of null | 0.6562 |
| **Percentile of observed gate Sharpe** | **99.83rd** |

Only 17 of 10,000 random subsamples produced a higher Sharpe.

**Interpretation:** strong evidence of signal. The gate identifies days
on which the model's prediction is informative, not a lucky 792-day
slice.

**Caveat:** this null tests day-selection, not stock-selection. It rules
out "the +0.94 is a sampling fluke" but is consistent with either
(i) a rare strong edge that the gate isolates, or (ii) a small
persistent edge that the gate scales by trading less often. Both are
favorable readings.

Code: `notebooks/null_test_pgate.ipynb`. Seed 42, 10,000 trials.
