"""
Gradient-Boosted Trees classifier for Phase 1 reproduction.

Paper parameters (H2O AdaBoost with shallow trees):
    - 100 trees (M_GBT = 100)
    - Max depth 3 (J_GBT = 3, allows two-way interactions)
    - Learning rate 0.1 (lambda_GBT = 0.1)
    - Feature subsampling per split: 15 out of 31 (~half)
    - Seed fixed to 1
    - XGBoost as agreed Python analogue
      (paper uses H2O GBM/AdaBoost — logged as reproduction deviation)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_xgb_model(seed: int = SEED) -> xgb.XGBClassifier:
    """
    Create an XGBClassifier with paper-matched parameters.
    """
    n_features = len(FEATURE_COLS)  # 31
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        colsample_bynode=15 / n_features,  # ~0.484, 15 of 31 features per split
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def train_xgb(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """
    Fit the XGB model on training data.
    """
    model.fit(X_train[FEATURE_COLS], y_train)
    return model


def predict_xgb(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate probability predictions (P(y=1)) for test data.
    """
    return model.predict_proba(X_test[FEATURE_COLS])[:, 1]
