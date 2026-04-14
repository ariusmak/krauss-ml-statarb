"""
XGBoost extension models for Phase 2.

Separate classifier + regressor (no multi-task trees).
Same hyperparameters as Phase 1 XGB where applicable.

Classifier: P(U_t > 0) — same as Phase 1 target
Regressor:  U_t = r_{i,t+1} - median_j(r_{j,t+1})
"""

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_xgb_classifier(seed: int = SEED) -> xgb.XGBClassifier:
    """XGB classifier with Phase 1 hyperparameters."""
    n_features = len(FEATURE_COLS)
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        colsample_bynode=15 / n_features,
        min_child_weight=10,
        reg_lambda=0,
        reg_alpha=0,
        gamma=1e-5,
        max_bin=20,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def build_xgb_regressor(seed: int = SEED) -> xgb.XGBRegressor:
    """XGB regressor for U_t with matched hyperparameters."""
    n_features = len(FEATURE_COLS)
    return xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        colsample_bynode=15 / n_features,
        min_child_weight=10,
        reg_lambda=0,
        reg_alpha=0,
        gamma=1e-5,
        max_bin=20,
        objective="reg:pseudohubererror",
        random_state=seed,
        n_jobs=-1,
    )


def train_xgb_extension(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    y_cls_train: pd.Series,
    u_reg_train: pd.Series,
) -> tuple[xgb.XGBClassifier, xgb.XGBRegressor]:
    """Train both classifier and regressor."""
    clf.fit(X_train[FEATURE_COLS], y_cls_train)
    reg.fit(X_train[FEATURE_COLS], u_reg_train)
    return clf, reg


def predict_xgb_extension(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate P_hat and U_hat predictions.

    Returns
    -------
    p_hat : np.ndarray
        P(U > 0) for each row.
    u_hat : np.ndarray
        Predicted U_t for each row.
    """
    p_hat = clf.predict_proba(X_test[FEATURE_COLS])[:, 1]
    u_hat = reg.predict(X_test[FEATURE_COLS])
    return p_hat, u_hat
