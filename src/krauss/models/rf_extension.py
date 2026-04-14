"""
Random Forest extension models for Phase 2.

Separate classifier + regressor (no multi-task trees).
Same hyperparameters as Phase 1 RF where applicable.

Classifier: P(U_t > 0) — same as Phase 1 target
Regressor:  U_t = r_{i,t+1} - median_j(r_{j,t+1})
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_rf_classifier(seed: int = SEED) -> RandomForestClassifier:
    """RF classifier with Phase 1 hyperparameters."""
    n_features = len(FEATURE_COLS)
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        max_features=int(np.floor(np.sqrt(n_features))),  # 5
        bootstrap=True,
        max_samples=0.6320,
        criterion="entropy",
        random_state=seed,
        n_jobs=-1,
    )


def build_rf_regressor(seed: int = SEED) -> RandomForestRegressor:
    """RF regressor for U_t with matched hyperparameters."""
    n_features = len(FEATURE_COLS)
    return RandomForestRegressor(
        n_estimators=1000,
        max_depth=20,
        max_features=int(np.floor(np.sqrt(n_features))),  # 5
        bootstrap=True,
        max_samples=0.6320,
        criterion="squared_error",
        random_state=seed,
        n_jobs=-1,
    )


def train_rf_extension(
    clf: RandomForestClassifier,
    reg: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_cls_train: pd.Series,
    u_reg_train: pd.Series,
) -> tuple[RandomForestClassifier, RandomForestRegressor]:
    """Train both classifier and regressor."""
    clf.fit(X_train[FEATURE_COLS], y_cls_train)
    reg.fit(X_train[FEATURE_COLS], u_reg_train)
    return clf, reg


def predict_rf_extension(
    clf: RandomForestClassifier,
    reg: RandomForestRegressor,
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
