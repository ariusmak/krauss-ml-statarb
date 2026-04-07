"""
Random Forest classifier for Phase 1 reproduction.

Paper parameters:
    - 1000 trees (B_RAF = 1000)
    - Max depth 20 (J_RAF = 20)
    - Feature subsampling: floor(sqrt(p)) where p=31 -> 5
    - Seed fixed to 1
    - sklearn RandomForestClassifier as agreed Python analogue
      (paper uses H2O — logged as reproduction deviation)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_rf_model(seed: int = SEED) -> RandomForestClassifier:
    """
    Create a RandomForestClassifier with paper-matched parameters.
    """
    n_features = len(FEATURE_COLS)  # 31
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        max_features=int(np.floor(np.sqrt(n_features))),  # 5
        random_state=seed,
        n_jobs=-1,
    )


def train_rf(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit the RF on training data."""
    model.fit(X_train[FEATURE_COLS], y_train)
    return model


def predict_rf(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate probability predictions (P(y=1)) for test data.
    """
    return model.predict_proba(X_test[FEATURE_COLS])[:, 1]
