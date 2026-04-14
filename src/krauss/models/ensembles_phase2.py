"""
Phase 2 ensemble: ENS1 only.

ENS1 extension logic (per CLAUDE.md 6.6):
    1. Average P_hat across the three model families (DNN, XGB, RF).
    2. Average U_hat across the three model families.
    3. Compute three score families from the ensembled outputs:
       - P-only:    rank by P_hat_ens
       - U-only:    rank by U_hat_ens
       - Composite: rank by (2 * P_hat_ens - 1) * U_hat_ens
"""

import numpy as np


def ens1_p_hat(
    p_dnn: np.ndarray,
    p_xgb: np.ndarray,
    p_rf: np.ndarray,
) -> np.ndarray:
    """ENS1: equal-weight average of P_hat across model families."""
    return (p_dnn + p_xgb + p_rf) / 3.0


def ens1_u_hat(
    u_dnn: np.ndarray,
    u_xgb: np.ndarray,
    u_rf: np.ndarray,
) -> np.ndarray:
    """ENS1: equal-weight average of U_hat across model families."""
    return (u_dnn + u_xgb + u_rf) / 3.0


def composite_score(p_hat: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    """
    Composite ranking score: (2P - 1) * U.

    Maps P from [0, 1] to [-1, 1], then scales by predicted excess return.
    High composite = high confidence of positive excess * large expected magnitude.
    """
    return (2.0 * p_hat - 1.0) * u_hat


def compute_score_families(
    p_hat: np.ndarray,
    u_hat: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute all three score families from P_hat and U_hat.

    Returns
    -------
    dict with keys 'p_only', 'u_only', 'composite', each an ndarray of scores.
    """
    return {
        "p_only": p_hat,
        "u_only": u_hat,
        "composite": composite_score(p_hat, u_hat),
    }
