"""
Multitask DNN for Phase 2 extension.

Architecture: shared trunk (same as Phase 1) + two heads.
    Trunk:  31 -> Maxout(31) -> Maxout(10) -> Maxout(5)
    Classification head: Linear(5, 1) -> sigmoid -> P_t = P(U_t > 0)
    Regression head:     Linear(5, 1) -> U_t (excess return)

Loss:
    L = lambda * BCE(P_hat, 1{U_t > 0}) + (1 - lambda) * Huber(U_hat, U_t)

    where lambda = 0.5 (equal weight) by default.

Targets:
    U_t = r_{i,t+1} - median_j(r_{j,t+1})
    Classification target = 1{U_t > 0}

Training matches Phase 1 conventions:
    - Maxout with 2 channels per hidden unit
    - Input dropout 0.1, hidden dropout 0.5
    - L1 regularization 1e-5 (weights only)
    - ADADELTA optimizer (rho=0.99, eps=1e-8)
    - Same early stopping logic (sample-count-based scoring)
    - Seed fixed to 1
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


class MaxoutLayer(nn.Module):
    """Maxout activation with 2 channels per unit."""

    def __init__(self, in_features: int, out_features: int, n_channels: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features * n_channels)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(x.size(0), self.out_features, self.n_channels)
        out, _ = out.max(dim=2)
        return out


class MultitaskDNN(nn.Module):
    """
    Shared trunk + two heads for joint classification/regression.

    Trunk: 31 -> Maxout(31) -> Maxout(10) -> Maxout(5)
    Classification head: Linear(5, 1) -> P(U > 0)
    Regression head: Linear(5, 1) -> U_hat
    """

    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(p=0.1)

        self.h1 = MaxoutLayer(31, 31, n_channels=2)
        self.drop1 = nn.Dropout(p=0.5)

        self.h2 = MaxoutLayer(31, 10, n_channels=2)
        self.drop2 = nn.Dropout(p=0.5)

        self.h3 = MaxoutLayer(10, 5, n_channels=2)
        self.drop3 = nn.Dropout(p=0.5)

        # Classification head: P(U_t > 0)
        self.cls_head = nn.Linear(5, 1)

        # Regression head: U_t
        self.reg_head = nn.Linear(5, 1)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.drop1(self.h1(x))
        x = self.drop2(self.h2(x))
        x = self.drop3(self.h3(x))

        p_logit = self.cls_head(x).squeeze(-1)  # (batch,)
        u_hat = self.reg_head(x).squeeze(-1)     # (batch,)
        return p_logit, u_hat

    def predict(self, x):
        """Return P_hat and U_hat for inference."""
        self.eval()
        with torch.no_grad():
            p_logit, u_hat = self.forward(x)
            p_hat = torch.sigmoid(p_logit)
        return p_hat, u_hat


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_multitask_dnn(seed: int = SEED) -> MultitaskDNN:
    """Create the multitask DNN with paper-matched trunk."""
    _set_seed(seed)
    model = MultitaskDNN()
    for name, param in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
    return model


def train_multitask_dnn(
    model: MultitaskDNN,
    X_train: pd.DataFrame,
    y_cls_train: pd.Series,
    u_reg_train: pd.Series,
    cls_weight: float = 0.5,
    epochs: int = 400,
    batch_size: int = 1024,
    l1_lambda: float = 1e-5,
    score_every_n_samples: int = 750_000,
    scoring_window: int = 5,
    scoring_patience: int = 5,
    score_samples: int = 10000,
    seed: int = SEED,
) -> MultitaskDNN:
    """
    Train with joint BCE + Huber loss.

    Loss = cls_weight * BCE(p_logit, y_cls) + (1 - cls_weight) * Huber(u_hat, u_reg)

    Parameters
    ----------
    y_cls_train : pd.Series
        Binary target: 1{U_t > 0}.
    u_reg_train : pd.Series
        Continuous target: U_t = r_{i,t+1} - median_j(r_{j,t+1}).
    cls_weight : float
        Weight for classification loss (default 0.5).
    """
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = torch.tensor(
        X_train[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32
    ).to(device)
    y_cls = torch.tensor(
        y_cls_train.values.astype(np.float32), dtype=torch.float32
    ).to(device)
    u_reg = torch.tensor(
        u_reg_train.values.astype(np.float32), dtype=torch.float32
    ).to(device)

    n = len(X)
    reg_weight = 1.0 - cls_weight

    # Scoring subset
    rng = np.random.RandomState(seed)
    score_n = min(score_samples, n)
    score_idx = torch.tensor(
        rng.choice(n, size=score_n, replace=False), dtype=torch.long
    )
    X_score = X[score_idx]
    y_cls_score = y_cls[score_idx]
    u_reg_score = u_reg[score_idx]

    weight_params = [
        p for name, p in model.named_parameters() if "bias" not in name
    ]

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.99, eps=1e-8)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    huber_loss_fn = nn.HuberLoss(delta=1.0)

    import time as _time
    score_history = []
    best_avg_score = float("inf")
    best_state = {}
    no_improve_count = 0
    stopped = False

    t_start = _time.time()
    total_samples = 0
    last_score_samples = 0

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(
            seed + epoch
        ))
        X_epoch = X[perm]
        y_cls_epoch = y_cls[perm]
        u_reg_epoch = u_reg[perm]

        model.train()
        for i in range(0, n, batch_size):
            x_b = X_epoch[i : i + batch_size]
            y_b = y_cls_epoch[i : i + batch_size]
            u_b = u_reg_epoch[i : i + batch_size]

            optimizer.zero_grad()
            p_logit, u_hat = model(x_b)

            loss_cls = bce_loss_fn(p_logit, y_b)
            loss_reg = huber_loss_fn(u_hat, u_b)
            loss = cls_weight * loss_cls + reg_weight * loss_reg

            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in weight_params)
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            total_samples += len(x_b)

            if total_samples - last_score_samples >= score_every_n_samples:
                model.eval()
                with torch.no_grad():
                    p_logit_s, u_hat_s = model(X_score)
                    s_cls = bce_loss_fn(p_logit_s, y_cls_score).item()
                    s_reg = huber_loss_fn(u_hat_s, u_reg_score).item()
                    score_loss = cls_weight * s_cls + reg_weight * s_reg
                model.train()

                score_history.append(score_loss)
                last_score_samples = total_samples

                if len(score_history) >= scoring_window:
                    avg_score = np.mean(score_history[-scoring_window:])
                    if avg_score < best_avg_score:
                        best_avg_score = avg_score
                        best_state = {
                            k: v.cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                elapsed = _time.time() - t_start
                ep_frac = total_samples / n
                print(f"      MT-DNN score #{len(score_history)}: "
                      f"joint={score_loss:.6f} "
                      f"cls={s_cls:.6f} reg={s_reg:.6f} "
                      f"avg={np.mean(score_history[-scoring_window:]):.6f} "
                      f"best={best_avg_score:.6f} "
                      f"no_imp={no_improve_count}/{scoring_patience} "
                      f"ep~{ep_frac:.1f} "
                      f"[{elapsed:.0f}s]",
                      flush=True)

                if no_improve_count >= scoring_patience:
                    stopped = True
                    break

        if stopped:
            break

    elapsed = _time.time() - t_start
    ep_frac = total_samples / n
    print(f"      MT-DNN done: epochs~{ep_frac:.1f}, scores={len(score_history)}, "
          f"[{elapsed:.0f}s]", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model


def predict_multitask_dnn(
    model: MultitaskDNN,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate P_hat and U_hat predictions.

    Returns
    -------
    p_hat : np.ndarray
        P(U > 0) probability for each row.
    u_hat : np.ndarray
        Predicted excess return for each row.
    """
    device = next(model.parameters()).device
    model.eval()
    X = torch.tensor(
        X_test[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        p_logit, u_hat = model(X)
        p_hat = torch.sigmoid(p_logit)
    return p_hat.cpu().numpy(), u_hat.cpu().numpy()
