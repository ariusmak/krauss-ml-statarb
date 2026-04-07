"""
Deep Neural Network classifier for Phase 1 reproduction.

Paper parameters (H2O DNN):
    - Architecture: 31-31-10-5-2 (input-H1-H2-H3-output)
    - Maxout activation: f(a1, a2) = max(a1, a2) with two channels
    - Output: 2-class softmax
    - Hidden dropout ratio: 0.5
    - Input dropout ratio: 0.1
    - L1 regularization: 1e-5
    - Optimizer: ADADELTA
    - Up to 400 epochs
    - Early stopping
    - Seed fixed to 1

Maxout implementation note:
    Each maxout unit takes two linear projections and returns the max.
    For a layer with k maxout units, the weight matrix projects from
    input_dim to 2*k, then we reshape and take max over pairs.
    This doubles the parameter count per hidden layer vs standard ReLU.

    Paper's parameter count (2746) is consistent with this.

Reproduction deviation: PyTorch implementation vs H2O.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


class MaxoutLayer(nn.Module):
    """
    Maxout activation with 2 channels per unit.

    For k output units: linear projects to 2*k, then max over pairs.
    """

    def __init__(self, in_features: int, out_features: int, n_channels: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features * n_channels)

    def forward(self, x):
        out = self.linear(x)
        # Reshape to (batch, out_features, n_channels) and take max
        out = out.view(x.size(0), self.out_features, self.n_channels)
        out, _ = out.max(dim=2)
        return out


class KraussDNN(nn.Module):
    """
    31-31-10-5-2 DNN with maxout hidden activations and softmax output.

    Matches the paper's architecture:
        Input (31) -> Maxout H1 (31) -> Maxout H2 (10) -> Maxout H3 (5) -> Softmax (2)

    With dropout:
        Input dropout: 0.1
        Hidden dropout: 0.5
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

        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.drop1(self.h1(x))
        x = self.drop2(self.h2(x))
        x = self.drop3(self.h3(x))
        x = self.output(x)  # raw logits for cross-entropy
        return x

    def predict_proba(self, x):
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dnn_model(seed: int = SEED) -> KraussDNN:
    """Create the DNN with paper architecture."""
    _set_seed(seed)
    model = KraussDNN()
    return model


def train_dnn(
    model: KraussDNN,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epochs: int = 400,
    batch_size: int = 1024,
    l1_lambda: float = 1e-5,
    patience: int = 20,
    val_fraction: float = 0.1,
    seed: int = SEED,
) -> KraussDNN:
    """
    Train the DNN with ADADELTA, cross-entropy loss, L1 regularization,
    and early stopping.

    Parameters
    ----------
    model : KraussDNN
    X_train : pd.DataFrame
        Feature columns.
    y_train : pd.Series
        Binary target.
    epochs : int
        Maximum number of epochs.
    batch_size : int
    l1_lambda : float
        L1 regularization strength.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    val_fraction : float
        Fraction of training data held out for early stopping validation.
    seed : int

    Returns
    -------
    Trained KraussDNN
    """
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = torch.tensor(
        X_train[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32
    )
    y = torch.tensor(y_train.values.astype(np.int64), dtype=torch.long)

    # Train/val split for early stopping
    n = len(X)
    n_val = max(int(n * val_fraction), 1)
    indices = torch.randperm(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # ADADELTA optimizer (paper uses H2O's ADADELTA)
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # L1 regularization
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

        # Validation for early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model


def predict_dnn(
    model: KraussDNN,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate probability predictions (P(y=1)) for test data.

    Returns
    -------
    np.ndarray
        Probability of class 1 for each row.
    """
    device = next(model.parameters()).device
    model.eval()
    X = torch.tensor(
        X_test[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)
    return probs[:, 1].cpu().numpy()
