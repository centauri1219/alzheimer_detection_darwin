# -*- coding: utf-8 -*-
"""
models/mlp_model.py - PyTorch MLP individual analysis.

Public API
----------
train_and_predict(X_tr, y_tr, X_te, epochs=150) -> np.ndarray
analyze(...)                                     -> None  (saves plots)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import SEED, save_fig

# -- device --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- architecture --------------------------------------------------------------

class HandwritingMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# -- public helpers ------------------------------------------------------------

def train_and_predict(X_tr, y_tr, X_te, epochs=150, lr=0.002):
    """Train a fresh MLP on (X_tr, y_tr) and return predictions for X_te."""
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)

    model = HandwritingMLP(X_tr.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        crit(model(X_tr_t), y_tr_t).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        return torch.max(model(X_te_t), 1)[1].cpu().numpy()


# -- individual analysis -------------------------------------------------------

def _record_loss_curve(X_tr, y_tr, epochs=200, lr=0.002):
    """Return per-epoch training loss for one run."""
    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.LongTensor(y_tr).to(device)
    model = HandwritingMLP(X_tr.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def analyze(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout, n_pca, le):
    """
    Individual MLP analysis:
      (a) Training-loss curves under PCA and LDA projections
      (b) Epoch sensitivity: accuracy vs number of epochs (5-fold CV)
    """
    print("\n-- MLP Analysis -------------------------------------------------")
    print(f"   Device: {device}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # -- (a) loss curves --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        # Use the full CV set for a representative curve
        reducer = (PCA(n_components=n_pca) if tech == "PCA"
                   else LDA(n_components=1, solver="eigen", shrinkage="auto"))
        X_r = reducer.fit_transform(X_cv_scaled, y_cv)
        losses = _record_loss_curve(X_r, y_cv, epochs=200)

        ax.plot(losses, color="#9C27B0", linewidth=2)
        ax.set_title(f"MLP Training Loss Curve ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")

    plt.suptitle("MLP - Training Convergence",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_mlp_loss_curves.png")

    # -- (b) epoch sensitivity --------------------------------------------------
    epoch_vals = [25, 50, 75, 100, 150, 200, 300]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        mean_accs, std_accs = [], []
        for ep in epoch_vals:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                # LDA: eigen solver + Ledoit-Wolf shrinkage for high-dim stability
                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1, solver="eigen", shrinkage="auto"))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                preds = train_and_predict(X_tr_r, y_tr, X_val_r, epochs=ep)
                fold_accs.append(accuracy_score(y_val, preds))

            mean_accs.append(np.mean(fold_accs))
            std_accs.append(np.std(fold_accs))

        mean_accs = np.array(mean_accs)
        std_accs  = np.array(std_accs)

        ax.plot(epoch_vals, mean_accs, marker="s", markersize=6,
                color="#9C27B0", linewidth=2, label="CV mean accuracy")
        ax.fill_between(epoch_vals,
                        mean_accs - std_accs,
                        mean_accs + std_accs,
                        alpha=0.2, color="#9C27B0")
        ax.set_title(f"MLP: Epoch Sensitivity ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.legend(fontsize=9)
        # Adaptive ylim so epoch-to-epoch differences are visible
        y_lo = max(0.0, float((mean_accs - std_accs).min()) - 0.04)
        y_hi = min(1.02, float((mean_accs + std_accs).max()) + 0.04)
        ax.set_ylim(y_lo, y_hi)

    plt.suptitle("MLP - Epoch Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_mlp_epoch_sensitivity.png")

    print("MLP individual analysis complete - 2 plots saved.")
