# -*- coding: utf-8 -*-
"""
models/svm_model.py - Support Vector Machine individual analysis.

Public API
----------
train(X_tr, y_tr, **kwargs)  -> fitted SVC
predict(model, X_te)         -> np.ndarray of labels
analyze(...)                 -> None  (saves plots)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from config import SEED, save_fig


# -- public helpers ------------------------------------------------------------

def train(X_tr, y_tr, kernel="linear", C=0.5):
    model = SVC(kernel=kernel, C=C)
    model.fit(X_tr, y_tr)
    return model


def predict(model, X_te):
    return model.predict(X_te)


# -- individual analysis -------------------------------------------------------

def analyze(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout, n_pca, le):
    """
    Individual SVM analysis:
      (a) C-regularisation sweep (log-scale) for linear kernel
      (b) Kernel comparison (linear, rbf, poly) on 5-fold CV
    """
    print("\n-- SVM Analysis -------------------------------------------------")

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    C_vals = np.logspace(-3, 2, 20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        mean_accs, std_accs = [], []
        for C in C_vals:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                # LDA: eigen solver + Ledoit-Wolf shrinkage to handle high-dim data
                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1, solver="eigen", shrinkage="auto"))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                svm = SVC(kernel="linear", C=C)
                svm.fit(X_tr_r, y_tr)
                fold_accs.append(accuracy_score(y_val, svm.predict(X_val_r)))

            mean_accs.append(np.mean(fold_accs))
            std_accs.append(np.std(fold_accs))

        mean_accs = np.array(mean_accs)
        std_accs  = np.array(std_accs)
        best_C    = C_vals[int(np.argmax(mean_accs))]

        ax.semilogx(C_vals, mean_accs, marker="o", markersize=4,
                    color="#E05A3A", linewidth=2, label="CV mean accuracy")
        ax.fill_between(C_vals,
                        mean_accs - std_accs,
                        mean_accs + std_accs,
                        alpha=0.2, color="#E05A3A")
        ax.axvline(best_C, color="#2196A3", linestyle="--",
                   label=f"Best C = {best_C:.4f}")
        ax.set_title(f"SVM: C-Regularisation Sweep ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("C (regularisation strength, log scale)")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.legend(fontsize=9)
        # Adaptive ylim: zoom in on actual variation
        y_lo = max(0.0, float((mean_accs - std_accs).min()) - 0.04)
        y_hi = min(1.02, float((mean_accs + std_accs).max()) + 0.04)
        ax.set_ylim(y_lo, y_hi)

    plt.suptitle("SVM - Regularisation Sensitivity (Linear Kernel)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_svm_C_sweep.png")

    # -- kernel comparison (bar + std error bars) ------------------------------
    kernels = ["linear", "rbf", "poly"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        kernel_accs = {k: [] for k in kernels}
        for k in kernels:
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1, solver="eigen", shrinkage="auto"))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                svm = SVC(kernel=k, C=0.5)
                svm.fit(X_tr_r, y_tr)
                kernel_accs[k].append(accuracy_score(y_val, svm.predict(X_val_r)))

        kernel_colors = {"linear": "#E05A3A", "rbf": "#4CAF50", "poly": "#9C27B0"}
        means = [np.mean(kernel_accs[k]) for k in kernels]
        stds  = [np.std(kernel_accs[k], ddof=1) for k in kernels]

        bars = ax.bar(kernels, means,
                      yerr=stds, capsize=10,
                      color=[kernel_colors[k] for k in kernels], alpha=0.82,
                      error_kw=dict(linewidth=2, ecolor="dimgray"))
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    m + max(stds) * 0.15 + 0.005,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

        ax.set_title(f"SVM: Kernel Comparison ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("5-fold CV Accuracy (mean ± std)")
        y_top = min(1.05, max(means) + max(stds) + 0.08)
        ax.set_ylim(0.0, y_top)

    plt.suptitle("SVM - Kernel Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_svm_kernel_comparison.png")

    print("SVM individual analysis complete - 2 plots saved.")
