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

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1))
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
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("SVM - Regularisation Sensitivity (Linear Kernel)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_svm_C_sweep.png")

    # -- kernel comparison ------------------------------------------------------
    kernels = ["linear", "rbf", "poly"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        kernel_accs = {k: [] for k in kernels}
        for k in kernels:
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                svm = SVC(kernel=k, C=0.5)
                svm.fit(X_tr_r, y_tr)
                kernel_accs[k].append(accuracy_score(y_val, svm.predict(X_val_r)))

        ax.boxplot(
            [kernel_accs[k] for k in kernels],
            labels=kernels,
            patch_artist=True,
            boxprops=dict(facecolor="#E05A3A", alpha=0.5),
            medianprops=dict(color="#2196A3", linewidth=2),
        )
        ax.set_title(f"SVM: Kernel Comparison ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("SVM - Kernel Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_svm_kernel_comparison.png")

    print("SVM individual analysis complete - 2 plots saved.")
