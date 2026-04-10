# -*- coding: utf-8 -*-
"""
models/knn_model.py - K-Nearest Neighbours individual analysis.

Public API
----------
train(X_tr, y_tr, **kwargs)        -> fitted KNeighborsClassifier
predict(model, X_te)               -> np.ndarray of labels
analyze(X_cv_scaled, y_cv,
        X_holdout_scaled, y_holdout,
        n_pca, le)                 -> None  (saves plots)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from config import SEED, save_fig


# -- public helpers ------------------------------------------------------------

def train(X_tr, y_tr, n_neighbors=5, weights="distance"):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_tr, y_tr)
    return model


def predict(model, X_te):
    return model.predict(X_te)


# -- individual analysis -------------------------------------------------------

def analyze(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout, n_pca, le):
    """
    Individual KNN analysis:
      (a) k-sensitivity curve (k = 1..25) under both PCA and LDA on 5-fold CV
      (b) Effect of distance weighting  (uniform vs distance)
    """
    print("\n-- KNN Analysis -------------------------------------------------")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    k_values = list(range(1, 26))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        mean_accs, std_accs = [], []
        for k in k_values:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
                knn.fit(X_tr_r, y_tr)
                fold_accs.append(accuracy_score(y_val, knn.predict(X_val_r)))

            mean_accs.append(np.mean(fold_accs))
            std_accs.append(np.std(fold_accs))

        mean_accs = np.array(mean_accs)
        std_accs  = np.array(std_accs)
        best_k    = k_values[int(np.argmax(mean_accs))]

        ax.plot(k_values, mean_accs, marker="o", markersize=4,
                color="#2196A3", linewidth=2, label="CV mean accuracy")
        ax.fill_between(k_values,
                        mean_accs - std_accs,
                        mean_accs + std_accs,
                        alpha=0.2, color="#2196A3")
        ax.axvline(best_k, color="#E05A3A", linestyle="--",
                   label=f"Best k = {best_k}")
        ax.set_title(f"KNN: k-Sensitivity ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("k (number of neighbours)")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("KNN - Hyperparameter Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_knn_k_sensitivity.png")

    # -- weighting scheme comparison -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        results = {w: [] for w in ["uniform", "distance"]}
        for w in ["uniform", "distance"]:
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                knn = KNeighborsClassifier(n_neighbors=5, weights=w)
                knn.fit(X_tr_r, y_tr)
                results[w].append(accuracy_score(y_val, knn.predict(X_val_r)))

        ax.boxplot(
            [results["uniform"], results["distance"]],
            labels=["Uniform", "Distance"],
            patch_artist=True,
            boxprops=dict(facecolor="#2196A3", alpha=0.6),
            medianprops=dict(color="#E05A3A", linewidth=2),
        )
        ax.set_title(f"KNN: Weighting Scheme ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("KNN - Distance Weighting Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_knn_weighting_comparison.png")

    print("KNN individual analysis complete - 2 plots saved.")
