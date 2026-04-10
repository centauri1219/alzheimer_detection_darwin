# -*- coding: utf-8 -*-
"""
models/rf_model.py - Random Forest individual analysis.

Public API
----------
train(X_tr, y_tr, **kwargs)  -> fitted RandomForestClassifier
predict(model, X_te)         -> np.ndarray of labels
analyze(...)                 -> None  (saves plots)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import SEED, save_fig


# -- public helpers ------------------------------------------------------------

def train(X_tr, y_tr, n_estimators=200, max_depth=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def predict(model, X_te):
    return model.predict(X_te)


# -- individual analysis -------------------------------------------------------

def analyze(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout, n_pca, le):
    """
    Individual Random Forest analysis:
      (a) OOB error vs number of trees (raw features - RF handles high-dim well)
      (b) Feature importance (top 20) from a model trained on full data
      (c) n_estimators sensitivity on 5-fold CV under PCA and LDA
    """
    print("\n-- Random Forest Analysis ---------------------------------------")

    # -- (a) OOB error vs n_estimators (raw scaled features) ------------------
    n_tree_vals = list(range(10, 310, 20))
    oob_errors  = []

    for n in n_tree_vals:
        rf = RandomForestClassifier(
            n_estimators=n, oob_score=True,
            random_state=SEED, n_jobs=-1
        )
        rf.fit(X_cv_scaled, y_cv)
        oob_errors.append(1.0 - rf.oob_score_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_tree_vals, oob_errors, marker="o", markersize=4,
            color="#4CAF50", linewidth=2)
    ax.set_title("Random Forest: OOB Error vs Number of Trees",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("OOB Error Rate")
    ax.axvline(n_tree_vals[int(np.argmin(oob_errors))],
               color="#E05A3A", linestyle="--",
               label=f"Best n = {n_tree_vals[int(np.argmin(oob_errors))]}")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "model_rf_oob_error.png")

    # -- (b) Feature importance (top 20, raw features) -------------------------
    import pandas as pd
    rf_full = RandomForestClassifier(
        n_estimators=200, oob_score=True,
        random_state=SEED, n_jobs=-1
    )
    rf_full.fit(X_cv_scaled, y_cv)
    imp = pd.Series(rf_full.feature_importances_).nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    imp.sort_values().plot.barh(ax=ax, color="#4CAF50", alpha=0.8)
    ax.set_title("Random Forest: Top 20 Feature Importances (Gini)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    save_fig(fig, "model_rf_feature_importance.png")

    # -- (c) n_estimators sensitivity under PCA / LDA -------------------------
    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_tree_sweep = [10, 25, 50, 100, 150, 200, 300]
    fig, axes    = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tech in zip(axes, ["PCA", "LDA"]):
        mean_accs, std_accs = [], []
        for n in n_tree_sweep:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_cv_scaled, y_cv):
                X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
                y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

                reducer = (PCA(n_components=n_pca) if tech == "PCA"
                           else LDA(n_components=1))
                X_tr_r  = reducer.fit_transform(X_tr, y_tr)
                X_val_r = reducer.transform(X_val)

                rf = RandomForestClassifier(
                    n_estimators=n, random_state=SEED, n_jobs=-1
                )
                rf.fit(X_tr_r, y_tr)
                fold_accs.append(accuracy_score(y_val, rf.predict(X_val_r)))

            mean_accs.append(np.mean(fold_accs))
            std_accs.append(np.std(fold_accs))

        mean_accs = np.array(mean_accs)
        std_accs  = np.array(std_accs)
        best_n    = n_tree_sweep[int(np.argmax(mean_accs))]

        ax.plot(n_tree_sweep, mean_accs, marker="s", markersize=6,
                color="#4CAF50", linewidth=2, label="CV mean accuracy")
        ax.fill_between(n_tree_sweep,
                        mean_accs - std_accs,
                        mean_accs + std_accs,
                        alpha=0.2, color="#4CAF50")
        ax.axvline(best_n, color="#E05A3A", linestyle="--",
                   label=f"Best n = {best_n}")
        ax.set_title(f"RF: n_estimators Sensitivity ({tech})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("Random Forest - Tree-Count Sensitivity",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "model_rf_n_estimators_sensitivity.png")

    print("Random Forest individual analysis complete - 3 plots saved.")
