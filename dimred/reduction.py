# -*- coding: utf-8 -*-
"""
dimred/reduction.py - PCA and LDA dimensionality-reduction analysis.

Public API
----------
run_dimred(X_cv_scaled, y_cv) -> n_components_pca (int)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import COLORS, save_fig


def run_dimred(X_cv_scaled, y_cv, feature_names):
    """
    Fit PCA and LDA on the scaled CV data, produce diagnostic plots,
    and return the number of PCA components that explain ≥95% variance.

    Parameters
    ----------
    X_cv_scaled  : np.ndarray
    y_cv         : np.ndarray
    feature_names: list[str]  - original feature column names

    Returns
    -------
    n_components_pca : int
    """
    print("\n== Dimensionality Reduction =====================================")

    # -- PCA elbow + LDA separation --------------------------------------------
    pca_full = PCA().fit(X_cv_scaled)
    cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_pca = int(np.argmax(cum_var >= 0.95)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(cum_var) + 1), cum_var,
                 color="steelblue", lw=2)
    axes[0].fill_between(range(1, len(cum_var) + 1), cum_var,
                         alpha=0.15, color="steelblue")
    axes[0].axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
    axes[0].axvline(x=n_components_pca, color="green", linestyle=":",
                    label=f"Elbow: {n_components_pca} components")
    axes[0].set_title("PCA Elbow Plot", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Variance")
    axes[0].legend()

    lda_viz = LDA(n_components=1).fit_transform(X_cv_scaled, y_cv)
    for lbl, color, val in zip(["Healthy", "Alzheimer"], COLORS, [0, 1]):
        sns.kdeplot(lda_viz[y_cv == val].flatten(), fill=True, color=color,
                    label=lbl, ax=axes[1], alpha=0.5)
    axes[1].set_title("LDA: 1D Class Separation",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Linear Discriminant 1")
    axes[1].legend()

    plt.suptitle("5. - Dimensionality Reduction",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "dimred_01_pca_elbow_lda_separation.png")

    print(f"PCA: {n_components_pca} components retain 95% variance.")

    # -- LDA feature weights (top 20) -----------------------------------------
    import pandas as pd
    lda_full = LDA(n_components=1).fit(X_cv_scaled, y_cv)
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Weight":  np.abs(lda_full.coef_[0]),
    }).sort_values("Weight", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Weight", y="Feature", data=feat_imp.head(20),
                palette="rocket", ax=ax)
    ax.set_title("Top 20 Most Influential Features (LDA Weights)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "dimred_02_lda_feature_weights.png")

    print("Dimensionality-reduction analysis complete - 2 plots saved.\n")
    return n_components_pca
