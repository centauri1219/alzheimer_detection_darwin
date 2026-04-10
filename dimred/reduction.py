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

    # Rule-of-thumb cap: with n_train samples, don't use more than n_train//10
    # PCA components or else distance-based / linear models grossly overfit.
    # 95% is too greedy here (174 samples, 450 features → 87 dims!)
    n_at_80  = int(np.argmax(cum_var >= 0.80)) + 1
    n_cap    = max(5, X_cv_scaled.shape[0] // 10)   # e.g. 139//10 = 13
    n_components_pca = min(n_at_80, n_cap)
    var_retained = cum_var[n_components_pca - 1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(cum_var) + 1), cum_var,
                 color="steelblue", lw=2)
    axes[0].fill_between(range(1, len(cum_var) + 1), cum_var,
                         alpha=0.15, color="steelblue")
    axes[0].axhline(y=0.80, color="orange", linestyle="--",
                    label="80% variance")
    axes[0].axhline(y=0.95, color="red",    linestyle=":",
                    alpha=0.5, label="95% variance (reference)")
    axes[0].axvline(x=n_components_pca, color="green", linestyle="-",
                    linewidth=2,
                    label=f"Selected: {n_components_pca} components ({var_retained:.0f}% var)")
    axes[0].set_title("PCA Elbow Plot", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Variance")
    axes[0].legend()

    lda_viz = LDA(n_components=1, solver="eigen", shrinkage="auto").fit_transform(X_cv_scaled, y_cv)
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

    print(f"PCA: {n_components_pca} components selected "
          f"({var_retained:.1f}% variance; capped at n_train//10={n_cap}, "
          f"80%-threshold was {n_at_80} dims).")

    # -- LDA feature weights (top 20) -----------------------------------------
    import pandas as pd
    lda_full = LDA(n_components=1, solver="eigen", shrinkage="auto").fit(X_cv_scaled, y_cv)
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Weight":  np.abs(lda_full.coef_[0]),
    }).sort_values("Weight", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Weight", y="Feature", data=feat_imp.head(20),
                hue="Feature", palette="rocket", legend=False, ax=ax)
    ax.set_title("Top 20 Most Influential Features (LDA Weights)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "dimred_02_lda_feature_weights.png")

    print("Dimensionality-reduction analysis complete - 2 plots saved.\n")
    return n_components_pca
