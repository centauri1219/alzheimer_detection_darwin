# -*- coding: utf-8 -*-
"""
eda/analysis.py - Exploratory Data Analysis plots.

Public API
----------
run_eda(X_cv, y_cv, df_eda) -> correlations (pd.Series)
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from config import COLORS, PALETTE, save_fig


# -- helpers -------------------------------------------------------------------

def _get_group(col: str) -> str:
    """Strip trailing digit(s) to recover the handwriting metric name."""
    return re.sub(r"\d+$", "", col)


# -- main entry point ----------------------------------------------------------

def run_eda(X_cv: pd.DataFrame, y_cv, df_eda: pd.DataFrame):
    """
    Run and save all EDA visualisations.

    Parameters
    ----------
    X_cv   : raw (unscaled) CV feature DataFrame
    y_cv   : integer label array  (0=Healthy, 1=Alzheimer)
    df_eda : X_cv with an extra 'Label' string column

    Returns
    -------
    correlations : pd.Series - Pearson r of every feature with the target label
    """
    print("\n== EDA ==================================================")

    # -- 3.1 Class distribution ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df_eda["Label"].value_counts()

    axes[0].bar(counts.index, counts.values,
                color=COLORS, edgecolor="white", width=0.5)
    for i, (lbl, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 0.5, str(cnt),
                     ha="center", fontweight="bold", fontsize=13)
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Subjects")
    axes[0].set_ylim(0, counts.max() * 1.15)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=COLORS, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Balance (%)", fontsize=14, fontweight="bold")

    plt.suptitle("3.1 - Class Distribution",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "eda_01_class_distribution.png")

    # -- 3.2 Feature statistics ------------------------------------------------
    desc = X_cv.describe().T
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    axes[0].scatter(desc["mean"], desc["std"], alpha=0.5,
                    color=PALETTE["Alzheimer"], s=20)
    axes[0].set_title("Mean vs Std Dev (spread map)",
                       fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean")
    axes[0].set_ylabel("Std Dev")

    skewness = X_cv.skew().sort_values()
    axes[1].hist(skewness, bins=40, color=PALETTE["Healthy"],
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--",
                    linewidth=1.5, label="Symmetry line")
    axes[1].set_title("Distribution of Feature Skewness",
                       fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Skewness")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle("3.2 - Feature-Level Statistics Overview",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "eda_02_feature_statistics.png")

    # -- 3.3 KDE – top 10 discriminating features ------------------------------
    correlations = X_cv.corrwith(pd.Series(y_cv, index=X_cv.index))
    top_pos  = correlations.nlargest(5).index.tolist()
    top_neg  = correlations.nsmallest(5).index.tolist()
    top10    = top_neg + top_pos

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, feat in zip(axes.flatten(), top10):
        for label, color in PALETTE.items():
            subset = df_eda[df_eda["Label"] == label][feat]
            sns.kdeplot(subset, ax=ax, fill=True, color=color,
                        alpha=0.45, label=label, linewidth=1.5)
        corr_val = correlations[feat]
        ax.set_title(f"{feat}\n(r = {corr_val:.3f})", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)

    plt.suptitle("3.3 - KDE Distributions of Top 10 Discriminating Features",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_03_kde_top10_features.png")

    # -- 3.6 Correlation heatmap (top 20) --------------------------------------
    top20       = correlations.abs().nlargest(20).index.tolist()
    corr_matrix = X_cv[top20].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(
        "3.6 - Inter-Feature Correlation Heatmap (Top 20 Discriminating Features)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    save_fig(fig, "eda_04_correlation_heatmap.png")

    # -- 3.7 Target correlation bar chart --------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    corr_sorted = correlations.sort_values()
    bar_colors  = [
        PALETTE["Alzheimer"] if v > 0 else PALETTE["Healthy"]
        for v in corr_sorted.values
    ]
    ax.bar(range(len(corr_sorted)), corr_sorted.values,
           color=bar_colors, alpha=0.75, width=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([])
    ax.set_xlabel("Features (sorted by correlation)")
    ax.set_ylabel("Pearson r with Target")
    ax.set_title(
        "3.7 - All-Feature Correlation with Target Label\n"
        "(Red = positively linked to Alzheimer, Blue = negatively linked)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig(fig, "eda_05_target_correlation_bar.png")

    # -- 3.8 Feature group analysis --------------------------------------------
    groups         = pd.Series({col: _get_group(col) for col in X_cv.columns})
    group_mean_corr = correlations.groupby(groups).mean().sort_values()

    fig, ax = plt.subplots(figsize=(14, max(5, len(group_mean_corr) * 0.35)))
    bar_col = [
        PALETTE["Alzheimer"] if v > 0 else PALETTE["Healthy"]
        for v in group_mean_corr.values
    ]
    ax.barh(group_mean_corr.index, group_mean_corr.values,
            color=bar_col, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Pearson r with Target")
    ax.set_title("3.8 - Mean Discriminative Power by Handwriting Feature Group",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_06_feature_group_analysis.png")

    # -- 3.9 PCA & LDA projections ---------------------------------------------
    # NOTE: eda_07 (signal spread / IQR scatter) was removed – not informative.
    scaler_eda = StandardScaler()
    X_cv_sc    = scaler_eda.fit_transform(X_cv)

    pca_model = PCA(n_components=2)
    pca2      = pca_model.fit_transform(X_cv_sc)

    # Use eigen solver + Ledoit-Wolf shrinkage for well-conditioned LDA
    lda_model = LDA(n_components=1, solver="eigen", shrinkage="auto")
    lda1      = lda_model.fit_transform(X_cv_sc, y_cv).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for lbl, color, val in zip(["Healthy", "Alzheimer"], COLORS, [0, 1]):
        mask = y_cv == val
        axes[0].scatter(pca2[mask, 0], pca2[mask, 1],
                        c=color, label=lbl, alpha=0.65, s=55,
                        edgecolors="white", linewidths=0.5)
    axes[0].set_title("3.9a - PCA 2D Projection",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel(
        f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(
        f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[0].legend()

    jitter = np.random.default_rng(0).normal(0, 0.08, size=len(lda1))
    for lbl, color, val in zip(["Healthy", "Alzheimer"], COLORS, [0, 1]):
        mask = y_cv == val
        axes[1].scatter(lda1[mask], jitter[mask],
                        c=color, label=lbl, alpha=0.65, s=55,
                        edgecolors="white", linewidths=0.5)
    axes[1].set_title("3.9b - LDA 1D Projection (jittered for visibility)",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Linear Discriminant 1")
    axes[1].set_yticks([])
    axes[1].legend()

    plt.suptitle("3.9 - Dimensionality Reduction: Cluster Separation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eda_07_pca_lda_projections.png")

    # -- 3.12 Pairplot – top 4 features ---------------------------------------
    top4  = correlations.abs().nlargest(4).index.tolist()
    pp_df = df_eda[top4 + ["Label"]].copy()

    g = sns.pairplot(
        pp_df, hue="Label", palette=PALETTE,
        plot_kws=dict(alpha=0.55, s=40, edgecolor="white", linewidth=0.3),
        diag_kind="kde", diag_kws=dict(fill=True, alpha=0.5),
    )
    g.fig.suptitle("3.12 - Pairplot: Top 4 Discriminating Features",
                   fontsize=14, fontweight="bold", y=1.02)
    save_fig(g.fig, "eda_08_pairplot_top4.png")

    print("EDA complete - 8 plots saved.\n")
    return correlations
