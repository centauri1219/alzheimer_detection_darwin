# -*- coding: utf-8 -*-
"""
evaluation/compare.py - Cross-validation, holdout evaluation,
confusion matrices, decision boundaries, and all-model comparison plots.

Public API
----------
run_cv(X_cv_scaled, y_cv, n_pca)
    -> df_cv : pd.DataFrame

run_holdout(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout, n_pca, le)
    -> df_holdout : pd.DataFrame

plot_comparisons(df_cv, df_holdout,
                 X_cv_scaled, y_cv, n_pca, le)
    -> None  (saves plots)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import COLORS, PALETTE, SEED, save_fig
from models.mlp_model import train_and_predict as mlp_predict


# -- internal helpers ----------------------------------------------------------

def _get_reducer(tech: str, n_pca: int):
    return (PCA(n_components=n_pca) if tech == "PCA"
            else LDA(n_components=1))


def _fit_models(X_tr, y_tr, n_pca):
    """Fit all four models on (X_tr, y_tr) reduced by PCA and LDA."""
    fitted = {}
    for tech in ["PCA", "LDA"]:
        reducer = _get_reducer(tech, n_pca)
        X_r = reducer.fit_transform(X_tr, y_tr)

        knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_r, y_tr)
        svm = SVC(kernel="linear", C=0.5).fit(X_r, y_tr)
        rf  = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                     n_jobs=-1).fit(X_r, y_tr)
        fitted[tech] = dict(reducer=reducer, knn=knn, svm=svm, rf=rf,
                            X_r=X_r, y_tr=y_tr)
    return fitted


# -- public functions ----------------------------------------------------------

def run_cv(X_cv_scaled, y_cv, n_pca: int) -> pd.DataFrame:
    """5-fold stratified cross-validation for all 4 models × PCA/LDA."""
    print("\n== Cross-Validation =============================================")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv_scaled, y_cv), 1):
        X_tr, X_val = X_cv_scaled[tr_idx], X_cv_scaled[val_idx]
        y_tr, y_val = y_cv[tr_idx],         y_cv[val_idx]

        for tech in ["PCA", "LDA"]:
            reducer  = _get_reducer(tech, n_pca)
            X_tr_r   = reducer.fit_transform(X_tr, y_tr)
            X_val_r  = reducer.transform(X_val)

            knn   = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_tr_r, y_tr)
            svm   = SVC(kernel="linear", C=0.5).fit(X_tr_r, y_tr)
            rf    = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                           n_jobs=-1).fit(X_tr_r, y_tr)
            mlp_p = mlp_predict(X_tr_r, y_tr, X_val_r)

            for model_name, preds in [
                ("KNN",  knn.predict(X_val_r)),
                ("SVM",  svm.predict(X_val_r)),
                ("RF",   rf.predict(X_val_r)),
                ("MLP",  mlp_p),
            ]:
                rows.append({
                    "Fold":      fold,
                    "Technique": tech,
                    "Model":     model_name,
                    "Accuracy":  accuracy_score(y_val, preds),
                })

        print(f"  Fold {fold}/5 done.")

    df_cv = pd.DataFrame(rows)
    print("\n-- CV Mean Accuracy ---------------------------------------------")
    print(df_cv.groupby(["Technique", "Model"])["Accuracy"].mean().unstack())
    return df_cv


def run_holdout(X_cv_scaled, y_cv, X_holdout_scaled, y_holdout,
                n_pca: int, le) -> pd.DataFrame:
    """Final holdout evaluation for all 4 models × PCA/LDA."""
    print("\n== Holdout Evaluation ===========================================")
    rows = []

    for tech in ["PCA", "LDA"]:
        reducer    = _get_reducer(tech, n_pca)
        X_cv_r     = reducer.fit_transform(X_cv_scaled, y_cv)
        X_ho_r     = reducer.transform(X_holdout_scaled)

        knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_cv_r, y_cv)
        svm = SVC(kernel="linear", C=0.5).fit(X_cv_r, y_cv)
        rf  = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                     n_jobs=-1).fit(X_cv_r, y_cv)

        for model_name, preds in [
            ("KNN",  knn.predict(X_ho_r)),
            ("SVM",  svm.predict(X_ho_r)),
            ("RF",   rf.predict(X_ho_r)),
            ("MLP",  mlp_predict(X_cv_r, y_cv, X_ho_r)),
        ]:
            rows.append({
                "Technique":       tech,
                "Model":           model_name,
                "Holdout_Accuracy": accuracy_score(y_holdout, preds),
            })

    df_holdout = pd.DataFrame(rows)
    print(df_holdout.pivot(index="Model", columns="Technique",
                           values="Holdout_Accuracy").round(4))
    best = df_holdout.loc[df_holdout["Holdout_Accuracy"].idxmax()]
    print(f"\nBest: {best['Model']} + {best['Technique']} "
          f"-> Holdout accuracy: {best['Holdout_Accuracy']:.4f}")
    return df_holdout


def plot_comparisons(df_cv, df_holdout,
                     X_cv_scaled, y_cv,
                     X_holdout_scaled, y_holdout,
                     n_pca: int, le) -> None:
    """Generate and save all comparison plots."""
    print("\n== Generating Comparison Plots ==================================")

    # -- eval_01 CV accuracy bar ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x="Model", y="Accuracy", hue="Technique",
                data=df_cv, palette="coolwarm", ax=axes[0],
                order=["KNN", "SVM", "RF", "MLP"])
    axes[0].axhline(0.90, color="gray", linestyle="--",
                    alpha=0.5, label="90% line")
    axes[0].set_title("CV Accuracy: PCA vs LDA",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend()

    sns.boxplot(x="Model", y="Accuracy", hue="Technique",
                data=df_cv, palette="Set3", ax=axes[1],
                order=["KNN", "SVM", "RF", "MLP"])
    axes[1].set_title("Accuracy Distribution (5 folds)",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylim(0.5, 1.05)

    plt.suptitle("7. - Cross-Validation Results (All 4 Models)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eval_01_cv_accuracy_bar_box.png")

    # -- eval_02 Holdout heatmap ------------------------------------------------
    pivot = df_holdout.pivot(index="Model", columns="Technique",
                             values="Holdout_Accuracy")
    pivot = pivot.loc[["KNN", "SVM", "RF", "MLP"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu",
                vmin=0.7, vmax=1.0, linewidths=0.5,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)
    ax.set_title("Holdout Accuracy - All Models × Reduction Technique",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "eval_02_holdout_heatmap.png")

    # -- eval_03 Confusion matrices (best technique) ---------------------------
    best_row  = df_holdout.loc[df_holdout["Holdout_Accuracy"].idxmax()]
    best_tech = best_row["Technique"]

    reducer   = _get_reducer(best_tech, n_pca)
    X_cv_r    = reducer.fit_transform(X_cv_scaled, y_cv)
    X_ho_r    = reducer.transform(X_holdout_scaled)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_cv_r, y_cv)
    svm = SVC(kernel="linear", C=0.5).fit(X_cv_r, y_cv)
    rf  = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                 n_jobs=-1).fit(X_cv_r, y_cv)
    mlp_preds = mlp_predict(X_cv_r, y_cv, X_ho_r)

    all_preds  = [knn.predict(X_ho_r), svm.predict(X_ho_r),
                  rf.predict(X_ho_r), mlp_preds]
    model_lbls = ["KNN", "SVM", "RF", "MLP"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, m, p in zip(axes, model_lbls, all_preds):
        sns.heatmap(confusion_matrix(y_holdout, p),
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_,
                    yticklabels=le.classes_, ax=ax)
        acc = accuracy_score(y_holdout, p)
        ax.set_title(f"{m} ({best_tech})\nAcc={acc:.3f}",
                     fontweight="bold", fontsize=11)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.suptitle(f"Confusion Matrices - Holdout Set ({best_tech} reduction)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eval_03_confusion_matrices.png")

    # -- eval_04 Decision boundaries (2D PCA) ---------------------------------
    pca2     = PCA(n_components=2)
    X_cv_2d  = pca2.fit_transform(X_cv_scaled)

    knn_2d = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_cv_2d, y_cv)
    svm_2d = SVC(kernel="linear", C=0.5).fit(X_cv_2d, y_cv)
    rf_2d  = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                    n_jobs=-1).fit(X_cv_2d, y_cv)

    x_min = X_cv_2d[:, 0].min() - 1
    x_max = X_cv_2d[:, 0].max() + 1
    y_min = X_cv_2d[:, 1].min() - 1
    y_max = X_cv_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.15),
                         np.arange(y_min, y_max, 0.15))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    Z_knn = knn_2d.predict(mesh).reshape(xx.shape)
    Z_svm = svm_2d.predict(mesh).reshape(xx.shape)
    Z_rf  = rf_2d.predict(mesh).reshape(xx.shape)
    Z_mlp = mlp_predict(X_cv_2d, y_cv, mesh).reshape(xx.shape)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, m, Z in zip(axes, model_lbls, [Z_knn, Z_svm, Z_rf, Z_mlp]):
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_cv_2d[:, 0], X_cv_2d[:, 1], c=y_cv,
                   cmap="coolwarm", edgecolors="k", s=35, linewidths=0.4)
        ax.set_title(f"Decision Boundary: {m}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.suptitle("Model Decision Boundaries (2D PCA Projection)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eval_04_decision_boundaries.png")

    # -- eval_05 Model summary comparison (grouped bar + radar) ---------------
    # Compute summary: best accuracy per model (max over PCA/LDA)
    cv_summary = (df_cv.groupby(["Model", "Technique"])["Accuracy"]
                  .mean().reset_index())
    ho_summary = df_holdout.copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # -- grouped bar: CV mean accuracy
    sns.barplot(x="Model", y="Accuracy", hue="Technique",
                data=cv_summary, palette=["#2196A3", "#E05A3A"],
                ax=axes[0], order=["KNN", "SVM", "RF", "MLP"])
    axes[0].axhline(0.90, color="gray", linestyle=":", alpha=0.7)
    axes[0].set_title("Mean CV Accuracy by Model & Reduction",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylim(0.6, 1.05)
    axes[0].set_ylabel("5-fold CV Accuracy")

    # -- grouped bar: holdout accuracy
    sns.barplot(x="Model", y="Holdout_Accuracy", hue="Technique",
                data=ho_summary, palette=["#2196A3", "#E05A3A"],
                ax=axes[1], order=["KNN", "SVM", "RF", "MLP"])
    axes[1].axhline(0.90, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title("Holdout Accuracy by Model & Reduction",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylim(0.6, 1.05)
    axes[1].set_ylabel("Holdout Accuracy")

    plt.suptitle("All-Model Performance Summary - CV vs Holdout",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "eval_05_model_summary_cv_holdout.png")

    # -- eval_06 Radar chart ---------------------------------------------------
    models_order = ["KNN", "SVM", "RF", "MLP"]
    # Use best (max PCA/LDA) CV accuracy and holdout accuracy per model
    cv_best  = (df_cv.groupby(["Model", "Technique"])["Accuracy"]
                .mean().groupby("Model").max())
    ho_best  = df_holdout.groupby("Model")["Holdout_Accuracy"].max()

    categories   = ["CV Accuracy", "Holdout Accuracy"]
    model_colors = ["#2196A3", "#E05A3A", "#4CAF50", "#9C27B0"]
    angles       = np.linspace(0, 2 * np.pi, len(categories),
                               endpoint=False).tolist()
    angles      += angles[:1]          # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0.5, 1.0)

    for m, col in zip(models_order, model_colors):
        vals  = [cv_best.get(m, 0), ho_best.get(m, 0)]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=col, label=m)
        ax.fill(angles, vals, alpha=0.12, color=col)

    ax.set_title("Radar: CV vs Holdout Accuracy",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=11)
    plt.tight_layout()
    save_fig(fig, "eval_06_radar_cv_vs_holdout.png")

    print("Comparison plots complete - 6 plots saved.")
