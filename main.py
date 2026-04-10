# -*- coding: utf-8 -*-
"""
main.py - Entry point for the PRML Alzheimer-detection project.

Run with:
    python main.py

All plots are saved to plots/ with descriptive filenames.
"""

import time
import torch

# -- project modules -----------------------------------------------------------
from config import PLOTS_DIR
from data.loader import load_data
from eda.analysis import run_eda
from dimred.reduction import run_dimred

from models import knn_model, svm_model, mlp_model, rf_model
from evaluation.compare import (
    run_cv,
    run_holdout,
    plot_comparisons,
)


def main():
    t0 = time.time()

    print("=" * 65)
    print("  PRML PROJECT - Alzheimer Detection via Handwriting Analysis")
    print("=" * 65)
    print(f"  PyTorch MLP will run on: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  All plots -> {PLOTS_DIR}\n")

    # -- 1. Load & preprocess data ---------------------------------------------
    (X_cv, X_holdout,
     y_cv, y_holdout,
     le, scaler,
     X_cv_scaled, X_holdout_scaled,
     df_eda) = load_data()

    # -- 2. Exploratory Data Analysis ------------------------------------------
    correlations = run_eda(X_cv, y_cv, df_eda)

    # -- 3. Dimensionality Reduction Analysis ----------------------------------
    n_pca = run_dimred(X_cv_scaled, y_cv,
                       feature_names=X_cv.columns.tolist())

    # -- 4. Individual Model Analyses ------------------------------------------
    shared = dict(
        X_cv_scaled      = X_cv_scaled,
        y_cv             = y_cv,
        X_holdout_scaled = X_holdout_scaled,
        y_holdout        = y_holdout,
        n_pca            = n_pca,
        le               = le,
    )
    knn_model.analyze(**shared)
    svm_model.analyze(**shared)
    mlp_model.analyze(**shared)
    rf_model.analyze(**shared)

    # -- 5. Cross-Validation (all 4 models) ------------------------------------
    df_cv = run_cv(X_cv_scaled, y_cv, n_pca)

    # -- 6. Holdout Evaluation (all 4 models) ----------------------------------
    df_holdout = run_holdout(
        X_cv_scaled, y_cv,
        X_holdout_scaled, y_holdout,
        n_pca, le,
    )

    # -- 7. Comparison Plots ----------------------------------------------------
    plot_comparisons(
        df_cv, df_holdout,
        X_cv_scaled, y_cv,
        X_holdout_scaled, y_holdout,
        n_pca, le,
    )

    # -- Done ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  All done in {elapsed:.1f}s")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
