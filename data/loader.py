# -*- coding: utf-8 -*-
"""
data/loader.py - Data loading, encoding, splitting, and scaling.

Public API
----------
load_data() -> (X_cv, X_holdout, y_cv, y_holdout, le, scaler, df_eda)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DATA_PATH, SEED


def load_data():
    """
    Load data.csv, encode labels, perform stratified 80/20 split,
    fit a StandardScaler on the CV pool, and return everything.

    Returns
    -------
    X_cv            : pd.DataFrame  - raw CV features
    X_holdout       : pd.DataFrame  - raw holdout features
    y_cv            : np.ndarray    - encoded CV labels  (0=Healthy, 1=Alzheimer)
    y_holdout       : np.ndarray    - encoded holdout labels
    le              : LabelEncoder  - fitted encoder (le.classes_ for class names)
    scaler          : StandardScaler - fitted on X_cv only
    X_cv_scaled     : np.ndarray
    X_holdout_scaled: np.ndarray
    df_eda          : pd.DataFrame  - X_cv + 'Label' column (string) for EDA
    """
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["ID", "class"], errors="ignore")
    y_raw = df[["class"]]

    le = LabelEncoder()
    y  = le.fit_transform(y_raw.values.ravel())   # H -> 0, P -> 1

    print(f"Dataset loaded : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Classes        : {dict(zip(le.classes_, np.bincount(y)))}")

    X_cv, X_holdout, y_cv, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"Split          -> CV pool: {X_cv.shape[0]} | Holdout: {X_holdout.shape[0]}")

    scaler           = StandardScaler()
    X_cv_scaled      = scaler.fit_transform(X_cv)
    X_holdout_scaled = scaler.transform(X_holdout)

    df_eda = X_cv.copy()
    df_eda["Label"] = pd.Categorical(
        np.where(y_cv == 0, "Healthy", "Alzheimer"),
        categories=["Healthy", "Alzheimer"],
    )

    return (
        X_cv, X_holdout,
        y_cv, y_holdout,
        le, scaler,
        X_cv_scaled, X_holdout_scaled,
        df_eda,
    )
