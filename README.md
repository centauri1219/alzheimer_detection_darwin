# 🧠 Handwriting-Based Alzheimer's Disease Detection

> A machine-learning pipeline that classifies **Healthy** vs **Alzheimer's** subjects from kinematic handwriting features, using the [DARWIN dataset](https://archive.ics.uci.edu/dataset/732/darwin) from the UCI Machine Learning Repository.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Package Requirements](#package-requirements)
- [Run Instructions](#run-instructions)
- [Pipeline Summary](#pipeline-summary)

---

## Project Overview

Alzheimer's disease can subtly alter fine motor control, making handwriting analysis a promising non-invasive screening tool. This project explores that idea by building and comparing several classifiers on the **DARWIN** dataset:

| Property | Detail |
|---|---|
| **Dataset** | DARWIN — Diagnosis of Alzheimer's disease through handwriting analysis |
| **Source** | [UCI ML Repository (ID 732)](https://archive.ics.uci.edu/dataset/732/darwin) |
| **Instances** | 174 subjects (89 Healthy, 85 Alzheimer's) |
| **Features** | 451 kinematic handwriting features (air time, pressure, stroke velocity, etc.) |
| **Task** | Binary classification (Healthy / Alzheimer's) |

### Models

| Model | Implementation |
|---|---|
| **K-Nearest Neighbours (KNN)** | scikit-learn `KNeighborsClassifier` — distance-weighted, k-sensitivity analysis (k = 1…25) |
| **Support Vector Machine (SVM)** | scikit-learn `SVC` — linear kernel, C-regularisation sweep, kernel comparison (linear / RBF / poly) |
| **Random Forest (RF)** | scikit-learn `RandomForestClassifier` — 200 estimators, OOB error tracking, feature-importance ranking |
| **Multi-Layer Perceptron (MLP)** | PyTorch `nn.Module` — architecture: *Input → 64 → 32 → 2* with ReLU activations, 20% Dropout, Adam optimiser (lr = 0.002, weight decay 1e-4), CrossEntropyLoss |

### Feature Engineering & Dimensionality Reduction

- **PCA** — Principal Component Analysis retaining components up to an 80%-variance threshold (capped at `n_train // 10` to prevent overfitting on small datasets).
- **LDA** — Linear Discriminant Analysis projecting to a single discriminant axis (eigen solver + Ledoit-Wolf shrinkage for numerical stability with high-dimensional data).

All models are evaluated under **both** PCA and LDA reductions via **5-fold stratified cross-validation** and a held-out **20% test set**.

---

## Directory Structure

```
alzheimer_detection_darwin/
│
├── main.py                  # Entry point — runs the full pipeline
├── config.py                # Shared paths, seed, colour palette, plot helper
├── data.csv                 # DARWIN dataset (174 × 453)
│
├── data/
│   ├── __init__.py
│   └── loader.py            # Load CSV, encode labels, stratified split, StandardScaler
│
├── eda/
│   ├── __init__.py
│   └── analysis.py          # 8 EDA visualisations (class balance, KDE, heatmaps, pairplot …)
│
├── dimred/
│   ├── __init__.py
│   └── reduction.py         # PCA elbow analysis, LDA separation, feature-weight ranking
│
├── models/
│   ├── __init__.py
│   ├── knn_model.py         # KNN: k-sensitivity, uniform-vs-distance weighting
│   ├── svm_model.py         # SVM: C-sweep, kernel comparison
│   ├── rf_model.py          # RF: OOB error, feature importance, tree-count sensitivity
│   └── mlp_model.py         # MLP (PyTorch): loss curves, epoch sensitivity
│
├── evaluation/
│   ├── __init__.py
│   └── compare.py           # 5-fold CV, holdout eval, confusion matrices, decision boundaries
│
├── plots/                   # Auto-generated — all 25 diagnostic plots saved here
│
├── prmlproj.py              # Legacy single-file notebook export (reference only)
├── run_log.txt              # Sample console output from a full run
└── README.md                # ← You are here
```

---

## Package Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | KNN, SVM, Random Forest, PCA, LDA, metrics, preprocessing |
| `matplotlib` | Plotting framework |
| `seaborn` | Statistical visualisation layer |
| `torch` (PyTorch) | MLP model (GPU-accelerated when CUDA is available) |

### `requirements.txt`

Create a `requirements.txt` in the project root with the following content:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
```

> **Note:** For PyTorch with CUDA support, follow the [official install guide](https://pytorch.org/get-started/locally/) to select the correct wheel for your OS & CUDA version. The CPU-only wheel works out of the box for evaluation purposes.

---

## Run Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd alzheimer_detection_darwin
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

The script will:

1. **Load & preprocess** the DARWIN dataset (`data.csv`) — label-encode, stratified 80/20 split, standard-scale.
2. **Exploratory Data Analysis** — generate 8 plots (class distribution, KDE, correlation heatmap, pairplot, etc.).
3. **Dimensionality Reduction** — PCA elbow plot, LDA 1D separation, top-20 LDA feature weights.
4. **Individual Model Analyses** — hyperparameter sensitivity plots for KNN, SVM, RF, and MLP.
5. **Cross-Validation** — 5-fold stratified CV for all 4 models under both PCA and LDA.
6. **Holdout Evaluation** — final accuracy on the 20% test set, confusion matrices, and 2D decision boundaries.
7. **Comparison Plots** — side-by-side CV vs holdout bar charts and accuracy heatmap.

All **25 plots** are saved to the `plots/` directory. Console output logs accuracy tables and timing information.

---

## Pipeline Summary

```
data.csv
  │
  ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Data Loader │ ──▶ │  EDA (8 plots)   │ ──▶ │  Dim. Reduction      │
│  80/20 split │     │  Distributions,  │     │  PCA (elbow, 80%     │
│  StandardScl │     │  Correlations    │     │    variance cap)     │
└──────────────┘     └──────────────────┘     │  LDA (1D, shrinkage) │
                                               └──────────┬───────────┘
                                                          │
                          ┌───────────────────────────────┘
                          ▼
            ┌──────────────────────────┐
            │  Model Training & Eval   │
            │  ┌─────┐ ┌─────┐        │
            │  │ KNN │ │ SVM │        │
            │  └─────┘ └─────┘        │
            │  ┌─────┐ ┌─────┐        │
            │  │ RF  │ │ MLP │        │
            │  └─────┘ └─────┘        │
            │  5-fold CV + Holdout     │
            └──────────┬───────────────┘
                       │
                       ▼
            ┌──────────────────────────┐
            │  Comparison & Reporting  │
            │  Confusion matrices,     │
            │  Decision boundaries,    │
            │  CV vs Holdout summaries │
            └──────────────────────────┘
                       │
                       ▼
                   plots/  (25 PNGs)
```

---

## License

This project is developed for academic purposes as part of a **Pattern Recognition & Machine Learning (PRML)** course.

The DARWIN dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/732/darwin) under its original terms.
