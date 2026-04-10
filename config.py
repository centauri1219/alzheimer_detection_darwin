# -*- coding: utf-8 -*-
"""
config.py - Shared configuration for the PRML Alzheimer detection project.
All modules import from here to ensure consistent styling and paths.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
ROOT_DIR  = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data.csv"
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# -- Reproducibility -----------------------------------------------------------
SEED = 42

# -- Aesthetics ----------------------------------------------------------------
PALETTE  = {"Healthy": "#2196A3", "Alzheimer": "#E05A3A"}
COLORS   = [PALETTE["Healthy"], PALETTE["Alzheimer"]]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       150,
})


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Save *fig* to PLOTS_DIR/<filename>.png and close it."""
    out = PLOTS_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(ROOT_DIR)}")
