#!/usr/bin/env python3
"""
Precision vs Recall for SBS only, with F1 iso-lines.
Three panels: no noise, 5% noise, 10% noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

NOISE_PANELS = [
    ("clean",   "No noise"),
    ("noise5",  "5% noise"),
    ("noise10", "10% noise"),
]

F1_LEVELS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
F1_COLOR  = "#aaaaaa"

SC_COLOR  = "#1f77b4"
SPA_COLOR = "#ff7f0e"


def f1_recall_curve(f1: float, p_range: np.ndarray) -> np.ndarray:
    """Recall as a function of precision for a fixed F1 score."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (f1 * p_range) / (2 * p_range - f1)
    r[(r < 0) | (r > 1)] = np.nan
    return r


def plot_sbs(csv_path: Path, out_path: Path, dpi: int = 150) -> None:
    df = pd.read_csv(csv_path)
    sbs = df[df["mut_type"] == "SBS"].copy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(11, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    p_grid = np.linspace(0.01, 1.0, 500)

    for ax, (noise_key, title) in zip(axes, NOISE_PANELS):
        # ── Data points ───────────────────────────────────────────────────────
        row = sbs[sbs["noise"] == noise_key].iloc[0]

        ax.scatter(
            row["sc_mean_precision"], row["sc_mean_recall"],
            c=SC_COLOR, marker="o", s=120, zorder=4,
            edgecolors="white", linewidths=1.4,
            label="sigconfide",
        )
        ax.scatter(
            row["spa_mean_precision"], row["spa_mean_recall"],
            c=SPA_COLOR, marker="^", s=140, zorder=4,
            edgecolors="white", linewidths=1.4,
            label="SPA",
        )

        # Annotate F1 value next to each point
        sc_f1  = row["sc_mean_f1"]
        spa_f1 = row["spa_mean_f1"]
        ax.annotate(
            f"F1={sc_f1:.3f}",
            xy=(row["sc_mean_precision"], row["sc_mean_recall"]),
            xytext=(4, -10), textcoords="offset points",
            fontsize=7.5, color=SC_COLOR,
        )
        ax.annotate(
            f"F1={spa_f1:.3f}",
            xy=(row["spa_mean_precision"], row["spa_mean_recall"]),
            xytext=(4, 5), textcoords="offset points",
            fontsize=7.5, color=SPA_COLOR,
        )

        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.7)

    # Axis limits
    all_p = pd.concat([sbs["sc_mean_precision"], sbs["spa_mean_precision"]])
    all_r = pd.concat([sbs["sc_mean_recall"],    sbs["spa_mean_recall"]])
    pad  = 0.03
    lo_x = max(0, all_p.min() - pad)
    hi_x = min(1, all_p.max() + pad)
    lo_y = max(0, all_r.min() - pad)
    hi_y = min(1, all_r.max() + pad)
    for ax in axes:
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)

    fig.supxlabel("Precision (mean)", fontsize=11)
    fig.supylabel("Sensitivity (mean recall)", fontsize=11)
    fig.suptitle("SBS — sigconfide vs SPA", fontsize=12, fontweight="bold")

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SC_COLOR,
               markersize=9, markeredgecolor="white", label="sigconfide"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=SPA_COLOR,
               markersize=10, markeredgecolor="white", label="SPA"),
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path, default=Path("compare_spa_summary.csv"))
    p.add_argument("--output", type=Path, default=Path("compare_spa_sbs_f1.png"))
    p.add_argument("--dpi",    type=int,  default=150)
    args = p.parse_args()
    plot_sbs(args.input, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
