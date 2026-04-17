#!/usr/bin/env python3
"""
Per-sample SBS comparison: sigconfide vs SPA.
Violin + box plots for precision, recall and F1 across noise conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

NOISE_ORDER = ["clean", "noise5", "noise10"]
NOISE_LABELS = {"clean": "No noise", "noise5": "5% noise", "noise10": "10% noise"}
METRICS = [
    ("sc_precision", "spa_precision", "Precision"),
    ("sc_recall",    "spa_recall",    "Recall"),
    ("sc_f1",        "spa_f1",        "F1"),
]

SC_COLOR  = "#1f77b4"
SPA_COLOR = "#ff7f0e"


def plot_sbs_method(csv_path: Path, out_path: Path, dpi: int = 150) -> None:
    df = pd.read_csv(csv_path)
    sbs = df[df["mut_type"] == "SBS"].copy()

    n_metrics = len(METRICS)
    n_noise   = len(NOISE_ORDER)

    fig, axes = plt.subplots(
        n_metrics, n_noise,
        figsize=(11, 9),
        sharex="col",
        constrained_layout=True,
    )

    positions_sc  = np.array([1])
    positions_spa = np.array([2])

    for col, noise_key in enumerate(NOISE_ORDER):
        sub = sbs[sbs["noise"] == noise_key]

        for row, (sc_col, spa_col, metric_label) in enumerate(METRICS):
            ax = axes[row, col]

            sc_vals  = sub[sc_col].dropna().values
            spa_vals = sub[spa_col].dropna().values

            # Violin
            parts = ax.violinplot(
                [sc_vals, spa_vals],
                positions=[1, 2],
                widths=0.6,
                showmedians=False,
                showextrema=False,
            )
            colors = [SC_COLOR, SPA_COLOR]
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_alpha(0.35)
                body.set_edgecolor(color)

            # Box on top of violin
            bp = ax.boxplot(
                [sc_vals, spa_vals],
                positions=[1, 2],
                widths=0.22,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2.2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(
                    marker=".", markersize=3, alpha=0.4,
                    markeredgewidth=0,
                ),
                notch=False,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
            for element in ["whiskers", "caps", "fliers"]:
                for item, color in zip(
                    bp[element][::2] + bp[element][1::2]
                    if element == "fliers"
                    else [bp[element][i] for i in range(len(bp[element]))],
                    [c for c in colors for _ in range(len(bp[element]) // 2)],
                ):
                    item.set_color(color)

            # Annotate medians
            for vals, pos, color in [
                (sc_vals,  1, SC_COLOR),
                (spa_vals, 2, SPA_COLOR),
            ]:
                med = np.median(vals)
                ax.text(
                    pos, med + 0.003, f"{med:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold",
                )

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["sigconfide", "SPA"], fontsize=9)
            ax.set_ylim(
                max(0, min(sc_vals.min(), spa_vals.min()) - 0.05),
                min(1.02, max(sc_vals.max(), spa_vals.max()) + 0.06),
            )
            ax.grid(True, axis="y", alpha=0.3, linestyle=":", linewidth=0.7)

            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if row == 0:
                ax.set_title(NOISE_LABELS[noise_key], fontsize=11)

    fig.suptitle("SBS — sigconfide vs SPA (per sample)", fontsize=13, fontweight="bold")

    handles = [
        mpatches.Patch(color=SC_COLOR,  label="sigconfide"),
        mpatches.Patch(color=SPA_COLOR, label="SPA"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path, default=Path("compare_spa_results.csv"))
    p.add_argument("--output", type=Path, default=Path("compare_spa_sbs_method.png"))
    p.add_argument("--dpi",    type=int,  default=150)
    args = p.parse_args()
    plot_sbs_method(args.input, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
