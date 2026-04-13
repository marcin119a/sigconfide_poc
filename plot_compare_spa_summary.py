#!/usr/bin/env python3
"""
Precision (X) vs sensitivity / mean recall (Y) from compare_spa_summary.csv.
Three panels: no noise, 5% noise, 10% noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

NOISE_PANELS = [
    ("clean", "No noise"),
    ("noise5", "5% noise"),
    ("noise10", "10% noise"),
]

MUT_ORDER = ["SBS", "DBS", "ID", "CN"]
MUT_COLORS = {
    "SBS": "#1f77b4",
    "DBS": "#ff7f0e",
    "ID": "#2ca02c",
    "CN": "#d62728",
}


def plot_summary(csv_path: Path, out_path: Path, dpi: int = 150) -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (noise_key, title) in zip(axes, NOISE_PANELS):
        sub = df[df["noise"] == noise_key]
        for _, row in sub.iterrows():
            mt = row["mut_type"]
            c = MUT_COLORS[mt]
            ax.scatter(
                row["sc_mean_precision"],
                row["sc_mean_recall"],
                c=c,
                marker="o",
                s=85,
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )
            ax.scatter(
                row["spa_mean_precision"],
                row["spa_mean_recall"],
                c=c,
                marker="^",
                s=100,
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

    # Shared axis limits with small margin
    all_x = pd.concat(
        [df["sc_mean_precision"], df["spa_mean_precision"]], ignore_index=True
    )
    all_y = pd.concat(
        [df["sc_mean_recall"], df["spa_mean_recall"]], ignore_index=True
    )
    pad = 0.02
    lo = min(all_x.min(), all_y.min()) - pad
    hi = max(all_x.max(), all_y.max()) + pad
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    fig.supxlabel("Precision (mean)", fontsize=11)
    fig.supylabel("Sensitivity (mean recall)", fontsize=11)

    # Legend: mutation types (color) + methods (markers)
    mut_handles = [
        mpatches.Patch(color=MUT_COLORS[m], label=m) for m in MUT_ORDER
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=9,
            markerfacecolor="gray",
            markeredgecolor="white",
            markeredgewidth=1,
            label="sigconfide",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="None",
            markersize=10,
            markerfacecolor="gray",
            markeredgecolor="white",
            markeredgewidth=1,
            label="SPA",
        ),
    ]

    fig.legend(
        handles=mut_handles + method_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
        title="Mutation type / method",
        title_fontsize=10,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("compare_spa_summary.csv"),
        help="Summary CSV from compare_spa.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("compare_spa_summary_pr.png"),
        help="Output image path",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    plot_summary(args.input, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
