#!/usr/bin/env python3
"""
Precision vs Sensitivity (mean recall) from compare_spa_summary.csv.
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
    ("clean",   "No noise"),
    ("noise5",  "5% noise"),
    ("noise10", "10% noise"),
]

MUT_ORDER  = ["SBS", "DBS", "ID", "CN"]
MUT_COLORS = {
    "SBS": "#1f77b4",
    "DBS": "#ff7f0e",
    "ID":  "#2ca02c",
    "CN":  "#d62728",
}


def plot_summary(csv_path: Path, out_path: Path, dpi: int = 150) -> None:
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(11, 4.2),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    for ax, (noise_key, title) in zip(axes, NOISE_PANELS):
        sub = df[df["noise"] == noise_key]
        for _, row in sub.iterrows():
            mt = row["mut_type"]
            c  = MUT_COLORS.get(mt, "black")
            ax.scatter(
                row["sc_mean_precision"], row["sc_mean_recall"],
                c=c, marker="o", s=85, zorder=3,
                edgecolors="white", linewidths=1.2,
            )
            if pd.notna(row.get("spa_mean_precision")):
                ax.scatter(
                    row["spa_mean_precision"], row["spa_mean_recall"],
                    c=c, marker="^", s=100, zorder=3,
                    edgecolors="white", linewidths=1.2,
                )
            ax.annotate(
                mt,
                xy=(row["sc_mean_precision"], row["sc_mean_recall"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=c,
            )
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

    # Shared axis limits
    num_cols = [c for c in df.columns if c not in ("mut_type", "noise", "n_samples")]
    all_p = pd.concat([df["sc_mean_precision"], df["spa_mean_precision"].dropna()])
    all_r = pd.concat([df["sc_mean_recall"],    df["spa_mean_recall"].dropna()])
    pad = 0.03
    lo = min(all_p.min(), all_r.min()) - pad
    hi = max(all_p.max(), all_r.max()) + pad
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    # Diagonal reference line
    for ax in axes:
        lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
                min(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color="grey", linewidth=0.7, linestyle=":", alpha=0.6)

    fig.supxlabel("Precision (mean)", fontsize=11)
    fig.supylabel("Sensitivity (mean recall)", fontsize=11)
    fig.suptitle("sigconfide vs SPA — Diaz-Gay 2023 benchmark", fontsize=12,
                 fontweight="bold", y=1.02)

    mut_handles = [
        mpatches.Patch(color=MUT_COLORS[m], label=m)
        for m in MUT_ORDER if m in df["mut_type"].values
    ]
    method_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=9, markerfacecolor="gray",
               markeredgecolor="white", markeredgewidth=1, label="sigconfide"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=10, markerfacecolor="gray",
               markeredgecolor="white", markeredgewidth=1, label="SPA"),
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
    p.add_argument("--input",  type=Path, default=Path("compare_spa_summary.csv"))
    p.add_argument("--output", type=Path, default=Path("compare_spa_summary_pr.png"))
    p.add_argument("--dpi",    type=int,  default=150)
    args = p.parse_args()
    plot_summary(args.input, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
