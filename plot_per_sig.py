"""
Per-signature recall vs precision scatter: sigconfide vs SPA.
4 subplots (SBS, DBS, ID, CN), each showing both methods as connected pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("per_sig_stats.csv")

MUT_TYPES = ["SBS", "DBS", "ID", "CN"]
SC_COLOR  = "#2196F3"   # blue
SPA_COLOR = "#F44336"   # red
MIN_PRESENT = 10

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, mt in zip(axes, MUT_TYPES):
    sub = df[(df.mut_type == mt) & (df.n_present >= MIN_PRESENT)].dropna(
        subset=["sc_recall", "sc_precision", "spa_recall", "spa_precision"]
    ).copy()

    # draw connecting lines first (gray)
    for _, r in sub.iterrows():
        ax.plot(
            [r.sc_recall, r.spa_recall],
            [r.sc_precision, r.spa_precision],
            color="gray", alpha=0.35, linewidth=0.8, zorder=1,
        )

    # SC points
    ax.scatter(sub.sc_recall, sub.sc_precision,
               color=SC_COLOR, s=50, zorder=3, label="sigconfide", alpha=0.9)
    # SPA points
    ax.scatter(sub.spa_recall, sub.spa_precision,
               color=SPA_COLOR, s=50, zorder=3, marker="D", label="SPA", alpha=0.9)

    # annotate signatures with large recall diff or extreme precision
    for _, r in sub.iterrows():
        diff = r.sc_recall - r.spa_recall
        prec_bad = min(r.sc_precision, r.spa_precision) < 0.5

        if abs(diff) > 0.08 or prec_bad:
            # label between the two points
            mx = (r.sc_recall + r.spa_recall) / 2
            my = (r.sc_precision + r.spa_precision) / 2
            ax.annotate(
                r.signature,
                xy=(mx, my),
                fontsize=6.5,
                ha="center", va="bottom",
                color="black",
                alpha=0.85,
            )

    ax.set_title(mt, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.axhline(1.0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(1.0, color="gray", linewidth=0.4, linestyle="--")
    ax.grid(True, alpha=0.3)

    # text count
    ax.text(0.02, 0.02, f"n={len(sub)} sigs", transform=ax.transAxes,
            fontsize=8, color="gray")

# legend
sc_patch  = mpatches.Patch(color=SC_COLOR,  label="sigconfide")
spa_patch = mpatches.Patch(color=SPA_COLOR, label="SigProfilerAssignment")
fig.legend(handles=[sc_patch, spa_patch], loc="upper center",
           ncol=2, fontsize=11, frameon=True,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle("Per-signature Precision vs Recall\n(lines connect the same signature across methods)",
             fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig("per_sig_scatter.png", dpi=150, bbox_inches="tight")
print("Saved per_sig_scatter.png")


# ── second plot: recall diff bar chart ────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for ax, mt in zip(axes2, MUT_TYPES):
    sub = df[(df.mut_type == mt) & (df.n_present >= MIN_PRESENT)].dropna(
        subset=["recall_diff"]
    ).sort_values("recall_diff", ascending=True).copy()

    colors = [SC_COLOR if v > 0 else SPA_COLOR for v in sub.recall_diff]
    bars = ax.barh(sub.signature, sub.recall_diff, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{mt}  (SC recall − SPA recall)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Δ recall  (positive = SC better)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # annotate n_present
    for bar, (_, r) in zip(bars, sub.iterrows()):
        ax.text(
            bar.get_width() + (0.005 if bar.get_width() >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"n={int(r.n_present)}",
            va="center",
            ha="left" if bar.get_width() >= 0 else "right",
            fontsize=6,
            color="gray",
        )

sc_patch  = mpatches.Patch(color=SC_COLOR,  label="SC better")
spa_patch = mpatches.Patch(color=SPA_COLOR, label="SPA better")
fig2.legend(handles=[sc_patch, spa_patch], loc="upper center",
            ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.01))
fig2.suptitle("Recall gap per signature  (SC − SPA)", fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig("per_sig_recall_diff.png", dpi=150, bbox_inches="tight")
print("Saved per_sig_recall_diff.png")
