"""
Per-signature TP/FP/FN analysis: sigconfide vs SigProfilerAssignment.

For each signature, across all samples where it is present in ground truth,
compute:
  - recall  = TP / (TP + FN)   how often correctly detected
  - FDR     = FP / (TP + FP)   how often falsely called

Output:
  per_sig_stats.csv   — full table
  Printed ranked tables for the biggest SC vs SPA differences
"""

import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("compare_spa_results.csv")

# ── per-signature counters ──────────────────────────────────────────────────
def parse_sigs(s):
    if pd.isna(s) or s == "":
        return set()
    return set(s.split(","))


stats = defaultdict(lambda: dict(
    mut_type=None,
    sc_tp=0, sc_fp=0, sc_fn=0,
    spa_tp=0, spa_fp=0, spa_fn=0,
    n_present=0,   # times it appears in ground truth
    n_absent=0,    # times it does NOT appear in ground truth
))

for _, row in df.iterrows():
    true  = parse_sigs(row["true_sigs"])
    sc    = parse_sigs(row["sc_pred_sigs"])
    spa   = parse_sigs(row.get("spa_pred_sigs", ""))
    mt    = row["mut_type"]

    all_considered = true | sc | spa
    for sig in all_considered:
        s = stats[sig]
        s["mut_type"] = mt
        if sig in true:
            s["n_present"] += 1
            if sig in sc:   s["sc_tp"] += 1
            else:           s["sc_fn"] += 1
            if sig in spa:  s["spa_tp"] += 1
            else:           s["spa_fn"] += 1
        else:
            s["n_absent"] += 1
            if sig in sc:   s["sc_fp"] += 1
            if sig in spa:  s["spa_fp"] += 1

rows = []
for sig, s in stats.items():
    sc_prec  = s["sc_tp"]  / (s["sc_tp"]  + s["sc_fp"])  if (s["sc_tp"]  + s["sc_fp"])  > 0 else np.nan
    sc_rec   = s["sc_tp"]  / (s["sc_tp"]  + s["sc_fn"])  if (s["sc_tp"]  + s["sc_fn"])  > 0 else np.nan
    spa_prec = s["spa_tp"] / (s["spa_tp"] + s["spa_fp"]) if (s["spa_tp"] + s["spa_fp"]) > 0 else np.nan
    spa_rec  = s["spa_tp"] / (s["spa_tp"] + s["spa_fn"]) if (s["spa_tp"] + s["spa_fn"]) > 0 else np.nan

    rows.append(dict(
        signature=sig,
        mut_type=s["mut_type"],
        n_present=s["n_present"],
        n_absent=s["n_absent"],
        sc_tp=s["sc_tp"], sc_fp=s["sc_fp"], sc_fn=s["sc_fn"],
        spa_tp=s["spa_tp"], spa_fp=s["spa_fp"], spa_fn=s["spa_fn"],
        sc_precision=sc_prec,
        sc_recall=sc_rec,
        spa_precision=spa_prec,
        spa_recall=spa_rec,
        recall_diff=sc_rec - spa_rec if (not np.isnan(sc_rec) and not np.isnan(spa_rec)) else np.nan,
        prec_diff=sc_prec - spa_prec if (not np.isnan(sc_prec) and not np.isnan(spa_prec)) else np.nan,
    ))

out = pd.DataFrame(rows).sort_values(["mut_type", "signature"])
out.to_csv("per_sig_stats.csv", index=False)
print(f"Saved per_sig_stats.csv  ({len(out)} signatures)")

# ── Print tables ────────────────────────────────────────────────────────────
FMT = "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
HDR = FMT.format("Signature", "n_pres", "sc_rec", "spa_rec", "Δrec",
                  "sc_prec", "spa_prec", "sc_fp", "spa_fp")

for mt in ["SBS", "DBS", "ID", "CN"]:
    sub = out[(out.mut_type == mt) & (out.n_present >= 10)].copy()
    if sub.empty:
        continue

    print(f"\n{'═'*90}")
    print(f"  {mt}  — signatures present in ≥10 samples")
    print(f"{'═'*90}")
    print(HDR)
    print("-" * 90)

    sub_sorted = sub.sort_values("recall_diff", ascending=False)
    for _, r in sub_sorted.iterrows():
        flag = ""
        if not np.isnan(r.recall_diff):
            if r.recall_diff >  0.1: flag = "  ← SC better recall"
            if r.recall_diff < -0.1: flag = "  ← SPA better recall"
        print(FMT.format(
            r.signature,
            int(r.n_present),
            f"{r.sc_recall:.3f}"  if not np.isnan(r.sc_recall)  else "N/A",
            f"{r.spa_recall:.3f}" if not np.isnan(r.spa_recall) else "N/A",
            f"{r.recall_diff:+.3f}" if not np.isnan(r.recall_diff) else "N/A",
            f"{r.sc_precision:.3f}"  if not np.isnan(r.sc_precision)  else "N/A",
            f"{r.spa_precision:.3f}" if not np.isnan(r.spa_precision) else "N/A",
            int(r.sc_fp),
            int(r.spa_fp),
        ) + flag)

# ── Summary: biggest recall gaps ────────────────────────────────────────────
print(f"\n{'═'*90}")
print("  TOP 10 signatures where SC recall > SPA recall (Δ > 0)")
print(f"{'═'*90}")
top_sc = out.dropna(subset=["recall_diff"]).nlargest(10, "recall_diff")
print(HDR)
for _, r in top_sc.iterrows():
    print(FMT.format(
        r.signature, int(r.n_present),
        f"{r.sc_recall:.3f}", f"{r.spa_recall:.3f}", f"{r.recall_diff:+.3f}",
        f"{r.sc_precision:.3f}" if not np.isnan(r.sc_precision) else "N/A",
        f"{r.spa_precision:.3f}" if not np.isnan(r.spa_precision) else "N/A",
        int(r.sc_fp), int(r.spa_fp),
    ))

print(f"\n{'═'*90}")
print("  TOP 10 signatures where SPA recall > SC recall (Δ < 0)")
print(f"{'═'*90}")
top_spa = out.dropna(subset=["recall_diff"]).nsmallest(10, "recall_diff")
print(HDR)
for _, r in top_spa.iterrows():
    print(FMT.format(
        r.signature, int(r.n_present),
        f"{r.sc_recall:.3f}", f"{r.spa_recall:.3f}", f"{r.recall_diff:+.3f}",
        f"{r.sc_precision:.3f}" if not np.isnan(r.sc_precision) else "N/A",
        f"{r.spa_precision:.3f}" if not np.isnan(r.spa_precision) else "N/A",
        int(r.sc_fp), int(r.spa_fp),
    ))
