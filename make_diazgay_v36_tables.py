"""
Comparison tables for the COSMIC v3.6 set: sigconfide against SigProfilerAssignment.

Reads the harness output in PCAWG_Benchmark/diazgay-v36/eval/ (written by
benchmark_isbs.py, one --append run per arm) and writes, next to the set:

    diazgay-v36/comparison.csv    one row per arm x ground truth x noise level:
                                  micro P / R / F1, mean F1, false positives split
                                  into distractors and true-but-absent signatures,
                                  false negatives, round time
    diazgay-v36/comparison.md     the same as Markdown tables, plus micro F1 by
                                  burden and per-signature recall on the hardest
                                  round (full ground truth, 10% noise)
    diazgay_v36_micro_f1.csv      the F1 table alone, the format README.md quotes

Arms are found by their --label: sc (sigconfide defaults), sc_mandatory
(sigconfide with --mandatory SBS1 SBS5) and spa.  Missing arms are skipped.

Usage
    python make_diazgay_v36_tables.py
    python make_diazgay_v36_tables.py --set-dir PCAWG_Benchmark/diazgay-v36-seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ARMS = {
    "sc": "sigconfide, default",
    "sc_mandatory": "sigconfide, SBS1/SBS5 forced",
    "spa": "SPA",
}
GT_ORDER = ["restricted", "full"]
LEVELS = ["clean", "noise5", "noise10"]
BURDEN_ORDER = ["<1k", "1k-5k", "5k-20k", ">20k"]


def load(set_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    ev = set_dir / "eval"
    summary = pd.read_csv(ev / "summary.csv")
    per_sig = pd.read_csv(ev / "per_signature.csv")
    burden = pd.read_csv(ev / "by_burden.csv")
    manifest = json.loads((set_dir / "generation_manifest.json").read_text())
    true = set(manifest["signatures"])
    present = [a for a in ARMS if a in set(summary.label)]
    missing = [a for a in ARMS if a not in present]
    if missing:
        print(f"arms not in eval/: {missing}")
    return summary, per_sig, burden, true


def comparison_rows(summary, per_sig, true) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for gt in GT_ORDER:
            for lv in LEVELS:
                s = summary[(summary.label == arm) & (summary.ground_truth == gt)
                            & (summary.noise == lv)]
                if s.empty:
                    continue
                s = s.iloc[0]
                q = per_sig[(per_sig.label == arm) & (per_sig.ground_truth == gt)
                            & (per_sig.noise == lv)]
                rows.append({
                    "arm": arm,
                    "arm_name": ARMS[arm],
                    "ground_truth": gt,
                    "noise": lv,
                    "n_samples": int(s.n_samples),
                    "n_dictionary": int(s.n_dictionary),
                    "micro_precision": round(float(s.micro_precision), 4),
                    "micro_recall": round(float(s.micro_recall), 4),
                    "micro_f1": round(float(s.micro_f1), 4),
                    "mean_f1": round(float(s.mean_f1), 4),
                    "mean_recall_weighted": round(float(s.mean_recall_weighted), 4),
                    "fp_total": int(q.fp.sum()),
                    "fp_distractor": int(q[~q.signature.isin(true)].fp.sum()),
                    "fp_true_absent": int(q[q.signature.isin(true)].fp.sum()),
                    "fn_total": int(q.fn.sum()),
                    "elapsed_s": round(float(s.elapsed_s), 1),
                })
    return pd.DataFrame(rows)


def md_prf(c: pd.DataFrame) -> str:
    arms = [a for a in ARMS if a in set(c.arm)]
    out = ["| Ground truth | Level | " + " | ".join(ARMS[a] for a in arms) + " |",
           "| --- | --- | " + " | ".join("---" for _ in arms) + " |"]
    for gt in GT_ORDER:
        for i, lv in enumerate(LEVELS):
            cells = []
            for a in arms:
                r = c[(c.arm == a) & (c.ground_truth == gt) & (c.noise == lv)]
                cells.append("-" if r.empty else
                             f"{r.micro_precision.iloc[0]:.3f} / {r.micro_recall.iloc[0]:.3f}"
                             f" / {r.micro_f1.iloc[0]:.3f}")
            label = gt if i == 0 else ""
            out.append(f"| {label} | {lv} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_fp(c: pd.DataFrame, gt: str) -> str:
    arms = [a for a in ARMS if a in set(c.arm)]
    out = ["| Level | " + " | ".join(ARMS[a] for a in arms) + " |",
           "| --- | " + " | ".join("---" for _ in arms) + " |"]
    for lv in LEVELS:
        cells = []
        for a in arms:
            r = c[(c.arm == a) & (c.ground_truth == gt) & (c.noise == lv)]
            cells.append("-" if r.empty else
                         f"{r.fp_total.iloc[0]} ({r.fp_distractor.iloc[0]} distractors)")
        out.append(f"| {lv} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_burden(burden: pd.DataFrame, gt: str, lv: str) -> str:
    b = burden[(burden.ground_truth == gt) & (burden.noise == lv)]
    arms = [a for a in ARMS if a in set(b.label)]
    out = ["| Burden | n | " + " | ".join(ARMS[a] for a in arms) + " |",
           "| --- | --- | " + " | ".join("---" for _ in arms) + " |"]
    for bin_ in BURDEN_ORDER:
        rows = b[b.burden_bin == bin_]
        if rows.empty:
            continue
        n = int(rows.n_samples.iloc[0])
        cells = [f"{rows[rows.label == a].micro_f1.iloc[0]:.3f}" for a in arms]
        out.append(f"| {bin_} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def md_per_signature(per_sig: pd.DataFrame, true: set[str], gt: str, lv: str) -> str:
    p = per_sig[(per_sig.ground_truth == gt) & (per_sig.noise == lv)
                & (per_sig.signature.isin(true))]
    arms = [a for a in ARMS if a in set(p.label)]
    n_true = p.groupby("signature").n_true.first()
    sigs = sorted(n_true.index, key=lambda s: -n_true[s])
    out = ["| Signature | tumours | " + " | ".join(f"{ARMS[a]} recall / FP" for a in arms) + " |",
           "| --- | --- | " + " | ".join("---" for _ in arms) + " |"]
    for s in sigs:
        cells = []
        for a in arms:
            r = p[(p.label == a) & (p.signature == s)].iloc[0]
            cells.append(f"{r.sensitivity:.2f} / {int(r.fp)}")
        out.append(f"| {s} | {int(n_true[s])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set-dir", type=Path, default=Path("PCAWG_Benchmark/diazgay-v36"))
    args = ap.parse_args()
    summary, per_sig, burden, true = load(args.set_dir)
    c = comparison_rows(summary, per_sig, true)
    c.to_csv(args.set_dir / "comparison.csv", index=False)

    f1 = c.pivot(index=["ground_truth", "noise"], columns="arm", values="micro_f1")
    f1 = f1.reindex(pd.MultiIndex.from_product([GT_ORDER, LEVELS]))
    f1 = f1[[a for a in ARMS if a in f1.columns]].reset_index()
    f1.columns = ["ground_truth", "noise"] + list(f1.columns[2:])
    f1.to_csv(args.set_dir.parent / f"{args.set_dir.name.replace('-', '_')}_micro_f1.csv",
              index=False)

    n = int(c.n_samples.iloc[0])
    md = [
        f"# {args.set_dir.name}: sigconfide against SigProfilerAssignment",
        "",
        f"{n} tumours, generated by make_diazgay_v36.py and scored by benchmark_isbs.py;"
        " regenerate with run_diazgay_v36.sh.  Micro-averaged precision / recall / F1:",
        "",
        md_prf(c),
        "",
        "False positive calls over all tumours, full ground truth (in brackets: how"
        " many landed on the distractor signatures, the rest on true signatures"
        " called in a tumour that lacks them):",
        "",
        md_fp(c, "full"),
        "",
        "Micro F1 by clean burden, full ground truth, 10% noise:",
        "",
        md_burden(burden, "full", "noise10"),
        "",
        "Per true signature, full ground truth, 10% noise (recall / false positive"
        " calls):",
        "",
        md_per_signature(per_sig, true, "full", "noise10"),
        "",
    ]
    (args.set_dir / "comparison.md").write_text("\n".join(md))
    print("\n".join(md[:14]))
    print(f"\nwrote {args.set_dir / 'comparison.csv'}, {args.set_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
