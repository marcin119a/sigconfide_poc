"""
Benchmark sigconfide on the ISBS in-silico breast set.

The set is produced by make_breast_synthetic.py: 560 SBS-96 catalogues built as
round(W @ H) from Nik-Zainal 2016 Supplementary Table 21, with the proportional
Gaussian perturbation of the Diaz-Gay et al. 2023 benchmark applied on top
(noise5 / noise10 share one draw of standard normals).  The layout mirrors the
Diaz-Gay files, so the same evaluation applies, but two things differ and are
the reason this script exists rather than another MUT_TYPES entry in
benchmark_diaz_gay.py:

  * Ground truth is available in two variants.  The restricted file lists the 12
    signatures Table 21 reports in breast, so the fitting dictionary contains no
    distractors; the full file pads the same activities to all 30 COSMIC v2
    signatures with zero rows, so 18 columns can only ever produce false
    positives.  The gap between the two is the quantity of interest.
  * Ground truth carries absolute activities, not just presence.  Burdens span
    three orders of magnitude across the 560 tumours, so a missed signature at
    0.2% of the profile and a missed signature at 40% are both one false
    negative under set metrics.  This script therefore also reports
    exposure-weighted recall, the exposure mass placed on signatures that are
    absent, and cosine between the true and fitted exposure vectors.

Metrics per sample
    detection   tp / fp / fn, precision, recall, F1, MCC over the dictionary
    weighted    recall weighted by true exposure; predicted mass on false
                positives; L1/2 distance and cosine between exposure vectors
    fit         cosine between the observed profile and its reconstruction

Arms
    --method sigconfide   hybrid_stepwise_selection (default)
    --method spa          SigProfilerAssignment.Analyzer.cosmic_fit, same
                          dictionary and same scoring; needs the SPA venv
    --label NAME          names the arm in the output; with --append several
                          runs accumulate in one set of CSVs and stay comparable

Outputs (--out-dir, default <benchmark-dir>/eval)
    per_sample.csv, per_signature.csv, by_burden.csv, summary.csv, run_info.json

Usage
    python benchmark_isbs.py
    python benchmark_isbs.py --ground-truth restricted full
    python benchmark_isbs.py --noise-levels clean --max-samples 20
    python benchmark_isbs.py --min-fit-improvement 0.002 --mandatory SBS1 SBS5
    .venv-spa/bin/python benchmark_isbs.py --method spa --label spa --append
    python benchmark_isbs.py --benchmark-dir ICGC-BRCA/synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sigconfide.estimates.selection import hybrid_stepwise_selection

# ── Benchmark layout ──────────────────────────────────────────────────────────
DEFAULT_DIR = Path("ICGC-BRCA_Benchmark/synthetic-seed1")

SAMPLE_FILES = {
    "clean": "Samples.txt",
    "noise5": "Samples_noise5.txt",
    "noise10": "Samples_noise10.txt",
}
GT_FILES = {
    "restricted": "ground.truth.syn.exposures.csv",
    "full": "ground.truth.syn.exposures.full.csv",
}
METADATA_FILE = "sample_metadata.csv"
MANIFEST_FILE = "generation_manifest.json"

# Burden strata.  The 560 breast genomes run from a few hundred to >100k
# mutations, and selection behaves differently at the two ends.
BURDEN_EDGES = [1_000, 5_000, 20_000]
BURDEN_LABELS = ["<1k", "1k-5k", "5k-20k", ">20k"]

KEY_COLS = ["label", "ground_truth", "noise", "sample"]

# ── sigconfide defaults (as in benchmark_diaz_gay.py) ─────────────────────────
R = 100
PRE_FILTER = 0.001


def burden_bin(n: float) -> str:
    return BURDEN_LABELS[int(np.searchsorted(BURDEN_EDGES, n, side="right"))]


def find_panel(base: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    hits = sorted(base.glob("COSMIC_*.txt"))
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one COSMIC_*.txt in {base}, found {len(hits)};"
            " pass --panel"
        )
    return hits[0]


def norm_name(s: str) -> str:
    return s.replace("-", ".").replace(":", ".")


# ── Scoring (shared by every arm) ─────────────────────────────────────────────
def score_sample(sample, m, true_exp, pred_exp, P, sig_names, seconds):
    """Score one fitted exposure vector against the ground-truth activities.

    pred_exp and true_exp are dense vectors over the fitting dictionary;
    pred_exp is expected to sum to 1 (or 0 for an empty fit).
    """
    n_sigs = len(sig_names)
    total = float(m.sum())

    true_mask = true_exp > 0
    pred_mask = pred_exp > 0

    tp = int(np.sum(true_mask & pred_mask))
    fp = int(np.sum(~true_mask & pred_mask))
    fn = int(np.sum(true_mask & ~pred_mask))
    tn = n_sigs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    # Exposure-aware view: a missed signature counts for what it actually
    # contributed, and a false positive for the mass it stole.
    true_frac = true_exp / true_exp.sum() if true_exp.sum() > 0 else true_exp
    recall_w = float(true_frac[true_mask & pred_mask].sum())
    fp_mass = float(pred_exp[~true_mask].sum())
    l1 = float(np.abs(pred_exp - true_frac).sum() / 2.0)

    nt, npd = np.linalg.norm(true_frac), np.linalg.norm(pred_exp)
    cos_exp = float(true_frac @ pred_exp / (nt * npd)) if nt > 0 and npd > 0 else 0.0

    if total > 0:
        m_norm = m / total
        recon = P @ pred_exp
        nr = np.linalg.norm(recon)
        cos_fit = float(m_norm @ recon / (np.linalg.norm(m_norm) * nr)) if nr else 0.0
    else:
        cos_fit = 0.0

    return {
        "sample": sample,
        "burden": total,
        "burden_bin": burden_bin(total),
        "n_true": int(true_mask.sum()),
        "n_pred": int(pred_mask.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "recall_weighted": recall_w,
        "fp_exposure": fp_mass,
        "exposure_l1": l1,
        "exposure_cosine": cos_exp,
        "fit_cosine": cos_fit,
        "seconds": round(seconds, 3),
        "true_sigs": ",".join(sig_names[true_mask].tolist()),
        "pred_sigs": ",".join(sig_names[pred_mask].tolist()),
        "missed_sigs": ",".join(sig_names[true_mask & ~pred_mask].tolist()),
        "spurious_sigs": ",".join(sig_names[~true_mask & pred_mask].tolist()),
    }


# ── sigconfide worker (separate process) ──────────────────────────────────────
def run_sample(args):
    (sample, m, true_exp, P, sig_names, r, pre_filter, mandatory_idx,
     min_fit_gain, overdispersion) = args

    t0 = time.time()
    if m.sum() <= 0:
        sel_idx = np.array([], dtype=int)
        exposures = np.array([])
    else:
        sel_idx, exposures, _ = hybrid_stepwise_selection(
            m,
            P,
            R=r,
            pre_filter_threshold=pre_filter,
            mandatory_indices=list(mandatory_idx) or None,
            min_fit_improvement=min_fit_gain,
            overdispersion=overdispersion,
        )

    pred_exp = np.zeros(P.shape[1])
    pred_exp[sel_idx] = exposures
    return score_sample(
        sample, m, true_exp, pred_exp, P, sig_names, time.time() - t0
    )


# ── Round setup: dictionary, catalogues, ground truth ─────────────────────────
def load_round(base: Path, panel_path: Path, gt_mode: str, noise: str, cfg):
    panel = pd.read_csv(panel_path, sep="\t", index_col=0)
    gt_raw = pd.read_csv(base / GT_FILES[gt_mode], index_col=0)
    samples = pd.read_csv(base / SAMPLE_FILES[noise], sep="\t", index_col=0)

    common_idx = panel.index.intersection(samples.index)
    if len(common_idx) != len(panel.index):
        raise SystemExit(
            f"{len(panel.index) - len(common_idx)} panel channels are missing from"
            f" {SAMPLE_FILES[noise]}"
        )
    panel = panel.loc[common_idx]
    samples = samples.loc[common_idx]

    # The ground-truth file defines the dictionary: restricted gives the 12
    # signatures actually used, full adds the 18 zero rows as distractors.
    gt_sigs = [s for s in gt_raw.index if s in panel.columns]
    unknown = [s for s in gt_raw.index if s not in panel.columns]
    if unknown:
        raise SystemExit(f"ground truth names signatures absent from panel: {unknown}")

    sample_ids = samples.columns.tolist()
    if cfg["max_samples"] is not None:
        sample_ids = sample_ids[: cfg["max_samples"]]

    gt_col_map = {norm_name(c): c for c in gt_raw.columns}
    true_exp = {}
    for s in sample_ids:
        col = s if s in gt_raw.columns else gt_col_map.get(norm_name(s))
        if col is None:
            raise SystemExit(f"sample {s} has no ground-truth column")
        true_exp[s] = gt_raw.loc[gt_sigs, col].values.astype(float)

    return panel[gt_sigs], samples, gt_sigs, sample_ids, true_exp


# ── Arm: sigconfide ───────────────────────────────────────────────────────────
def run_sigconfide(base, panel_path, gt_mode, noise, cfg):
    panel_sub, samples, gt_sigs, sample_ids, true_exp = load_round(
        base, panel_path, gt_mode, noise, cfg
    )
    P = panel_sub.values.astype(float)
    sig_names = np.array(gt_sigs)
    mandatory_idx = [gt_sigs.index(s) for s in cfg["mandatory"] if s in gt_sigs]

    tasks = [
        (
            s,
            samples[s].values.astype(float),
            true_exp[s],
            P,
            sig_names,
            cfg["R"],
            cfg["pre_filter"],
            mandatory_idx,
            cfg["min_fit_gain"],
            cfg["overdispersion"],
        )
        for s in sample_ids
    ]

    t0 = time.time()
    results = []
    total = len(tasks)
    step = max(1, total // 10)
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futs = [pool.submit(run_sample, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            if len(results) % step == 0 or len(results) == total:
                print(
                    f"    [{cfg['label']}/{gt_mode}/{noise}] {len(results)}/{total}"
                    f"  ({time.time() - t0:.1f}s)"
                )
    return results, sig_names, time.time() - t0


# ── Arm: SigProfilerAssignment ────────────────────────────────────────────────
def run_spa(base, panel_path, gt_mode, noise, cfg):
    """cosmic_fit over the whole matrix at once, scored exactly like sigconfide.

    SPA is a batch fitter, so the per-sample `seconds` column carries the round
    time divided by the sample count rather than a real per-sample measurement.
    """
    from SigProfilerAssignment import Analyzer as SPA

    panel_sub, samples, gt_sigs, sample_ids, true_exp = load_round(
        base, panel_path, gt_mode, noise, cfg
    )
    P = panel_sub.values.astype(float)
    sig_names = np.array(gt_sigs)

    with tempfile.TemporaryDirectory() as tmp:
        samp_path = os.path.join(tmp, "samples.txt")
        sig_path = os.path.join(tmp, "signatures.txt")
        out_path = os.path.join(tmp, "spa_out")
        samples[sample_ids].to_csv(samp_path, sep="\t")
        panel_sub.to_csv(sig_path, sep="\t")

        t0 = time.time()
        SPA.cosmic_fit(
            samples=samp_path,
            output=out_path,
            signature_database=sig_path,
            collapse_to_SBS96=False,
            make_plots=False,  # activity PDFs take longer than the fit on 3600 samples
        )
        elapsed = time.time() - t0

        act_path = os.path.join(
            out_path,
            "Assignment_Solution",
            "Activities",
            "Assignment_Solution_Activities.txt",
        )
        if not os.path.exists(act_path):
            raise SystemExit(f"SPA produced no activities file at {act_path}")
        act = pd.read_csv(act_path, sep="\t", index_col=0)

    # SPA rewrites sample names ('::' becomes '..'), so map back by normal form.
    act_map = {norm_name(str(i)): i for i in act.index}
    missing = [s for s in sample_ids if norm_name(s) not in act_map]
    if missing:
        raise SystemExit(f"{len(missing)} samples missing from SPA output, e.g. "
                         f"{missing[0]}")

    results = []
    per_sample_s = elapsed / max(1, len(sample_ids))
    for s in sample_ids:
        row = act.loc[act_map[norm_name(s)]]
        pred = np.array([float(row.get(sig, 0.0)) for sig in gt_sigs])
        if pred.sum() > 0:
            pred = pred / pred.sum()
        results.append(
            score_sample(
                s,
                samples[s].values.astype(float),
                true_exp[s],
                pred,
                P,
                sig_names,
                per_sample_s,
            )
        )
    print(f"    [{cfg['label']}/{gt_mode}/{noise}] {len(results)}/{len(sample_ids)}"
          f"  ({elapsed:.1f}s)")
    return results, sig_names, elapsed


ARMS = {"sigconfide": run_sigconfide, "spa": run_spa}


def run_round(base, panel_path, gt_mode, noise, cfg):
    results, sig_names, elapsed = ARMS[cfg["method"]](
        base, panel_path, gt_mode, noise, cfg
    )
    df = pd.DataFrame(results).sort_values("sample").reset_index(drop=True)
    df.insert(0, "noise", noise)
    df.insert(0, "ground_truth", gt_mode)
    df.insert(0, "method", cfg["method"])
    df.insert(0, "label", cfg["label"])
    return df, sig_names, elapsed


# ── Aggregation ───────────────────────────────────────────────────────────────
def micro(df: pd.DataFrame) -> dict:
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "micro_precision": p,
        "micro_recall": r,
        "micro_f1": 2 * p * r / (p + r) if (p + r) else 0.0,
    }


def tags(df: pd.DataFrame) -> dict:
    return {
        "label": df["label"].iloc[0],
        "method": df["method"].iloc[0],
        "ground_truth": df["ground_truth"].iloc[0],
        "noise": df["noise"].iloc[0],
    }


def summarise(df: pd.DataFrame, n_sigs: int, elapsed: float) -> dict:
    row = tags(df)
    row.update(
        {
            "n_samples": len(df),
            "n_dictionary": n_sigs,
            "mean_precision": df["precision"].mean(),
            "mean_recall": df["recall"].mean(),
            "mean_f1": df["f1"].mean(),
            "mean_mcc": df["mcc"].mean(),
            "mean_recall_weighted": df["recall_weighted"].mean(),
            "mean_fp_exposure": df["fp_exposure"].mean(),
            "mean_exposure_l1": df["exposure_l1"].mean(),
            "mean_exposure_cosine": df["exposure_cosine"].mean(),
            "mean_fit_cosine": df["fit_cosine"].mean(),
            "median_seconds": df["seconds"].median(),
            "elapsed_s": round(elapsed, 1),
        }
    )
    row.update(micro(df))
    return row


def per_signature(df: pd.DataFrame, sig_names: np.ndarray) -> pd.DataFrame:
    rows = []
    for sig in sig_names:
        in_true = df["true_sigs"].str.split(",").apply(lambda v: sig in v)
        in_pred = df["pred_sigs"].str.split(",").apply(lambda v: sig in v)
        tp = int((in_true & in_pred).sum())
        fp = int((~in_true & in_pred).sum())
        fn = int((in_true & ~in_pred).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        if tp + fp + fn == 0:
            f1 = np.nan  # signature neither present nor ever called
        elif sens and prec:
            f1 = 2 * sens * prec / (sens + prec)
        else:
            f1 = 0.0
        row = tags(df)
        row.update(
            {
                "signature": sig,
                "n_true": int(in_true.sum()),
                "n_pred": int(in_pred.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sensitivity": sens,
                "precision": prec,
                "f1": f1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def by_burden(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in BURDEN_LABELS:
        sub = df[df["burden_bin"] == label]
        if sub.empty:
            continue
        row = tags(df)
        row.update(
            {
                "burden_bin": label,
                "n_samples": len(sub),
                "median_burden": sub["burden"].median(),
                "mean_precision": sub["precision"].mean(),
                "mean_recall": sub["recall"].mean(),
                "mean_f1": sub["f1"].mean(),
                "mean_mcc": sub["mcc"].mean(),
                "mean_recall_weighted": sub["recall_weighted"].mean(),
                "mean_fp_exposure": sub["fp_exposure"].mean(),
            }
        )
        row.update(micro(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries)
    cols = [
        "label",
        "ground_truth",
        "noise",
        "n_samples",
        "n_dictionary",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_mcc",
        "mean_recall_weighted",
        "mean_fp_exposure",
        "mean_exposure_cosine",
    ]
    out = df[cols].copy()
    for c in cols[5:]:
        out[c] = out[c].map("{:.3f}".format)
    print("\n-- Summary " + "-" * 60)
    print(out.to_string(index=False))


def write_csv(df: pd.DataFrame, path: Path, append: bool, keys: list[str]) -> None:
    """Write, or merge into an existing file, dropping superseded rows.

    A rerun of the same arm replaces its own rows rather than duplicating them,
    so several --append runs build one comparable table.
    """
    if append and path.exists():
        old = pd.read_csv(path)
        keys = [k for k in keys if k in old.columns and k in df.columns]
        if keys:
            incoming = df[keys].drop_duplicates()
            merged = old.merge(incoming, on=keys, how="left", indicator=True)
            old = old[merged["_merge"].values == "left_only"]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--benchmark-dir", default=str(DEFAULT_DIR))
    p.add_argument("--panel", default=None, help="signature dictionary (TSV)")
    p.add_argument(
        "--ground-truth",
        nargs="+",
        default=["restricted"],
        choices=list(GT_FILES),
        help="restricted = 12 breast signatures; full = +18 distractors",
    )
    p.add_argument(
        "--noise-levels",
        nargs="+",
        default=list(SAMPLE_FILES),
        choices=list(SAMPLE_FILES),
    )
    p.add_argument("--method", default="sigconfide", choices=list(ARMS))
    p.add_argument("--label", default=None, help="arm name in the output (def: method)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--bootstraps", type=int, default=R, help=f"replicates (def {R})")
    p.add_argument("--pre-filter", type=float, default=PRE_FILTER)
    p.add_argument(
        "--mandatory",
        nargs="*",
        default=[],
        help="signatures forced into every fit, e.g. SBS1 SBS5",
    )
    p.add_argument(
        "--min-fit-improvement",
        type=float,
        default=None,
        help="backward elimination on reconstruction cosine; 0.002 is validated",
    )
    p.add_argument(
        "--overdispersion",
        type=float,
        default=None,
        help="per-channel CV of a gamma multiplier in the bootstrap; 0.1 matches"
        " the benchmark's noise rule",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="default <benchmark-dir>/eval")
    p.add_argument(
        "--append",
        action="store_true",
        help="merge into existing CSVs instead of overwriting (rows keyed by label)",
    )
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    base = Path(args.benchmark_dir)
    if not base.is_dir():
        raise SystemExit(f"benchmark directory not found: {base}")
    panel_path = find_panel(base, args.panel)
    for gt in args.ground_truth:
        if not (base / GT_FILES[gt]).exists():
            raise SystemExit(f"missing ground truth file: {base / GT_FILES[gt]}")
    for noise in args.noise_levels:
        if not (base / SAMPLE_FILES[noise]).exists():
            raise SystemExit(f"missing catalogue file: {base / SAMPLE_FILES[noise]}")

    out_dir = Path(args.out_dir) if args.out_dir else base / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "method": args.method,
        "label": args.label or args.method,
        "R": args.bootstraps,
        "pre_filter": args.pre_filter,
        "mandatory": args.mandatory,
        "min_fit_gain": args.min_fit_improvement,
        "overdispersion": args.overdispersion,
        "max_samples": args.max_samples,
        "workers": args.workers,
    }

    print(f"benchmark : {base}")
    print(f"dictionary: {panel_path.name}")
    print(f"arm       : {cfg['label']}  (method={cfg['method']})")
    if cfg["method"] == "sigconfide":
        print(f"settings  : R={cfg['R']} pre_filter={cfg['pre_filter']} "
              f"mandatory={cfg['mandatory'] or 'none'} "
              f"min_fit_improvement={cfg['min_fit_gain']} "
              f"overdispersion={cfg['overdispersion']}")

    per_sample, per_sig, burden, summaries = [], [], [], []
    t_global = time.time()
    for gt in args.ground_truth:
        for noise in args.noise_levels:
            print(f"\n-> {cfg['label']} / {gt} / {noise}")
            df, sig_names, elapsed = run_round(base, panel_path, gt, noise, cfg)
            per_sample.append(df)
            per_sig.append(per_signature(df, sig_names))
            burden.append(by_burden(df))
            s = summarise(df, len(sig_names), elapsed)
            summaries.append(s)
            print(
                f"   mean P={s['mean_precision']:.3f} R={s['mean_recall']:.3f}"
                f" F1={s['mean_f1']:.3f} MCC={s['mean_mcc']:.3f}"
                f" | weighted R={s['mean_recall_weighted']:.3f}"
                f" FP mass={s['mean_fp_exposure']:.3f}  ({s['elapsed_s']}s)"
            )

    print_table(summaries)
    total_elapsed = time.time() - t_global
    print(f"\nTotal time: {total_elapsed:.1f}s")

    sample_df = pd.concat(per_sample, ignore_index=True)
    meta_path = base / METADATA_FILE
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        keep = [c for c in meta.columns if c not in sample_df.columns or c == "sample"]
        sample_df = sample_df.merge(meta[keep], on="sample", how="left")

    ap = args.append
    write_csv(sample_df, out_dir / "per_sample.csv", ap, KEY_COLS)
    write_csv(
        pd.concat(per_sig, ignore_index=True),
        out_dir / "per_signature.csv",
        ap,
        ["label", "ground_truth", "noise", "signature"],
    )
    write_csv(
        pd.concat(burden, ignore_index=True),
        out_dir / "by_burden.csv",
        ap,
        ["label", "ground_truth", "noise", "burden_bin"],
    )
    write_csv(
        pd.DataFrame(summaries),
        out_dir / "summary.csv",
        ap,
        ["label", "ground_truth", "noise"],
    )

    manifest_path = base / MANIFEST_FILE
    run_info = {
        "benchmark_dir": str(base),
        "dictionary": str(panel_path),
        "method": cfg["method"],
        "label": cfg["label"],
        "ground_truth": args.ground_truth,
        "noise_levels": args.noise_levels,
        "max_samples": args.max_samples,
        "bootstraps": cfg["R"],
        "pre_filter_threshold": cfg["pre_filter"],
        "mandatory": cfg["mandatory"],
        "min_fit_improvement": cfg["min_fit_gain"],
        "overdispersion": cfg["overdispersion"],
        "elapsed_s": round(total_elapsed, 1),
        "generation_manifest": (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        ),
    }
    info_path = out_dir / "run_info.json"
    if ap and info_path.exists():
        prev = json.loads(info_path.read_text())
        runs = prev.get("runs", [prev]) if isinstance(prev, dict) else []
        runs = [r for r in runs if r.get("label") != cfg["label"]] + [run_info]
    else:
        runs = [run_info]
    info_path.write_text(json.dumps({"runs": runs}, indent=2))

    print(f"\nper-sample     -> {out_dir / 'per_sample.csv'}")
    print(f"per-signature  -> {out_dir / 'per_signature.csv'}")
    print(f"by burden      -> {out_dir / 'by_burden.csv'}")
    print(f"summary        -> {out_dir / 'summary.csv'}")
    print(f"run settings   -> {info_path}")


if __name__ == "__main__":
    main()
