"""
Compare sigconfide vs SigProfilerAssignment on the Diaz-Gay et al. 2023 benchmark.

For each (mut_type × noise_level) combination:
  1. sigconfide – hybrid_stepwise_selection, R bootstraps, parallel per sample
  2. SPA         – SigProfilerAssignment.Analyzer.cosmic_fit, same signature pool

Metrics: precision / recall / F1 — reported as mean over samples (macro) and
micro-averaged from pooled TP/FP/FN. Recall equals sensitivity here
(TP / (TP + FN) over selected signature sets). Summary CSV also includes runtimes.

Outputs → compare_spa_results.csv   (per-sample, both methods)
          compare_spa_summary.csv   (aggregates)
"""

import argparse
import os
import time
import tempfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

from sigconfide.estimates.selection import hybrid_stepwise_selection
from SigProfilerAssignment import Analyzer as SPA

# ── Configuration ──────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path("Supplementary_data_Diaz-Gay_et_al_2023_Benchmark")

MUT_TYPES = {
    "SBS": {
        "cosmic":  "SBS/COSMIC_v3.3_SBS_GRCh37.txt",
        "gt":      "SBS/ground.truth.syn.exposures.csv",
        "samples": {
            "clean":   "SBS/Samples.txt",
            "noise5":  "SBS/Samples_noise5.txt",
            "noise10": "SBS/Samples_noise10.txt",
        },
        "context_type": "96",
    },
    "DBS": {
        "cosmic":  "DBS/COSMIC_v3.3_DBS_GRCh37.txt",
        "gt":      "DBS/ground.truth.syn.exposures.DBS.csv",
        "samples": {
            "clean":   "DBS/Samples_DBS.txt",
            "noise5":  "DBS/Samples_DBS_noise5.txt",
            "noise10": "DBS/Samples_DBS_noise10.txt",
        },
        "context_type": "78",
    },
    "ID": {
        "cosmic":  "ID/COSMIC_v3.3_ID_GRCh37.txt",
        "gt":      "ID/ground.truth.syn.exposures.ID.csv",
        "samples": {
            "clean":   "ID/Samples_ID.txt",
            "noise5":  "ID/Samples_ID_noise5.txt",
            "noise10": "ID/Samples_ID_noise10.txt",
        },
        "context_type": "83",
    },
    "CN": {
        "cosmic":  "CN/COSMIC_v3.3_CN_GRCh37.txt",
        "gt":      "CN/ground.truth.syn.exposures.CN.csv",
        "samples": {
            "clean":   "CN/Samples_CN.txt",
            "noise5":  "CN/Samples_CN_noise5.txt",
            "noise10": "CN/Samples_CN_noise10.txt",
        },
        "context_type": "48",   # SPA may not support; handled below
    },
}

# sigconfide
R          = 100
PRE_FILTER = 0.001
MAX_WORKERS = None   # None = CPU count

# Permanently active signatures (never evicted by backward step) — analogous
# to SPA's permanent_sigs / background_sigs.  Only defined for mut types where
# biological evidence supports universal presence.
MANDATORY_SIGS = {
    "SBS": ["SBS1", "SBS5"],
    "DBS": [],
    "ID":  [],
    "CN":  [],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(s):
    return s.replace("-", ".").replace(":", ".")


def _metrics(true_sigs: set, pred_sigs: set) -> dict:
    tp = len(true_sigs & pred_sigs)
    fp = len(pred_sigs - true_sigs)
    fn = len(true_sigs - pred_sigs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)


def _aggregate(df: pd.DataFrame, prefix: str) -> dict:
    p_col, r_col, f_col = f"{prefix}_precision", f"{prefix}_recall", f"{prefix}_f1"
    tp = df[f"{prefix}_tp"].sum()
    fp = df[f"{prefix}_fp"].sum()
    fn = df[f"{prefix}_fn"].sum()
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) > 0 else 0.0)
    return {
        f"{prefix}_mean_precision": df[p_col].mean(),
        f"{prefix}_mean_recall":    df[r_col].mean(),
        f"{prefix}_mean_f1":        df[f_col].mean(),
        f"{prefix}_micro_precision": micro_p,
        f"{prefix}_micro_recall":    micro_r,
        f"{prefix}_micro_f1":        micro_f,
    }


# ── sigconfide worker (separate process) ────────────────────────────────────
def _sc_worker(args):
    sample, m, true_sigs_list, P, sig_names, R, pre_filter, mandatory_idx = args
    sel_idx, _, _ = hybrid_stepwise_selection(
        m, P, R=R, pre_filter_threshold=pre_filter,
        mandatory_indices=mandatory_idx if mandatory_idx else None,
    )
    pred = set(sig_names[sel_idx].tolist())
    true = set(true_sigs_list)
    return sample, true, pred


# ── sigconfide – full pass ──────────────────────────────────────────────────
def run_sigconfide(samples_df, gt_raw, cosmic_sub, sig_names, all_samples,
                   mut_type, noise_level):
    gt_col_map = {_norm(c): c for c in gt_raw.columns}

    mandatory_names = MANDATORY_SIGS.get(mut_type, [])
    sig_names_list  = sig_names.tolist()
    mandatory_idx   = [sig_names_list.index(s) for s in mandatory_names
                       if s in sig_names_list]

    tasks = []
    for sample in all_samples:
        m = samples_df[sample].values.astype(float)
        gt_col = sample if sample in gt_raw.columns \
                 else gt_col_map.get(_norm(sample))
        true_sigs = gt_raw.index[gt_raw[gt_col] > 0].tolist() if gt_col else []
        tasks.append((sample, m, true_sigs,
                      cosmic_sub.values.astype(float), sig_names, R, PRE_FILTER,
                      mandatory_idx))

    t0 = time.time()
    results = []
    total = len(tasks)
    print_every = max(1, total // 5)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_sc_worker, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            sample, true, pred = fut.result()
            m = _metrics(true, pred)
            results.append(dict(
                sample=sample,
                n_true=len(true), n_pred_sc=len(pred),
                sc_tp=m["tp"], sc_fp=m["fp"], sc_fn=m["fn"],
                sc_precision=m["precision"], sc_recall=m["recall"], sc_f1=m["f1"],
                true_sigs=",".join(sorted(true)),
                sc_pred_sigs=",".join(sorted(pred)),
            ))
            done += 1
            if done % print_every == 0 or done == total:
                print(f"    [sigconfide {mut_type}/{noise_level}] "
                      f"{done}/{total}  ({time.time()-t0:.1f}s)")

    return pd.DataFrame(results), round(time.time() - t0, 1)


# ── SigProfilerAssignment – full pass ───────────────────────────────────────
def run_spa(samples_df, gt_raw, cosmic_sub, all_samples,
            mut_type, noise_level, context_type, tmpdir):
    """
    Run SPA cosmic_fit; returns (per-sample metrics DataFrame, elapsed_s) or
    (None, None) if SPA errors or output is missing.
    """
    gt_col_map = {_norm(c): c for c in gt_raw.columns}

    samp_path = os.path.join(tmpdir, f"samples_{mut_type}_{noise_level}.txt")
    sig_path  = os.path.join(tmpdir, f"sigs_{mut_type}.txt")
    out_path  = os.path.join(tmpdir, f"spa_{mut_type}_{noise_level}")

    samples_df[all_samples].to_csv(samp_path, sep="\t")
    cosmic_sub.to_csv(sig_path, sep="\t")

    t0 = time.time()
    try:
        SPA.cosmic_fit(
            samples=samp_path,
            output=out_path,
            signature_database=sig_path,
            collapse_to_SBS96=False,
        )
    except Exception as e:
        print(f"    [SPA {mut_type}/{noise_level}] ERROR: {e}")
        return None, None

    act_path = os.path.join(
        out_path,
        "Assignment_Solution", "Activities",
        "Assignment_Solution_Activities.txt",
    )
    if not os.path.exists(act_path):
        print(f"    [SPA {mut_type}/{noise_level}] Activities file missing – skipping")
        return None, None

    act = pd.read_csv(act_path, sep="\t", index_col=0)
    elapsed = round(time.time() - t0, 1)

    # Activities column name → set of assigned signatures (per row / sample)
    result = {}
    for row_name in act.index:
        assigned = set(act.columns[act.loc[row_name] > 0].tolist())
        result[row_name] = assigned

    # Normalize sample names (SPA may turn '::' into '..')
    def _norm_spa(s):
        return s.replace("::", "..").replace("-", ".").replace(":", ".")

    norm_map = {_norm_spa(s): s for s in result}

    # Collect per-sample rows
    rows = []
    for sample in all_samples:
        # try to match row keys
        pred = (result.get(sample)
                or result.get(_norm_spa(sample))
                or norm_map.get(_norm_spa(sample)) and result[norm_map[_norm_spa(sample)]]
                or set())

        gt_col = sample if sample in gt_raw.columns \
                 else gt_col_map.get(_norm(sample))
        true = set(gt_raw.index[gt_raw[gt_col] > 0].tolist()) if gt_col else set()

        m = _metrics(true, pred)
        rows.append(dict(
            sample=sample,
            n_pred_spa=len(pred),
            spa_tp=m["tp"], spa_fp=m["fp"], spa_fn=m["fn"],
            spa_precision=m["precision"], spa_recall=m["recall"], spa_f1=m["f1"],
            spa_pred_sigs=",".join(sorted(pred)),
        ))

    return pd.DataFrame(rows), elapsed


# ── One round (mut_type × noise_level) ───────────────────────────────────────
def run_round(mut_type, noise_level, cfg, max_samples, tmpdir):
    base = BENCHMARK_DIR
    cosmic   = pd.read_csv(base / cfg["cosmic"], sep="\t", index_col=0)
    gt_raw   = pd.read_csv(base / cfg["gt"], index_col=0)
    samples  = pd.read_csv(base / cfg["samples"][noise_level], sep="\t", index_col=0)

    common_idx = cosmic.index.intersection(samples.index)
    cosmic  = cosmic.loc[common_idx]
    samples = samples.loc[common_idx]

    gt_sigs    = [s for s in gt_raw.index if s in cosmic.columns]
    cosmic_sub = cosmic[gt_sigs]
    sig_names  = np.array(gt_sigs)

    all_samples = samples.columns.tolist()
    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    # sigconfide
    print(f"  → sigconfide ...")
    sc_df, sc_time = run_sigconfide(
        samples, gt_raw, cosmic_sub, sig_names, all_samples,
        mut_type, noise_level,
    )
    sc_df = sc_df.set_index("sample")

    # SigProfilerAssignment
    print(f"  → SPA ...")
    spa_df, spa_time = run_spa(
        samples, gt_raw, cosmic_sub, all_samples,
        mut_type, noise_level, cfg["context_type"], tmpdir,
    )
    if spa_df is not None:
        spa_df = spa_df.set_index("sample")

    # Merge
    combined = sc_df.copy()
    if spa_df is not None:
        combined = combined.join(spa_df, how="left")
    else:
        for col in ["n_pred_spa", "spa_tp", "spa_fp", "spa_fn",
                    "spa_precision", "spa_recall", "spa_f1", "spa_pred_sigs"]:
            combined[col] = np.nan

    combined.insert(0, "noise",    noise_level)
    combined.insert(0, "mut_type", mut_type)
    combined = combined.reset_index().rename(columns={"index": "sample"})

    # Aggregates
    sc_agg  = _aggregate(combined.rename(columns=lambda c: c), "sc")
    spa_agg = _aggregate(combined.dropna(subset=["spa_f1"]), "spa") \
              if spa_df is not None else {}

    summary = {
        "mut_type": mut_type,
        "noise":    noise_level,
        "n_samples": len(all_samples),
        **sc_agg,
        **spa_agg,
        "sc_time_s":  sc_time,
        "spa_time_s": spa_time if spa_df is not None else None,
    }

    return combined, summary


# ── Print comparison tables ─────────────────────────────────────────────────
_RECALL_SENS_RENAME = {
    "sc_micro_recall":  "sc_micro_sens",
    "spa_micro_recall": "spa_micro_sens",
}


def _fmt_float_cols(df: pd.DataFrame, cols: list, ndigits: int = 4) -> pd.DataFrame:
    sub = df[[c for c in cols if c in df.columns]].copy()
    for c in sub.columns:
        if c in ("mut_type", "noise"):
            continue
        if c == "n_samples":
            sub[c] = pd.to_numeric(sub[c], errors="coerce").map(
                lambda x: str(int(x)) if pd.notna(x) else "N/A"
            )
            continue
        sub[c] = pd.to_numeric(sub[c], errors="coerce").map(
            lambda x, nd=ndigits: f"{x:.{nd}f}" if pd.notna(x) else "N/A"
        )
    return sub


def print_comparison(summaries):
    df = pd.DataFrame(summaries)
    base = ["mut_type", "noise", "n_samples"]

    print("\n── Mean over samples (average of per-sample P/R/F1) ───────────────────")
    cols = base + [
        "sc_mean_precision", "spa_mean_precision",
        "sc_mean_recall", "spa_mean_recall",
        "sc_mean_f1", "spa_mean_f1",
    ]
    print(_fmt_float_cols(df, cols, 4).to_string(index=False))

    print("\n── Micro-averaged (from pooled TP/FP/FN across samples) ────────────────")
    cols = base + [
        "sc_micro_precision", "spa_micro_precision",
        "sc_micro_recall", "spa_micro_recall",
        "sc_micro_f1", "spa_micro_f1",
    ]
    micro_tbl = _fmt_float_cols(df, cols, 4).rename(columns=_RECALL_SENS_RENAME)
    print(micro_tbl.to_string(index=False))

    print("\n── Runtime (s) ─────────────────────────────────────────────────────────")
    cols = base + ["sc_time_s", "spa_time_s"]
    tdf = df[[c for c in cols if c in df.columns]].copy()
    for c in ("sc_time_s", "spa_time_s"):
        if c in tdf.columns:
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce").map(
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
    print(tdf.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="sigconfide vs SigProfilerAssignment – Diaz-Gay 2023 benchmark"
    )
    p.add_argument("--mut-types", nargs="+", default=list(MUT_TYPES.keys()),
                   choices=list(MUT_TYPES.keys()),
                   help="Mutation types to run (default: all)")
    p.add_argument("--noise-levels", nargs="+",
                   default=["clean", "noise5", "noise10"],
                   choices=["clean", "noise5", "noise10"],
                   help="Noise levels (default: all)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of samples (for quick tests)")
    p.add_argument("--out-per-sample", default="compare_spa_results.csv",
                   help="CSV file for per-sample results")
    p.add_argument("--out-summary",    default="compare_spa_summary.csv",
                   help="CSV file for aggregate metrics")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    all_results   = []
    all_summaries = []
    t_global = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        for mut_type in args.mut_types:
            cfg = MUT_TYPES[mut_type]
            for noise_level in args.noise_levels:
                print(f"\n═══ {mut_type} / {noise_level} ═══")
                combined, summary = run_round(
                    mut_type, noise_level, cfg,
                    args.max_samples, tmpdir,
                )
                all_results.append(combined)
                all_summaries.append(summary)

                # Quick preview: P, recall (=sensitivity), F1 (mean + micro)
                sc_mp = summary.get("sc_mean_precision", float("nan"))
                sc_mr = summary.get("sc_mean_recall", float("nan"))
                sc_mf1 = summary.get("sc_mean_f1", float("nan"))
                sc_up = summary.get("sc_micro_precision", float("nan"))
                sc_ur = summary.get("sc_micro_recall", float("nan"))
                sc_uf1 = summary.get("sc_micro_f1", float("nan"))
                print(
                    f"  sigconfide  mean: P={sc_mp:.3f}  R/sens={sc_mr:.3f}  F1={sc_mf1:.3f}  |  "
                    f"micro: P={sc_up:.3f}  R/sens={sc_ur:.3f}  F1={sc_uf1:.3f}  "
                    f"({summary['sc_time_s']}s)"
                )
                sp_mp = summary.get("spa_mean_precision", float("nan"))
                sp_mr = summary.get("spa_mean_recall", float("nan"))
                sp_mf1 = summary.get("spa_mean_f1", float("nan"))
                sp_up = summary.get("spa_micro_precision", float("nan"))
                sp_ur = summary.get("spa_micro_recall", float("nan"))
                sp_uf1 = summary.get("spa_micro_f1", float("nan"))
                if pd.notna(sp_mf1):
                    print(
                        f"  SPA         mean: P={sp_mp:.3f}  R/sens={sp_mr:.3f}  F1={sp_mf1:.3f}  |  "
                        f"micro: P={sp_up:.3f}  R/sens={sp_ur:.3f}  F1={sp_uf1:.3f}  "
                        f"({summary['spa_time_s']}s)"
                    )
                else:
                    print("  SPA         (no result)")

    print_comparison(all_summaries)
    print(f"\nTotal time: {time.time()-t_global:.1f}s")

    pd.concat(all_results, ignore_index=True).to_csv(args.out_per_sample, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out_summary, index=False)
    print(f"\nPer-sample results → {args.out_per_sample}")
    print(f"Summary            → {args.out_summary}")
