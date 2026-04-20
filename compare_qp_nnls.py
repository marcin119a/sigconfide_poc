"""
Compare QP vs NNLS decomposition inside sigconfide's hybrid_stepwise_selection.

For each (mut_type × noise_level) combination both methods use identical settings:
  - same signature pool, same pre-filter threshold, same mandatory signatures
  - same R bootstraps, same bootstrap strategy, same parallel workers

Metrics: precision / recall / F1 — mean over samples (macro) and micro-averaged
from pooled TP/FP/FN.  Summary CSV includes runtimes.

Outputs → compare_qp_nnls_results.csv   (per-sample, both methods)
          compare_qp_nnls_summary.csv    (aggregates)
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from sigconfide.estimates.selection import hybrid_stepwise_selection
from sigconfide.decompose.qp import decomposeQP
from sigconfide.decompose.nnls import decomposeNNLS

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
    },
    "DBS": {
        "cosmic":  "DBS/COSMIC_v3.3_DBS_GRCh37.txt",
        "gt":      "DBS/ground.truth.syn.exposures.DBS.csv",
        "samples": {
            "clean":   "DBS/Samples_DBS.txt",
            "noise5":  "DBS/Samples_DBS_noise5.txt",
            "noise10": "DBS/Samples_DBS_noise10.txt",
        },
    },
    "ID": {
        "cosmic":  "ID/COSMIC_v3.3_ID_GRCh37.txt",
        "gt":      "ID/ground.truth.syn.exposures.ID.csv",
        "samples": {
            "clean":   "ID/Samples_ID.txt",
            "noise5":  "ID/Samples_ID_noise5.txt",
            "noise10": "ID/Samples_ID_noise10.txt",
        },
    },
    "CN": {
        "cosmic":  "CN/COSMIC_v3.3_CN_GRCh37.txt",
        "gt":      "CN/ground.truth.syn.exposures.CN.csv",
        "samples": {
            "clean":   "CN/Samples_CN.txt",
            "noise5":  "CN/Samples_CN_noise5.txt",
            "noise10": "CN/Samples_CN_noise10.txt",
        },
    },
}

R           = 100
PRE_FILTER  = 0.001
MAX_WORKERS = None   # None = CPU count

MANDATORY_SIGS = {
    "SBS": ["SBS1", "SBS5"],
    "DBS": [],
    "ID":  [],
    "CN":  [],
}

METHODS = {
    "qp":   decomposeQP,
    "nnls": decomposeNNLS,
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


# ── Worker (runs in separate process) ─────────────────────────────────────────
def _worker(args):
    sample, m, true_sigs_list, P, sig_names, R, pre_filter, mandatory_idx, bootstrap_method, method_name = args
    decompose_fn = METHODS[method_name]
    sel_idx, _, _ = hybrid_stepwise_selection(
        m, P, R=R, pre_filter_threshold=pre_filter,
        mandatory_indices=mandatory_idx if mandatory_idx else None,
        decomposition_method=decompose_fn,
        bootstrap_method=bootstrap_method,
    )
    pred = set(sig_names[sel_idx].tolist())
    true = set(true_sigs_list)
    return sample, true, pred


# ── Run one method on all samples ─────────────────────────────────────────────
def run_method(method_name, samples_df, gt_raw, cosmic_sub, sig_names,
               all_samples, mut_type, noise_level, bootstrap_method):
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
                      cosmic_sub.values.astype(float), sig_names,
                      R, PRE_FILTER, mandatory_idx, bootstrap_method, method_name))

    t0 = time.time()
    results = []
    total = len(tasks)
    print_every = max(1, total // 5)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_worker, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            sample, true, pred = fut.result()
            met = _metrics(true, pred)
            results.append(dict(
                sample=sample,
                n_true=len(true),
                **{f"n_pred_{method_name}": len(pred)},
                **{f"{method_name}_tp":        met["tp"]},
                **{f"{method_name}_fp":        met["fp"]},
                **{f"{method_name}_fn":        met["fn"]},
                **{f"{method_name}_precision": met["precision"]},
                **{f"{method_name}_recall":    met["recall"]},
                **{f"{method_name}_f1":        met["f1"]},
                **{f"{method_name}_pred_sigs": ",".join(sorted(pred))},
                true_sigs=",".join(sorted(true)),
            ))
            done += 1
            if done % print_every == 0 or done == total:
                print(f"    [{method_name} {mut_type}/{noise_level}] "
                      f"{done}/{total}  ({time.time()-t0:.1f}s)")

    return pd.DataFrame(results), round(time.time() - t0, 1)


# ── One round (mut_type × noise_level) ───────────────────────────────────────
def run_round(mut_type, noise_level, cfg, max_samples, bootstrap_method):
    base = BENCHMARK_DIR
    cosmic  = pd.read_csv(base / cfg["cosmic"], sep="\t", index_col=0)
    gt_raw  = pd.read_csv(base / cfg["gt"], index_col=0)
    samples = pd.read_csv(base / cfg["samples"][noise_level], sep="\t", index_col=0)

    common_idx = cosmic.index.intersection(samples.index)
    cosmic  = cosmic.loc[common_idx]
    samples = samples.loc[common_idx]

    gt_sigs    = [s for s in gt_raw.index if s in cosmic.columns]
    cosmic_sub = cosmic[gt_sigs]
    sig_names  = np.array(gt_sigs)

    all_samples = samples.columns.tolist()
    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    timings = {}
    dfs = {}
    for method_name in METHODS:
        print(f"  → {method_name} ...")
        df, elapsed = run_method(
            method_name, samples, gt_raw, cosmic_sub, sig_names,
            all_samples, mut_type, noise_level, bootstrap_method,
        )
        dfs[method_name] = df.set_index("sample")
        timings[method_name] = elapsed

    # Merge on sample; keep true_sigs from QP (identical for both)
    combined = dfs["qp"].copy()
    nnls_cols = [c for c in dfs["nnls"].columns if c != "true_sigs" and c != "n_true"]
    combined = combined.join(dfs["nnls"][nnls_cols], how="left")
    combined.insert(0, "noise",    noise_level)
    combined.insert(0, "mut_type", mut_type)
    combined = combined.reset_index().rename(columns={"index": "sample"})

    qp_agg   = _aggregate(combined, "qp")
    nnls_agg = _aggregate(combined, "nnls")

    summary = {
        "mut_type":    mut_type,
        "noise":       noise_level,
        "n_samples":   len(all_samples),
        **qp_agg,
        **nnls_agg,
        "qp_time_s":   timings["qp"],
        "nnls_time_s": timings["nnls"],
    }
    return combined, summary


# ── Print comparison tables ───────────────────────────────────────────────────
def _fmt_float(df: pd.DataFrame, cols: list, ndigits: int = 4) -> pd.DataFrame:
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
        "qp_mean_precision",   "nnls_mean_precision",
        "qp_mean_recall",      "nnls_mean_recall",
        "qp_mean_f1",          "nnls_mean_f1",
    ]
    print(_fmt_float(df, cols, 4).to_string(index=False))

    print("\n── Micro-averaged (from pooled TP/FP/FN across samples) ────────────────")
    cols = base + [
        "qp_micro_precision",   "nnls_micro_precision",
        "qp_micro_recall",      "nnls_micro_recall",
        "qp_micro_f1",          "nnls_micro_f1",
    ]
    print(_fmt_float(df, cols, 4).to_string(index=False))

    print("\n── Runtime (s) ─────────────────────────────────────────────────────────")
    cols = base + ["qp_time_s", "nnls_time_s"]
    tdf = df[[c for c in cols if c in df.columns]].copy()
    for c in ("qp_time_s", "nnls_time_s"):
        if c in tdf.columns:
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce").map(
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
    print(tdf.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="sigconfide: QP vs NNLS decomposition – Diaz-Gay 2023 benchmark"
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
    p.add_argument("--out-per-sample", default="compare_qp_nnls_results.csv",
                   help="CSV for per-sample results")
    p.add_argument("--out-summary",    default="compare_qp_nnls_summary.csv",
                   help="CSV for aggregate metrics")
    p.add_argument("--bootstrap-method", default="poisson",
                   choices=["multinomial", "poisson"],
                   help="Bootstrap resampling strategy (default: multinomial)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    all_results   = []
    all_summaries = []
    t_global = time.time()

    for mut_type in args.mut_types:
        cfg = MUT_TYPES[mut_type]
        for noise_level in args.noise_levels:
            print(f"\n═══ {mut_type} / {noise_level} ═══")
            combined, summary = run_round(
                mut_type, noise_level, cfg,
                args.max_samples,
                bootstrap_method=args.bootstrap_method,
            )
            all_results.append(combined)
            all_summaries.append(summary)

            for method in ("qp", "nnls"):
                mp  = summary.get(f"{method}_mean_precision",  float("nan"))
                mr  = summary.get(f"{method}_mean_recall",     float("nan"))
                mf1 = summary.get(f"{method}_mean_f1",         float("nan"))
                up  = summary.get(f"{method}_micro_precision", float("nan"))
                ur  = summary.get(f"{method}_micro_recall",    float("nan"))
                uf1 = summary.get(f"{method}_micro_f1",        float("nan"))
                t   = summary.get(f"{method}_time_s",          float("nan"))
                print(
                    f"  {method:<4}  mean: P={mp:.3f}  R={mr:.3f}  F1={mf1:.3f}  |  "
                    f"micro: P={up:.3f}  R={ur:.3f}  F1={uf1:.3f}  ({t:.1f}s)"
                )

    print_comparison(all_summaries)
    print(f"\nTotal time: {time.time()-t_global:.1f}s")

    pd.concat(all_results, ignore_index=True).to_csv(args.out_per_sample, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out_summary, index=False)
    print(f"\nPer-sample results → {args.out_per_sample}")
    print(f"Summary            → {args.out_summary}")
