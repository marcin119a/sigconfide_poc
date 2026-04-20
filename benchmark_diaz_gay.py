"""
Benchmark sigconfide on data from:
  Diaz-Gay et al. 2023 – Supplementary benchmark dataset

Supported mutation types : SBS, DBS, ID, CN
Noise levels             : clean, noise5, noise10
Metrics                  : precision / recall / F1 (per-sample mean + micro)

Outputs written to:
  benchmark_diaz_gay_results.csv   – per-sample
  benchmark_diaz_gay_summary.csv   – aggregates per (mut_type, noise)
"""

import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from sigconfide.estimates.selection import hybrid_stepwise_selection

# ── Configuration ──────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path("Supplementary_data_Diaz-Gay_et_al_2023_Benchmark")

MUT_TYPES = {
    "SBS": {
        "cosmic":    "SBS/COSMIC_v3.3_SBS_GRCh37.txt",
        "gt":        "SBS/ground.truth.syn.exposures.csv",
        "samples": {
            "clean":   "SBS/Samples.txt",
            "noise5":  "SBS/Samples_noise5.txt",
            "noise10": "SBS/Samples_noise10.txt",
        },
    },
    "DBS": {
        "cosmic":    "DBS/COSMIC_v3.3_DBS_GRCh37.txt",
        "gt":        "DBS/ground.truth.syn.exposures.DBS.csv",
        "samples": {
            "clean":   "DBS/Samples_DBS.txt",
            "noise5":  "DBS/Samples_DBS_noise5.txt",
            "noise10": "DBS/Samples_DBS_noise10.txt",
        },
    },
    "ID": {
        "cosmic":    "ID/COSMIC_v3.3_ID_GRCh37.txt",
        "gt":        "ID/ground.truth.syn.exposures.ID.csv",
        "samples": {
            "clean":   "ID/Samples_ID.txt",
            "noise5":  "ID/Samples_ID_noise5.txt",
            "noise10": "ID/Samples_ID_noise10.txt",
        },
    },
    "CN": {
        "cosmic":    "CN/COSMIC_v3.3_CN_GRCh37.txt",
        "gt":        "CN/ground.truth.syn.exposures.CN.csv",
        "samples": {
            "clean":   "CN/Samples_CN.txt",
            "noise5":  "CN/Samples_CN_noise5.txt",
            "noise10": "CN/Samples_CN_noise10.txt",
        },
    },
}

# sigconfide parameters
R              = 100       # number of bootstraps
PRE_FILTER     = 0.001    # pre-filter trace signatures (None = disabled)
MAX_WORKERS    = None     # None = auto (CPU count)


# ── Worker (runs in a separate process) ─────────────────────────────────────
def run_sample(args):
    sample, m, true_sigs_set, P, sig_names, R, pre_filter, bootstrap_method = args
    sel_idx, _, _ = hybrid_stepwise_selection(
        m, P, R=R, pre_filter_threshold=pre_filter, bootstrap_method=bootstrap_method
    )
    pred_sigs = set(sig_names[sel_idx].tolist())
    true_sigs = set(true_sigs_set)

    tp = len(true_sigs & pred_sigs)
    fp = len(pred_sigs - true_sigs)
    fn = len(true_sigs - pred_sigs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "sample":    sample,
        "n_true":    len(true_sigs),
        "n_pred":    len(pred_sigs),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "true_sigs": ",".join(sorted(true_sigs)),
        "pred_sigs": ",".join(sorted(pred_sigs)),
    }


# ── One benchmark round (mut_type × noise_level) ───────────────────────────
def run_benchmark(mut_type: str, noise_level: str, cfg: dict, max_samples=None,
                  bootstrap_method='multinomial'):
    base = BENCHMARK_DIR
    cosmic   = pd.read_csv(base / cfg["cosmic"], sep="\t", index_col=0)
    gt_raw   = pd.read_csv(base / cfg["gt"], index_col=0)      # rows=sigs, cols=samples
    samples  = pd.read_csv(base / cfg["samples"][noise_level], sep="\t", index_col=0)

    # Align mutation (row) indices between COSMIC and samples
    common_idx = cosmic.index.intersection(samples.index)
    cosmic  = cosmic.loc[common_idx]
    samples = samples.loc[common_idx]

    # Restrict COSMIC to signatures present in GT (realistic for this benchmark)
    gt_sigs    = [s for s in gt_raw.index if s in cosmic.columns]
    cosmic_sub = cosmic[gt_sigs]

    P         = cosmic_sub.values.astype(float)
    sig_names = np.array(gt_sigs)
    all_samples = samples.columns.tolist()

    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    # Normalized column name → original GT column name
    # Needed when Samples.txt replaces '-' and ':' with '.' (DBS, ID)
    def _norm(s):
        return s.replace("-", ".").replace(":", ".")

    gt_col_map = {_norm(c): c for c in gt_raw.columns}

    # Build tasks
    tasks = []
    for sample in all_samples:
        m = samples[sample].values.astype(float)
        # ground truth: signatures with exposure > 0 for this sample
        gt_col = sample if sample in gt_raw.columns \
                 else gt_col_map.get(_norm(sample))
        if gt_col is not None:
            true_sigs = gt_raw.index[gt_raw[gt_col] > 0].tolist()
        else:
            true_sigs = []
        tasks.append((sample, m, true_sigs, P, sig_names, R, PRE_FILTER, bootstrap_method))

    t0 = time.time()
    results = []
    done = 0
    total = len(tasks)
    print_every = max(1, total // 10)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(run_sample, task): task[0] for task in tasks}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % print_every == 0 or done == total:
                elapsed = time.time() - t0
                print(f"    [{mut_type}/{noise_level}] {done}/{total}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    df = pd.DataFrame(results).sort_values("sample").reset_index(drop=True)
    df.insert(0, "noise",    noise_level)
    df.insert(0, "mut_type", mut_type)

    # Aggregates
    mean_p  = df["precision"].mean()
    mean_r  = df["recall"].mean()
    mean_f1 = df["f1"].mean()
    tp_sum  = df["tp"].sum()
    fp_sum  = df["fp"].sum()
    fn_sum  = df["fn"].sum()
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) > 0 else 0.0)

    summary = {
        "mut_type":       mut_type,
        "noise":          noise_level,
        "n_samples":      total,
        "mean_precision": mean_p,
        "mean_recall":    mean_r,
        "mean_f1":        mean_f1,
        "micro_precision": micro_p,
        "micro_recall":   micro_r,
        "micro_f1":       micro_f,
        "elapsed_s":      round(elapsed, 1),
    }

    return df, summary


# ── Print summary table ───────────────────────────────────────────────────────
def print_summary_table(summaries):
    df = pd.DataFrame(summaries)
    cols = ["mut_type", "noise", "n_samples",
            "mean_precision", "mean_recall", "mean_f1",
            "micro_precision", "micro_recall", "micro_f1"]
    df = df[cols]
    for c in cols[3:]:
        df[c] = df[c].map("{:.3f}".format)
    print("\n── Summary ──────────────────────────────────────────────────────────────")
    print(df.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark sigconfide on Diaz-Gay et al. 2023 data"
    )
    p.add_argument(
        "--mut-types", nargs="+", default=list(MUT_TYPES.keys()),
        choices=list(MUT_TYPES.keys()),
        help="Mutation types to run (default: all)",
    )
    p.add_argument(
        "--noise-levels", nargs="+", default=["clean", "noise5", "noise10"],
        choices=["clean", "noise5", "noise10"],
        help="Noise levels (default: all)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for quick tests)",
    )
    p.add_argument(
        "--out-per-sample", default="benchmark_diaz_gay_results.csv",
        help="CSV file for per-sample results",
    )
    p.add_argument(
        "--out-summary", default="benchmark_diaz_gay_summary.csv",
        help="CSV file for aggregate metrics",
    )
    p.add_argument(
        "--bootstrap-method", default="multinomial",
        choices=["multinomial", "poisson"],
        help="Bootstrap resampling strategy: 'multinomial' (fixed total count, default) "
             "or 'poisson' (independent Poisson draws per mutation type)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results  = []
    all_summaries = []
    t_global = time.time()

    for mut_type in args.mut_types:
        cfg = MUT_TYPES[mut_type]
        for noise_level in args.noise_levels:
            print(f"\n→ {mut_type} / {noise_level}")
            df, summary = run_benchmark(
                mut_type, noise_level, cfg,
                max_samples=args.max_samples,
                bootstrap_method=args.bootstrap_method,
            )
            all_results.append(df)
            all_summaries.append(summary)

            # Quick stats printout
            print(f"   mean  P={summary['mean_precision']:.3f}  "
                  f"R={summary['mean_recall']:.3f}  "
                  f"F1={summary['mean_f1']:.3f}")
            print(f"   micro P={summary['micro_precision']:.3f}  "
                  f"R={summary['micro_recall']:.3f}  "
                  f"F1={summary['micro_f1']:.3f}  "
                  f"({summary['elapsed_s']}s)")

    print_summary_table(all_summaries)

    total_elapsed = time.time() - t_global
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Write outputs
    pd.concat(all_results, ignore_index=True).to_csv(args.out_per_sample, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out_summary, index=False)
    print(f"\nPer-sample results → {args.out_per_sample}")
    print(f"Summary            → {args.out_summary}")
