"""
Minimal example: running sigconfide on SBS data
(Diaz-Gay et al. 2023 – Supplementary benchmark).

For each chosen sample it:
  1. loads the COSMIC v3.3 SBS signature panel and the sample's mutational profile,
  2. selects the active signatures (hybrid_stepwise_selection),
  3. prints the selected signatures with their exposures,
  4. compares the result against the ground truth (if available).

At the end it prints an aggregate precision / recall / F1 over all processed
samples (mean over samples and micro-averaged from pooled TP/FP/FN).

Usage:
    python examples/run_sbs_example.py                        # all samples
    python examples/run_sbs_example.py "SP.Syn.Bladder-TCC::S.1"
    python examples/run_sbs_example.py S.1 S.2 S.3            # several by name
    python examples/run_sbs_example.py --max-samples 5        # first 5 samples
    python examples/run_sbs_example.py --sample-index 5 --noise noise5
    python examples/run_sbs_example.py --summary-only         # aggregate only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sigconfide.estimates.selection import hybrid_stepwise_selection

# ── Paths to the SBS data ───────────────────────────────────────────────────
DATA_DIR = Path("Supplementary_data_Diaz-Gay_et_al_2023_Benchmark") / "SBS"
COSMIC_FILE = DATA_DIR / "COSMIC_v3.3_SBS_GRCh37.txt"
GT_FILE = DATA_DIR / "ground.truth.syn.exposures.csv"
SAMPLE_FILES = {
    "clean": DATA_DIR / "Samples.txt",
    "noise5": DATA_DIR / "Samples_noise5.txt",
    "noise10": DATA_DIR / "Samples_noise10.txt",
}

# ── sigconfide parameters ───────────────────────────────────────────────────
R = 100                 # number of bootstrap replicates
PRE_FILTER = 0.001      # drop trace signatures up front (None = disabled)


def parse_args():
    p = argparse.ArgumentParser(description="sigconfide example on SBS data")
    p.add_argument("samples", nargs="*", default=None,
                   help="Sample column name(s) (default: all, or --sample-index)")
    p.add_argument("--sample-index", type=int, default=None,
                   help="Run a single sample by column index")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit the number of samples processed (for quick tests)")
    p.add_argument("--noise", choices=list(SAMPLE_FILES), default="clean",
                   help="Noise level of the input data (default clean)")
    p.add_argument("--summary-only", action="store_true",
                   help="Suppress per-sample output, print only the aggregate")
    return p.parse_args()


def select_samples(args, all_columns):
    """Resolve the CLI arguments into an ordered list of sample column names."""
    if args.samples:
        missing = [s for s in args.samples if s not in all_columns]
        if missing:
            raise SystemExit(
                f"No such sample(s): {missing}. "
                f"Available e.g.: {list(all_columns[:3])} ..."
            )
        chosen = list(args.samples)
    elif args.sample_index is not None:
        chosen = [all_columns[args.sample_index]]
    else:
        chosen = list(all_columns)

    if args.max_samples is not None:
        chosen = chosen[: args.max_samples]
    return chosen


def process_sample(sample, samples, P, sig_names, gt, quiet=False):
    """Run selection for one sample; return per-sample metrics (or None)."""
    m = samples[sample].values.astype(float)  # mutational profile (vector of 96)

    if not quiet:
        print(f"\nSample:            {sample}")
        print(f"Signature panel:   {P.shape[1]} COSMIC SBS signatures")
        print(f"Mutation count:    {int(m.sum())}")

    # Select active signatures + estimate exposures
    sel_idx, exposures, errors = hybrid_stepwise_selection(
        m, P, R=R, pre_filter_threshold=PRE_FILTER
    )
    pred = pd.Series(exposures, index=sig_names[sel_idx]).sort_values(ascending=False)

    if not quiet:
        print("Detected signatures (relative exposure):")
        for name, val in pred.items():
            print(f"  {name:<8} {val:6.3f}")

    # Compare against the ground truth
    if sample not in gt.columns:
        return None

    true_sigs = set(gt.index[gt[sample] > 0])
    pred_sigs = set(pred.index)
    tp = true_sigs & pred_sigs
    fp = pred_sigs - true_sigs
    fn = true_sigs - pred_sigs
    prec = len(tp) / len(pred_sigs) if pred_sigs else 0.0
    rec = len(tp) / len(true_sigs) if true_sigs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    if not quiet:
        print("Comparison with ground truth:")
        print(f"  true:     {sorted(true_sigs)}")
        print(f"  hits:     {sorted(tp)}")
        print(f"  false:    {sorted(fp)}")
        print(f"  missed:   {sorted(fn)}")
        print(f"  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

    return dict(tp=len(tp), fp=len(fp), fn=len(fn),
                precision=prec, recall=rec, f1=f1)


def print_summary(metrics, noise):
    """Aggregate per-sample metrics: mean over samples + micro-averaged."""
    if not metrics:
        print("\nNo samples with ground truth – nothing to aggregate.")
        return

    df = pd.DataFrame(metrics)
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    print(f"\n── Summary over {len(df)} samples (noise: {noise}) ──")
    print(
        f"  mean   precision={df['precision'].mean():.3f}  "
        f"recall={df['recall'].mean():.3f}  f1={df['f1'].mean():.3f}"
    )
    print(
        f"  micro  precision={micro_p:.3f}  "
        f"recall={micro_r:.3f}  f1={micro_f:.3f}"
    )


def main():
    args = parse_args()

    # 1. Load the data -------------------------------------------------------
    cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)  # 96 x N_signatures
    # 96 x N_samples
    samples = pd.read_csv(SAMPLE_FILES[args.noise], sep="\t", index_col=0)
    gt = pd.read_csv(GT_FILE, index_col=0)  # signatures x samples

    # Align the 96 mutation contexts between the panel and the samples
    common_idx = cosmic.index.intersection(samples.index)
    cosmic = cosmic.loc[common_idx]
    samples = samples.loc[common_idx]

    P = cosmic.values.astype(float)          # signature matrix (96 x N)
    sig_names = np.array(cosmic.columns)     # signature names (SBS1, SBS2, ...)

    # 2. Resolve which samples to run ---------------------------------------
    chosen = select_samples(args, samples.columns)
    print(f"Processing {len(chosen)} sample(s)  (noise: {args.noise})")

    # 3. Run each sample -----------------------------------------------------
    metrics = []
    for i, sample in enumerate(chosen, 1):
        if args.summary_only:
            print(f"\r  {i}/{len(chosen)} {sample:<40}", end="", flush=True)
        result = process_sample(
            sample, samples, P, sig_names, gt, quiet=args.summary_only
        )
        if result is not None:
            metrics.append(result)
    if args.summary_only:
        print()

    # 4. Aggregate -----------------------------------------------------------
    print_summary(metrics, args.noise)


if __name__ == "__main__":
    main()
