"""
Minimal example: running sigconfide on SBS data
(Diaz-Gay et al. 2023 – Supplementary benchmark).

For a chosen sample it:
  1. loads the COSMIC v3.3 SBS signature panel and the sample's mutational profile,
  2. selects the active signatures (hybrid_stepwise_selection),
  3. prints the selected signatures with their exposures,
  4. compares the result against the ground truth (if available).

Usage:
    python examples/run_sbs_example.py                       # first sample
    python examples/run_sbs_example.py "SP.Syn.Bladder-TCC::S.1"
    python examples/run_sbs_example.py --sample-index 5 --noise noise5
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
    p.add_argument("sample", nargs="?", default=None,
                   help="Sample column name (default: the one picked by --sample-index)")
    p.add_argument("--sample-index", type=int, default=0,
                   help="Sample index used when no name is given (default 0)")
    p.add_argument("--noise", choices=list(SAMPLE_FILES), default="clean",
                   help="Noise level of the input data (default clean)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load the data -------------------------------------------------------
    cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)          # 96 x N_signatures
    samples = pd.read_csv(SAMPLE_FILES[args.noise], sep="\t", index_col=0)  # 96 x N_samples
    gt = pd.read_csv(GT_FILE, index_col=0)                            # signatures x samples

    # Align the 96 mutation contexts between the panel and the samples
    common_idx = cosmic.index.intersection(samples.index)
    cosmic = cosmic.loc[common_idx]
    samples = samples.loc[common_idx]

    # Pick a sample
    sample = args.sample or samples.columns[args.sample_index]
    if sample not in samples.columns:
        raise SystemExit(f"No sample '{sample}'. Available e.g.: {list(samples.columns[:3])} ...")

    P = cosmic.values.astype(float)          # signature matrix (96 x N)
    sig_names = np.array(cosmic.columns)     # signature names (SBS1, SBS2, ...)
    m = samples[sample].values.astype(float)  # sample's mutational profile (vector of 96)

    print(f"Sample:            {sample}  (noise: {args.noise})")
    print(f"Signature panel:   {P.shape[1]} COSMIC SBS signatures")
    print(f"Mutation count:    {int(m.sum())}")

    # 2. Select active signatures + estimate exposures ----------------------
    sel_idx, exposures, errors = hybrid_stepwise_selection(
        m, P, R=R, pre_filter_threshold=PRE_FILTER
    )

    pred = pd.Series(exposures, index=sig_names[sel_idx]).sort_values(ascending=False)

    print("\nDetected signatures (relative exposure):")
    for name, val in pred.items():
        print(f"  {name:<8} {val:6.3f}")

    # 3. Compare against the ground truth -----------------------------------
    if sample in gt.columns:
        true_sigs = set(gt.index[gt[sample] > 0])
        pred_sigs = set(pred.index)
        tp = true_sigs & pred_sigs
        fp = pred_sigs - true_sigs
        fn = true_sigs - pred_sigs
        prec = len(tp) / len(pred_sigs) if pred_sigs else 0.0
        rec = len(tp) / len(true_sigs) if true_sigs else 0.0
        print("\nComparison with ground truth:")
        print(f"  true:     {sorted(true_sigs)}")
        print(f"  hits:     {sorted(tp)}")
        print(f"  false:    {sorted(fp)}")
        print(f"  missed:   {sorted(fn)}")
        print(f"  precision={prec:.3f}  recall={rec:.3f}")


if __name__ == "__main__":
    main()
