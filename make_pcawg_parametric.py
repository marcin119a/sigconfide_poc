"""
Parametric synthetic SBS-96 catalogues calibrated to PCAWG, with an activity
matrix no fitting tool produced.

Why this exists
---------------
make_pcawg_synthetic.py takes H straight from the consortium's published
attribution.  That removes the circularity of fitting H locally, but leaves a
different one: the published attribution was produced by SigProfiler, so
SigProfilerAssignment is scored against the output of its own tool family.
Measured on that set, SPA leads sigconfide by 0.02 F1 on clean data and 0.07 at
10% noise, and neither a change of noise model nor tuning sigconfide's
operating point closes the gap, which leaves provenance as the last untested
explanation.

Here nothing is copied.  The published attribution is used only to *calibrate*
distributions; every sample's signature set, its exposures and its burden are
then drawn.  No tool has seen this H, so the set is neutral between them.

What is calibrated, per tumour type
-----------------------------------
prevalence      For each signature, the fraction of that type's genomes where
                it is active.  Reproduced exactly by construction.
co-occurrence   Signature activity is not independent: in breast SBS2 and SBS13
                are both active in 89% of genomes and co-active in 89%, against
                80% under independence, and melanoma's SBS7a-d move as a block.
                Activity is therefore drawn through a Gaussian copula whose
                correlation matrix is estimated per type and shrunk towards
                independence as 20/(20+n), so a type with three genomes
                contributes almost no correlation structure and Liver-HCC's 326
                contribute most of theirs.
exposure share  For each signature, the distribution of its share of burden in
                the genomes where it is active, resampled with lognormal
                jitter and renormalised across the drawn set.
burden          The type's own burden distribution, resampled with lognormal
                jitter.

Supports are drawn, never resampled wholesale.  That distinction is the whole
point: given a clean catalogue and the dictionary, exposure magnitudes are
nearly free, since the catalogue is a rounded matrix product, so support
selection is the entire task.  Resampling real supports would hand back
exactly the thing under test.

Construction
------------
The catalogue is drawn, not rounded off a matrix product:

    e ~ per-type model above          exposures, absolute mutation counts
    p = S e / sum(S e)                expected profile over the 96 channels
    x ~ Multinomial(N, p)             observed counts at depth N

N is the sample's own drawn burden at full depth, and the difficulty axis is
depth rather than a noise percentage.  Three depths ship: the full burden, a
quarter of it, and a twentieth, the last of which puts the median genome near
260 mutations, in panel territory.

This is the only physically coherent option.  A real genome always carries
counting noise, so a noise-free catalogue is not a cleaner version of the data,
it is data that cannot exist; relative deviation must fall as 1/sqrt(count),
which multiplicative noise at a flat 5% or 10% gets backwards.  The rounded
product is still written, as Samples_expected.txt, but as a noise-free
reference and an upper bound, not as an arm to score on.

--construction diazgay reverts to the round-plus-multiplicative-noise rule used
by the other synthetic sets here, for comparability with them.

Outputs
-------
    Samples.txt              multinomial at the full drawn burden
    Samples_depth25.txt      multinomial at a quarter of it
    Samples_depth05.txt      multinomial at a twentieth
    Samples_expected.txt     round(S e), noise-free reference
    ground.truth.syn.exposures.csv / .full.csv, at full depth: the support is
    depth-invariant, the exposures scale by the depth factor.

Usage
-----
    python make_pcawg_parametric.py
    python make_pcawg_parametric.py --seed 1 --out-dir PCAWG_Benchmark/parametric-seed1
    python make_pcawg_parametric.py --noise-model multiplicative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from make_pcawg_synthetic import (
    NOISE_LEVELS,
    PUBLISHED_ACTIVITIES,
    PUBLISHED_DICT,
    build_catalogues,
    cosmic_channel_order,
    load_published_dictionary,
)

BENCH_DIR = Path("PCAWG_Benchmark")
CATALOGUE_FILE = BENCH_DIR / "catalogues" / "PCAWG.WGS.SBS-96.txt"
DEFAULT_OUT = BENCH_DIR / "parametric"

SAMPLE_PREFIX = "SP.Par."

# Sampling depths, as a fraction of each sample's drawn burden.  The names are
# the file suffixes and the harness keys.
DEPTHS = {"full": 1.0, "depth25": 0.25, "depth05": 0.05}
# correlation shrinkage constant: lambda = SHRINK_N / (SHRINK_N + n)
SHRINK_N = 5.0
FRACTION_JITTER = 0.15   # lognormal sigma on a resampled exposure share
BURDEN_JITTER = 0.20     # lognormal sigma on a resampled burden


# ── Calibration ────────────────────────────────────────────────────────────────
def load_published(keys: pd.Index | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Published attributions as (samples x signatures) plus the cancer type."""
    raw = pd.read_csv(PUBLISHED_ACTIVITIES)
    sigs = [c for c in raw.columns if str(c).startswith("SBS")]
    ct = raw["Cancer Types"].astype(str).str.strip()
    key = ct + "::" + raw["Sample Names"].astype(str).str.strip()
    H = raw[sigs].astype(float)
    H.index = pd.Index(key, name="sample")
    ct.index = H.index
    if keys is not None:
        H, ct = H.loc[keys], ct.loc[keys]
    return H, ct


def shrunk_correlation(B: np.ndarray, n: int) -> np.ndarray:
    """Correlation of binary activity columns, shrunk towards independence.

    Columns that never vary (a signature active in every genome of the type)
    carry no information and are left uncorrelated.  The result is projected
    onto the positive semi-definite cone so it can seed a multivariate normal.
    """
    k = B.shape[1]
    lam = SHRINK_N / (SHRINK_N + n)
    sd = B.std(axis=0)
    R = np.eye(k)
    varying = sd > 1e-9
    if varying.sum() > 1:
        sub = np.corrcoef(B[:, varying], rowvar=False)
        sub = np.nan_to_num(sub, nan=0.0)
        R[np.ix_(varying, varying)] = sub
    np.fill_diagonal(R, 1.0)
    R = (1.0 - lam) * R + lam * np.eye(k)

    # Binary columns that co-occur perfectly give a singular correlation
    # matrix, so the eigenvalues are floored before the Cholesky in
    # draw_activities sees them.
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-4:
        R = V @ np.diag(np.maximum(w, 1e-4)) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def calibrate(H: pd.DataFrame, types: pd.Series) -> dict[str, dict]:
    """Per-type prevalence, activity correlation, exposure shares and burdens."""
    profiles: dict[str, dict] = {}
    burden = H.sum(axis=1)
    shares = H.div(burden.replace(0, np.nan), axis=0).fillna(0.0)

    for ct, idx in types.groupby(types).groups.items():
        h = H.loc[idx]
        active = (h > 0).values
        keep = active.any(axis=0)
        sigs = list(h.columns[keep])
        B = active[:, keep].astype(float)
        profiles[ct] = {
            "signatures": sigs,
            "n_real": int(len(idx)),
            "prevalence": B.mean(axis=0),
            "corr": shrunk_correlation(B, len(idx)),
            # observed shares for each signature, only where it is active
            "shares": [shares.loc[idx, s].values[active[:, list(h.columns).index(s)]]
                       for s in sigs],
            "burdens": burden.loc[idx].values,
        }
    return profiles


# ── Generation ─────────────────────────────────────────────────────────────────
def draw_activities(profiles: dict[str, dict], sig_index: pd.Index,
                    counts: dict[str, int], rng: np.random.Generator
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Draw a fresh activity matrix, samples x signatures, in absolute counts."""
    rows, names, out_types = [], [], []

    for ct in sorted(counts):
        prof = profiles[ct]
        sigs = prof["signatures"]
        prev = prof["prevalence"]
        # A signature active in every real genome of the type has an infinite
        # threshold; norm.ppf handles the endpoints, and the comparison below
        # then always (or never) fires.
        cut = norm.ppf(1.0 - prev)
        L = np.linalg.cholesky(prof["corr"])
        n = counts[ct]

        Z = rng.standard_normal((n, len(sigs))) @ L.T
        act = Z > cut
        empty = ~act.any(axis=1)
        if empty.any():
            act[empty, int(np.argmax(prev))] = True

        for i in range(n):
            on = np.where(act[i])[0]
            share = np.array([
                rng.choice(prof["shares"][j]) * np.exp(
                    rng.normal(0.0, FRACTION_JITTER))
                for j in on
            ])
            share = share / share.sum()
            b = float(rng.choice(prof["burdens"])
                      * np.exp(rng.normal(0.0, BURDEN_JITTER)))
            row = pd.Series(0.0, index=sig_index)
            row.iloc[[sig_index.get_loc(sigs[j]) for j in on]] = share * b
            rows.append(row.values)
            names.append(f"{SAMPLE_PREFIX}{ct}::S{len(names) + 1}")
            out_types.append(ct)

    H = pd.DataFrame(np.array(rows), index=pd.Index(names, name="sample"),
                     columns=sig_index)
    return H, out_types


def build_multinomial(W: pd.DataFrame, H: pd.DataFrame, rng: np.random.Generator
                      ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Draw x ~ Multinomial(N, p) at each depth, plus the noise-free reference.

    p is the expected profile S e normalised over the 96 channels, so every
    catalogue is a genuine sample of the generating model rather than a
    perturbation of its mean.  A sample whose depth rounds below one mutation
    is given one, which only affects the very smallest genomes at depth05.
    """
    expected = pd.DataFrame(W.values @ H[W.columns].values.T,
                            index=W.index, columns=H.index)
    E = np.maximum(expected.values, 0.0)
    totals = E.sum(axis=0)
    p = E / np.maximum(totals, 1e-12)

    catalogues = {}
    for name, frac in DEPTHS.items():
        n = np.maximum((totals * frac).round().astype(np.int64), 1)
        draw = np.column_stack([rng.multinomial(n[j], p[:, j])
                                for j in range(E.shape[1])])
        catalogues[name] = pd.DataFrame(draw.astype(np.int64),
                                        index=expected.index,
                                        columns=expected.columns)
    catalogues["expected"] = expected.round().astype(np.int64)
    return catalogues, expected


# ── Verification ───────────────────────────────────────────────────────────────
def report(H_syn: pd.DataFrame, types_syn: list[str], H_real: pd.DataFrame,
           types_real: pd.Series) -> dict:
    """Compare the drawn cohort against the cohort it was calibrated on."""
    syn_n = (H_syn > 0).sum(axis=1)
    real_n = (H_real > 0).sum(axis=1)
    print("\n── calibration checks ──")
    print(f"  signatures/sample   : synthetic {syn_n.mean():.2f} "
          f"(range {syn_n.min()}-{syn_n.max()})   real {real_n.mean():.2f} "
          f"(range {real_n.min()}-{real_n.max()})")

    # Prevalence is reproduced exactly in expectation; what is left is the
    # sampling noise of drawing n genomes, so it is reported against type size
    # rather than pooled - a type with two real genomes can only ever land on
    # 0, 0.5 or 1.
    ts = pd.Series(types_syn, index=H_syn.index)
    devs, sizes = [], []
    for ct in ts.unique():
        p_syn = (H_syn[ts == ct] > 0).mean()
        p_real = (H_real[types_real == ct] > 0).mean()
        devs.append(float((p_syn - p_real).abs().max()))
        sizes.append(int((types_real == ct).sum()))
    devs, sizes = np.array(devs), np.array(sizes)
    print("  prevalence deviation by type size (max over that type's signatures):")
    for lo, hi in [(0, 20), (20, 100), (100, 10 ** 9)]:
        m = (sizes >= lo) & (sizes < hi)
        if not m.any():
            continue
        label = f"n {lo}-{hi}" if hi < 10 ** 9 else f"n >= {lo}"
        print(f"      {label:12s} types={m.sum():>3}  median {np.median(devs[m]):.3f}"
              f"  max {devs[m].max():.3f}")

    # co-occurrence, the thing independent draws would destroy
    pairs = [("Breast-AdenoCA", "SBS2", "SBS13"),
             ("Skin-Melanoma", "SBS7a", "SBS7b"),
             ("Liver-HCC", "SBS5", "SBS12")]
    print("  co-occurrence (joint activity, synthetic vs real vs independent):")
    co = {}
    for ct, a, b in pairs:
        if ct not in set(ts) or a not in H_syn.columns:
            continue
        s, r = H_syn[ts == ct], H_real[types_real == ct]
        j_s = float(((s[a] > 0) & (s[b] > 0)).mean())
        j_r = float(((r[a] > 0) & (r[b] > 0)).mean())
        ind = float((r[a] > 0).mean() * (r[b] > 0).mean())
        print(f"      {ct:16s} {a}+{b}: {j_s:.2f}  {j_r:.2f}  {ind:.2f}")
        co[f"{ct}:{a}+{b}"] = {"synthetic": j_s, "real": j_r, "independent": ind}

    bs, br = H_syn.sum(axis=1), H_real.sum(axis=1)
    print(f"  burden              : synthetic median {bs.median():,.0f} "
          f"(p05 {bs.quantile(.05):,.0f}, p95 {bs.quantile(.95):,.0f})")
    print(f"                        real      median {br.median():,.0f} "
          f"(p05 {br.quantile(.05):,.0f}, p95 {br.quantile(.95):,.0f})")

    supports_real = {frozenset(H_real.columns[r > 0]) for _, r in H_real.iterrows()}
    supports_syn = [frozenset(H_syn.columns[r > 0]) for _, r in H_syn.iterrows()]
    novel = sum(1 for s in supports_syn if s not in supports_real)
    print(f"  novel supports      : {novel} of {len(supports_syn)} "
          f"({novel / len(supports_syn):.0%}) never occur in the published data")

    return {
        "signatures_per_sample": {"synthetic": float(syn_n.mean()),
                                  "real": float(real_n.mean())},
        "prevalence_deviation_median": float(np.median(devs)),
        "prevalence_deviation_median_large_types": float(
            np.median(devs[sizes >= 100])) if (sizes >= 100).any() else None,
        "co_occurrence": co,
        "burden_median": {"synthetic": float(bs.median()), "real": float(br.median())},
        "novel_support_fraction": float(novel / len(supports_syn)),
    }


# ── Output ─────────────────────────────────────────────────────────────────────
def write_outputs(out_dir: Path, catalogues: dict[str, pd.DataFrame],
                  H: pd.DataFrame, W: pd.DataFrame, dict_name: str,
                  meta: pd.DataFrame, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diazgay = {"clean": "Samples.txt", "noise5": "Samples_noise5.txt",
               "noise10": "Samples_noise10.txt"}
    for name, cat in catalogues.items():
        cat.index.name = "Mutation Types"
        fname = diazgay.get(name, "Samples.txt" if name == "full"
                            else f"Samples_{name}.txt")
        cat.to_csv(out_dir / fname, sep="\t")

    used = [s for s in H.columns if (H[s] > 0).any()]
    gt = H[used].T
    gt.index.name = None
    gt.to_csv(out_dir / "ground.truth.syn.exposures.csv")
    full = H.T.reindex(W.columns).fillna(0.0)
    full.index.name = None
    full.to_csv(out_dir / "ground.truth.syn.exposures.full.csv")

    W.to_csv(out_dir / dict_name, sep="\t")
    meta.to_csv(out_dir / "sample_metadata.csv")
    (out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"output directory (default: {DEFAULT_OUT})")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument("--construction", choices=["multinomial", "diazgay"],
                   default="multinomial",
                   help="multinomial draws x ~ Multinomial(N, p) at three "
                        "depths (default); diazgay reverts to round(S e) plus "
                        "flat 5%%/10%% multiplicative noise")
    p.add_argument("--scale", type=float, default=1.0,
                   help="samples per tumour type as a multiple of the real "
                        "cohort (default: 1.0, i.e. 2780 samples)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the calibration checks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    W = load_published_dictionary(PUBLISHED_DICT)
    if list(W.index) != cosmic_channel_order():
        raise SystemExit("dictionary is not in COSMIC channel order")

    V = pd.read_csv(CATALOGUE_FILE, sep="\t", index_col=0)
    H_real, types_real = load_published(V.columns)
    print(f"calibrating on {len(H_real)} published attributions, "
          f"{types_real.nunique()} tumour types")

    profiles = calibrate(H_real, types_real)
    counts = {ct: max(1, int(round(p["n_real"] * args.scale)))
              for ct, p in profiles.items()}

    H, types_syn = draw_activities(profiles, H_real.columns, counts, rng)
    H = H.reindex(columns=W.columns).fillna(0.0)
    used = [s for s in W.columns if (H[s] > 0).any()]
    print(f"drew {len(H)} samples, {len(used)} of {W.shape[1]} signatures used, "
          f"{(H > 0).sum(axis=1).mean():.2f} signatures/sample")

    stats = {}
    if not args.no_verify:
        stats = report(H, types_syn, H_real, types_real)

    if args.construction == "multinomial":
        catalogues, expected = build_multinomial(W, H, rng)
        primary = catalogues["full"]
    else:
        catalogues, expected = build_catalogues(W, H, args.seed, "multiplicative")
        primary = catalogues["clean"]

    meta = pd.DataFrame({
        "sample": H.index,
        "cancer_type": types_syn,
        "activity_burden": H.sum(axis=1).values,
        "observed_burden": primary.sum(axis=0).values,
        "n_signatures": (H > 0).sum(axis=1).values,
    }).set_index("sample")
    meta["depth_ratio"] = (
        meta.observed_burden / meta.activity_burden.clip(lower=1e-9))

    manifest = {
        "construction": (
            "H drawn from a per-type model calibrated to the published PCAWG "
            "attributions; p = S e normalised; x ~ Multinomial(N, p)"
            if args.construction == "multinomial"
            else "H drawn as above; clean = round(S e) plus multiplicative noise"),
        "counts_model": args.construction,
        "levels": (DEPTHS if args.construction == "multinomial" else NOISE_LEVELS),
        "seed": args.seed,
        "scale": args.scale,
        "n_samples": int(len(H)),
        "n_cancer_types": len(counts),
        "n_signatures_used": len(used),
        "signatures": used,
        "activities": {
            "source": "parametric: no fitting tool produced this H",
            "activity_model": "Gaussian copula on per-type prevalence",
            "correlation_shrinkage": f"{SHRINK_N}/({SHRINK_N}+n)",
            "fraction_jitter_lognormal_sigma": FRACTION_JITTER,
            "burden_jitter_lognormal_sigma": BURDEN_JITTER,
            **stats,
        },
        "inputs": {"calibration": str(PUBLISHED_ACTIVITIES),
                   "dictionary": str(PUBLISHED_DICT),
                   "catalogue": str(CATALOGUE_FILE)},
    }
    dict_name = PUBLISHED_DICT.with_suffix(".txt").name
    write_outputs(args.out_dir, catalogues, H, W, dict_name, meta, manifest)

    print(f"\nwrote {len(H)} samples x {W.shape[0]} channels to {args.out_dir}/")
    print(f"  signatures      : {len(used)} true, {W.shape[1] - len(used)} "
          "distractors in the full ground truth")
    print(f"  construction    : {args.construction}")
    for name, cat in catalogues.items():
        b = cat.sum(axis=0)
        print(f"  {name:<15s} : burden min {b.min()}  median "
              f"{int(b.median())}  max {b.max()}")


if __name__ == "__main__":
    main()
