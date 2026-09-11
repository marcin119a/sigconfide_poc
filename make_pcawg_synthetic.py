"""
Diaz-Gay-style synthetic SBS-96 catalogues built from the PCAWG WGS cohort.

Same construction as make_breast_synthetic.py, driven by the 2780 PCAWG whole
genomes in PCAWG_Benchmark/ instead of the 560 breast genomes:

    clean        = round(W @ H)                      # deterministic product
    noise-x%     = floor(clean * (1 + (x/100) * Z))  # Z ~ N(0,1), clipped at 0

with a SINGLE draw of Z shared across noise levels, so noise10 is noise5 with
the same standard normals scaled by two.  As in the breast set there is no
Poisson or multinomial resampling: the clean catalogue carries no counting
noise, and the perturbation is proportional to each channel's count.

Where H comes from
------------------
Default, --activities published: the consortium's own per-sample attributions
from Alexandrov et al. 2020, `PCAWG_sigProfiler_SBS_signatures_in_samples.csv`,
against the signature definitions those counts were computed with,
`sigProfiler_SBS_signatures_2019_05_22.csv`.  This is the exact analogue of the
breast set reading H off Nik-Zainal Table 21, and it is the reason to prefer
this mode: the ground truth was produced by a third party, before this
repository existed, by a procedure none of the methods under test implement.

The provenance is checkable rather than assumed.  The attribution file carries
an `Accuracy` column, which is the consortium's own cosine between W @ H and
the real catalogue.  Recomputing that cosine here reproduces their column to a
median absolute difference of 3e-4, correlation 1.0000, which is what proves
the W and the H assembled here are the pair they used.  The generator asserts
it.

Alternative, --activities fitted: H is fitted from the real catalogues by NNLS
onto the chosen dictionary, then restricted by a per-tumour-type dictionary
(signatures clearing --min-frac of burden in --min-prevalence of that type's
samples), then refitted.  This mode exists for comparison only and its output
should not be used to rank selection methods.  Reproducing the construction
rule -- NNLS then a 3% threshold, three lines and no model -- scores F1 0.956
on the fitted clean set against sigconfide's 0.925, because H was *defined* as
NNLS-plus-threshold and anything more conservative loses recall at the
boundary.  A benchmark its own null model wins is measuring the construction.

What the two modes look like
----------------------------
                          published        fitted (min-frac 0.03)
    signatures/sample        3.95              8.86
    signatures used         54 of 65          63 of 78
    reconstruction cosine    0.969             0.986

The fitted mode reconstructs the real catalogues better precisely because it
overfits them; 8.86 signatures explain more than 3.95.  Better reconstruction
is not better ground truth here.

Low burden
----------
PCAWG burdens span 21 to 2.4M mutations.  Rounding in `clean` costs the
low-burden samples part of their mutations; they are kept and flagged as
`rounding_loss` in sample_metadata.csv rather than dropped, and --min-burden
filters them out.  Below roughly 100 mutations this construction is the wrong
tool and a multinomial generator is the right one.

Outputs mirror the Diaz-Gay layout, as the breast set does
----------------------------------------------------------
    Samples.txt / Samples_noise5.txt / Samples_noise10.txt
        96 channels x n samples, integer counts, index "Mutation Types",
        channels in the dictionary's own order.
    ground.truth.syn.exposures.csv
        Signatures actually used x samples, the no-distractor variant.
    ground.truth.syn.exposures.full.csv
        Padded to the full dictionary with zero rows, so the unused signatures
        become distractors that can only ever be false positives.
    <dictionary>.txt
        The fitting dictionary, tab-separated, channels in COSMIC order.
    sample_metadata.csv, generation_manifest.json

Usage
-----
    python make_pcawg_synthetic.py
    python make_pcawg_synthetic.py --seed 1 --out-dir PCAWG_Benchmark/synthetic-seed1
    python make_pcawg_synthetic.py --activities fitted --min-frac 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCH_DIR = Path("PCAWG_Benchmark")
CATALOGUE_FILE = BENCH_DIR / "catalogues" / "PCAWG.WGS.SBS-96.txt"
METADATA_FILE = BENCH_DIR / "pcawg_sample_metadata.csv"
PUBLISHED_DIR = BENCH_DIR / "published"
PUBLISHED_ACTIVITIES = (PUBLISHED_DIR
                        / "PCAWG_sigProfiler_SBS_signatures_in_samples.csv")
PUBLISHED_DICT = PUBLISHED_DIR / "sigProfiler_SBS_signatures_2019_05_22.csv"
FITTED_DICT = Path(
    "Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/COSMIC_v3.3_SBS_GRCh37.txt"
)
DEFAULT_OUT = BENCH_DIR / "synthetic"

# ── Construction constants (measured on the Diaz-Gay files) ────────────────────
NOISE_LEVELS = {"noise5": 0.05, "noise10": 0.10}
SAMPLE_PREFIX = "SP.Syn."

# Depth of the counting model's second arm, as a fraction of the real burden.
COUNTING_DEPTH = 0.25

# Reference values from the Diaz-Gay SBS set, printed alongside ours by verify().
DG_REFERENCE = {
    "clean": {"chi2_df": 0.004, "cosine": 1.0000},
    "noise5": {"chi2_df": 0.499, "cosine": 0.9989, "sd_rel": 0.050},
    "noise10": {"chi2_df": 1.951, "cosine": 0.9959, "sd_rel": 0.100},
}

# Asserted so a catalogue swap is caught the way the breast generator catches a
# workbook swap.
PCAWG_N = 2780
PCAWG_TYPES = 37
PCAWG_TOTAL = 48_276_930

# Published attribution properties, likewise asserted.
PUBLISHED_SIGS = 65
PUBLISHED_USED = 54
ACCURACY_TOL = 0.002


# COSMIC orders the 96 channels by 5' base, then substitution, then 3' base.
# The published sigProfiler file is substitution-major instead, and the two are
# not interchangeable downstream: sigconfide multiplies catalogue against
# dictionary row-wise, and SigProfilerAssignment reads a custom signature
# database positionally, so it only assigns correctly when both files are in
# COSMIC's own order.  Everything written out is therefore put in that order,
# which is also what the breast set and the Diaz-Gay files use.
SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]


def cosmic_channel_order() -> list[str]:
    return [f"{a}[{s}]{b}" for a in BASES for s in SUBS for b in BASES]


# ── Inputs ─────────────────────────────────────────────────────────────────────
def load_catalogue(channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real PCAWG counts as (samples x 96) in `channels` order, plus metadata.

    The benchmark catalogue is written channels-in-rows in the mutation-class
    order the ICGC-BRCA tables use; the dictionaries are in COSMIC's 5'-base
    order.  Reindexing here is what keeps the row-wise product meaningful.
    """
    V = pd.read_csv(CATALOGUE_FILE, sep="\t", index_col=0).T
    missing = [c for c in channels if c not in V.columns]
    if missing:
        raise SystemExit(f"catalogue is missing channels: {missing[:5]}")
    V = V[channels]

    meta = pd.read_csv(METADATA_FILE).set_index("source_column").loc[V.index]

    if len(V) != PCAWG_N:
        raise SystemExit(f"expected {PCAWG_N} samples, got {len(V)}")
    if meta.cancer_type.nunique() != PCAWG_TYPES:
        raise SystemExit(f"expected {PCAWG_TYPES} cancer types")
    if int(V.values.sum()) != PCAWG_TOTAL:
        raise SystemExit(f"catalogue total changed: {int(V.values.sum())}")
    return V, meta


def load_cosmic_dictionary(path: Path) -> pd.DataFrame:
    """A tab-separated COSMIC dictionary, 96 channels x k signatures."""
    W = pd.read_csv(path, sep="\t", index_col=0)
    W.index.name = "Mutation Types"
    W = W.loc[cosmic_channel_order()]
    colsums = W.sum(axis=0)
    if not np.allclose(colsums, 1.0, atol=1e-4):
        bad = colsums[(colsums - 1.0).abs() > 1e-4]
        raise SystemExit(f"dictionary columns not normalised: {bad.to_dict()}")
    return W


def load_published_dictionary(path: Path) -> pd.DataFrame:
    """The sigProfiler definitions the published attributions were computed with.

    Ships as "Mutation.type,Trinucleotide,<signatures>" with values rounded to
    three significant figures, so columns sum to 1 only to about 5e-4 and are
    renormalised here.
    """
    raw = pd.read_csv(path)
    sub = raw["Mutation.type"].astype(str).str.strip()
    ctx = raw["Trinucleotide"].astype(str).str.strip()
    W = raw.iloc[:, 2:].astype(float)
    W.index = pd.Index(ctx.str[0] + "[" + sub + "]" + ctx.str[2],
                       name="Mutation Types")
    drift = float((W.sum(axis=0) - 1.0).abs().max())
    if drift > 5e-3:
        raise SystemExit(f"published dictionary is off by {drift:.4f}, not rounding")
    return (W / W.sum(axis=0)).loc[cosmic_channel_order()]


def load_published_activities(path: Path, keys: pd.Index
                              ) -> tuple[pd.DataFrame, pd.Series]:
    """Consortium attributions as (samples x 65) in absolute mutation counts.

    Keyed by "<CancerType>::<SP-id>", which is exactly how the catalogue names
    its columns, so the join needs no ID mapping at all.  Returns the
    activities and the published Accuracy column.
    """
    raw = pd.read_csv(path)
    sigs = [c for c in raw.columns if str(c).startswith("SBS")]
    if len(sigs) != PUBLISHED_SIGS:
        raise SystemExit(f"expected {PUBLISHED_SIGS} signatures, got {len(sigs)}")

    key = (raw["Cancer Types"].astype(str).str.strip() + "::"
           + raw["Sample Names"].astype(str).str.strip())
    H = raw[sigs].astype(float)
    H.index = pd.Index(key, name="sample")
    if not H.index.is_unique:
        raise SystemExit("published attributions contain duplicate sample keys")

    missing = [k for k in keys if k not in H.index]
    if missing:
        raise SystemExit(f"{len(missing)} catalogue samples have no attribution: "
                         f"{missing[:3]}")
    accuracy = raw["Accuracy"].astype(float)
    accuracy.index = H.index
    return H.loc[keys], accuracy.loc[keys]


# ── Activities: the fitted alternative ─────────────────────────────────────────
def _cosine(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Column-wise cosine between two 96 x n matrices."""
    norms = np.linalg.norm(P, axis=0) * np.linalg.norm(Q, axis=0)
    return np.divide((P * Q).sum(0), norms,
                     out=np.zeros(Q.shape[1]), where=norms > 0)


def fit_cosine(V: pd.DataFrame, W: pd.DataFrame, H: pd.DataFrame) -> np.ndarray:
    """How well the activities reproduce each real sample."""
    recon = W.values @ H[W.columns].values.T
    return _cosine(recon, V[W.index].values.astype(float).T)


def derive_activities(V: pd.DataFrame, W: pd.DataFrame, types: np.ndarray,
                      min_frac: float, min_prev: float
                      ) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    """Fit per-sample activities; see the module docstring for the caveats.

    NNLS onto the full dictionary, then a per-tumour-type dictionary admitting
    signatures that clear `min_frac` of burden in at least `min_prev` of that
    type's samples, then a refit inside it.  The per-type step is what removes
    single-sample fit artefacts, which a per-sample threshold cannot: a
    signature spuriously fitted in one genome survives any per-sample rule.
    """
    A = W.values
    Y = V.values.astype(float).T
    sig_names = np.array(W.columns)

    dense = np.array([nnls(A, Y[:, j])[0] for j in range(Y.shape[1])])
    dense_frac = dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-9)

    admitted: dict[str, np.ndarray] = {}
    for ct in np.unique(types):
        prevalence = (dense_frac[types == ct] >= min_frac).mean(axis=0)
        admitted[ct] = np.where(prevalence >= min_prev)[0]

    H = np.zeros_like(dense)
    for j in range(Y.shape[1]):
        k = admitted[types[j]]
        if len(k) == 0:
            k = np.array([int(dense[j].argmax())])
        h, _ = nnls(A[:, k], Y[:, j])
        keep = k[h >= min_frac * max(h.sum(), 1e-9)]
        if len(keep) == 0:
            keep = k[[int(h.argmax())]]
        h2, _ = nnls(A[:, keep], Y[:, j])
        H[j, keep] = h2

    H = pd.DataFrame(H, index=V.index, columns=W.columns)
    return H, {ct: list(sig_names[i]) for ct, i in admitted.items()}, (dense > 0).sum(1)


# ── Generation ─────────────────────────────────────────────────────────────────
def build_catalogues(W: pd.DataFrame, H: pd.DataFrame, seed: int,
                     noise_model: str = "multiplicative"
                     ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean and noisy catalogues, 96 channels x n samples, integer counts.

    multiplicative (default) is the Diaz-Gay rule, identical to
    make_breast_synthetic.build_catalogues: the clean set is the rounded
    product and every noise level floors the clean counts scaled by the *same*
    standard normal draw, so the levels are nested rather than independent.
    Its relative magnitude is flat in depth, 5% and 10% whether a channel holds
    12 counts or 12,000.

    counting replaces that with the noise a real genome actually carries.
    noise5 becomes a Poisson draw at the sample's own burden, noise10 a
    multinomial draw at a quarter of it.  Relative deviation then falls as
    1/sqrt(count), which is the opposite profile: measured on this cohort,
    0.246 in channels holding 10-30 counts against 0.022 above 1000, where the
    multiplicative model sits flat at 0.10.  The file names stay in the
    Diaz-Gay layout so both models drop into the same harness; the manifest
    records which one produced them.
    """
    expected = pd.DataFrame(W.values @ H[W.columns].values.T,
                            index=W.index, columns=H.index)
    clean = expected.round().astype(np.int64)
    rng = np.random.default_rng(seed)
    catalogues = {"clean": clean}

    if noise_model == "multiplicative":
        Z = rng.standard_normal(clean.shape)
        for name, sigma in NOISE_LEVELS.items():
            noisy = np.floor(clean.values * (1.0 + sigma * Z))
            catalogues[name] = pd.DataFrame(
                np.clip(noisy, 0, None).astype(np.int64),
                index=clean.index, columns=clean.columns,
            )
        return catalogues, expected

    E = np.maximum(expected.values, 0.0)
    catalogues["noise5"] = pd.DataFrame(
        rng.poisson(E).astype(np.int64), index=clean.index, columns=clean.columns
    )
    totals = E.sum(axis=0)
    p = E / np.maximum(totals, 1e-9)
    depth = np.maximum((totals * COUNTING_DEPTH).round().astype(int), 1)
    catalogues["noise10"] = pd.DataFrame(
        np.column_stack([rng.multinomial(depth[j], p[:, j])
                         for j in range(E.shape[1])]).astype(np.int64),
        index=clean.index, columns=clean.columns,
    )
    return catalogues, expected


# ── Verification ───────────────────────────────────────────────────────────────
def verify(catalogues: dict[str, pd.DataFrame], expected: pd.DataFrame,
           H: pd.DataFrame, cos: np.ndarray) -> dict:
    """Print the construction checks and the fidelity of the activities."""
    clean = catalogues["clean"].values.astype(float)
    E = expected.values

    print("\n── construction checks ──")
    ratio = clean.sum(axis=0) / np.maximum(H.sum(axis=1).values, 1)
    print(f"  clean == round(W @ H)       : {np.array_equal(clean, np.round(E))}")
    print(f"  burden vs activity row sums : median ratio {np.median(ratio):.6f}")
    print(f"  H reconstructs real PCAWG   : cosine median {np.median(cos):.5f}"
          f"  p05 {np.percentile(cos, 5):.4f}  min {cos.min():.4f}")
    print(f"  samples below cosine 0.90   : {int((cos < 0.90).sum())} of "
          f"{len(cos)}, flagged as fit_cosine in sample_metadata.csv")

    for name, cat in catalogues.items():
        obs = cat.values.astype(float)
        mu = np.maximum(E, 1e-9)
        chi2 = float(np.mean((obs - E) ** 2 / mu))
        norms = np.linalg.norm(obs, axis=0) * np.linalg.norm(E, axis=0)
        c = np.divide((obs * E).sum(0), norms, out=np.full(obs.shape[1], np.nan),
                      where=norms > 0)
        ref = DG_REFERENCE[name]
        print(f"\n  [{name}]  chi2/df {chi2:7.3f} (Diaz-Gay {ref['chi2_df']:.3f})"
              f"   cosine to truth {np.nanmedian(c):.4f}"
              f" (Diaz-Gay {ref['cosine']:.4f})")
        if name == "clean":
            continue
        flat0, flat = clean.ravel(), obs.ravel()
        edges = [10, 30, 100, 300, 1000, 10 ** 9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (flat0 >= lo) & (flat0 < hi)
            if m.sum() < 200:
                continue
            rel = (flat[m] - flat0[m]) / flat0[m]
            poisson = 1.0 / np.sqrt(flat0[m].mean())
            print(f"      counts {lo:>5}-{hi:<8} n={m.sum():>7} "
                  f" sd(rel)={rel.std():.4f}  target {ref['sd_rel']:.3f}"
                  f"   Poisson would be {poisson:.4f}")
    return {"fit_cosine_median": float(np.median(cos)),
            "fit_cosine_p05": float(np.percentile(cos, 5)),
            "samples_below_cosine_090": int((cos < 0.90).sum())}


# ── Output ─────────────────────────────────────────────────────────────────────
def write_outputs(out_dir: Path, catalogues: dict[str, pd.DataFrame],
                  H: pd.DataFrame, W: pd.DataFrame, dict_name: str,
                  meta: pd.DataFrame, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {"clean": "Samples.txt", "noise5": "Samples_noise5.txt",
                 "noise10": "Samples_noise10.txt"}
    for name, cat in catalogues.items():
        cat.index.name = "Mutation Types"
        cat.to_csv(out_dir / filenames[name], sep="\t")

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
        json.dumps(manifest, indent=2) + "\n"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"output directory (default: {DEFAULT_OUT})")
    p.add_argument("--activities", choices=["published", "fitted"],
                   default="published",
                   help="published consortium attributions (default) or a "
                        "local NNLS fit; see the module docstring")
    p.add_argument("--dictionary", type=Path, default=None,
                   help="signature dictionary; defaults to the published "
                        "sigProfiler definitions, or COSMIC v3.3 when "
                        "--activities fitted")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument("--noise-model", choices=["multiplicative", "counting"],
                   default="multiplicative",
                   help="multiplicative reproduces the Diaz-Gay rule (default); "
                        "counting uses Poisson at full burden and a multinomial "
                        "draw at 25%% depth")
    p.add_argument("--min-frac", type=float, default=0.03,
                   help="fitted mode: minimum share of a sample's burden for a "
                        "signature to be kept (default: 0.03)")
    p.add_argument("--min-prevalence", type=float, default=0.10,
                   help="fitted mode: share of a tumour type's samples in which "
                        "a signature must clear --min-frac (default: 0.10)")
    p.add_argument("--min-burden", type=int, default=0,
                   help="drop real samples below this many mutations "
                        "(default: 0, keep all)")
    p.add_argument("--sample-prefix", default=SAMPLE_PREFIX,
                   help="prefix for synthetic sample names; pass '' for bare IDs")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the construction checks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    published = args.activities == "published"

    if args.dictionary is not None:
        dict_path = args.dictionary
        W = (load_published_dictionary(dict_path)
             if dict_path.suffix == ".csv" else load_cosmic_dictionary(dict_path))
    elif published:
        dict_path = PUBLISHED_DICT
        W = load_published_dictionary(dict_path)
    else:
        dict_path = FITTED_DICT
        W = load_cosmic_dictionary(dict_path)

    dict_name = dict_path.with_suffix(".txt").name
    channels = list(W.index)
    V, meta = load_catalogue(channels)
    print(f"activities: {args.activities}")
    print(f"dictionary: {dict_path.name}, {W.shape[1]} signatures")

    if args.min_burden > 0:
        keep = V.sum(axis=1) >= args.min_burden
        print(f"burden filter >= {args.min_burden}: dropping "
              f"{int((~keep).sum())} of {len(V)} samples")
        V, meta = V[keep], meta[keep]

    admitted: dict[str, list[str]] = {}
    dense_n = np.zeros(len(V), dtype=int)

    if published:
        H, accuracy = load_published_activities(PUBLISHED_ACTIVITIES, V.index)
        cos = fit_cosine(V, W[[c for c in W.columns if c in H.columns]],
                         H[[c for c in H.columns if c in W.columns]])
        drift = float(np.abs(cos - accuracy.values).max())
        median_drift = float(np.median(np.abs(cos - accuracy.values)))
        print(f"provenance: recomputed cosine vs published Accuracy column, "
              f"median |diff| {median_drift:.2e}, max {drift:.2e}")
        if median_drift > ACCURACY_TOL:
            raise SystemExit(
                "recomputed accuracy does not match the published column; the "
                "dictionary and the attributions are not the pair used upstream"
            )
        used_n = int((H.sum(axis=0) > 0).sum())
        if used_n != PUBLISHED_USED:
            raise SystemExit(f"expected {PUBLISHED_USED} used signatures, "
                             f"got {used_n}")
        H = H.reindex(columns=W.columns).fillna(0.0)
    else:
        H, admitted, dense_n = derive_activities(
            V, W, meta.cancer_type.values, args.min_frac, args.min_prevalence
        )
        cos = fit_cosine(V, W, H)
        per_type = pd.Series({ct: len(v) for ct, v in admitted.items()})
        print(f"  type dictionaries: {per_type.min()}-{per_type.max()} signatures "
              f"(mean {per_type.mean():.1f})")

    used = [s for s in W.columns if (H[s] > 0).any()]
    print(f"activities: {(H > 0).sum(axis=1).mean():.2f} signatures/sample, "
          f"{len(used)} of {W.shape[1]} signatures ever used")

    names = [f"{args.sample_prefix}{ct}::{sid}"
             for ct, sid in zip(meta.cancer_type, meta.sample_id)]
    H.index = pd.Index(names, name="sample")

    catalogues, expected = build_catalogues(W, H, args.seed, args.noise_model)

    stats = {"fit_cosine_median": float(np.median(cos)),
             "fit_cosine_p05": float(np.percentile(cos, 5)),
             "samples_below_cosine_090": int((cos < 0.90).sum())}
    if not args.no_verify:
        stats = verify(catalogues, expected, H, cos)

    clean = catalogues["clean"]
    burden = clean.sum(axis=0)
    out_meta = pd.DataFrame({
        "sample": names,
        "source_column": V.index,
        "cancer_type": meta.cancer_type.values,
        "real_burden": V.sum(axis=1).values,
        "activity_burden": H.sum(axis=1).values,
        "clean_burden": burden.values,
        "n_signatures": (H > 0).sum(axis=1).values,
        "fit_cosine": cos,
    }).set_index("sample")
    if not published:
        out_meta["n_signatures_dense"] = dense_n
    out_meta["rounding_loss"] = (
        1.0 - out_meta.clean_burden / out_meta.activity_burden.clip(lower=1e-9)
    )

    manifest = {
        "construction": "clean = round(W @ H)",
        "noise_model": args.noise_model,
        "shared_normals": args.noise_model == "multiplicative",
        "noise_levels": (NOISE_LEVELS if args.noise_model == "multiplicative"
                         else {"noise5": "Poisson at full burden",
                               "noise10": f"multinomial at {COUNTING_DEPTH} depth"}),
        "seed": args.seed,
        "n_samples": int(H.shape[0]),
        "n_cancer_types": int(meta.cancer_type.nunique()),
        "n_signatures_used": len(used),
        "signatures": used,
        "activities": {
            "source": ("published: Alexandrov et al. 2020 PCAWG SigProfiler "
                       "attributions" if published
                       else "fitted: NNLS -> per-type prevalence filter -> refit"),
            "min_burden": args.min_burden,
            "signatures_per_sample_mean": float((H > 0).sum(axis=1).mean()),
            **({} if published else {"min_frac": args.min_frac,
                                     "min_prevalence": args.min_prevalence}),
            **stats,
        },
        "inputs": {
            "catalogue": str(CATALOGUE_FILE),
            "metadata": str(METADATA_FILE),
            "dictionary": str(dict_path),
            **({"activities": str(PUBLISHED_ACTIVITIES)} if published else {}),
        },
        **({"type_dictionaries": dict(sorted(admitted.items()))}
           if admitted else {}),
    }
    write_outputs(args.out_dir, catalogues, H, W, dict_name, out_meta, manifest)

    n_bad = int((out_meta.rounding_loss > 0.10).sum())
    print(f"\nwrote {H.shape[0]} samples x {len(channels)} channels "
          f"to {args.out_dir}/")
    print(f"  signatures      : {len(used)} true, "
          f"{W.shape[1] - len(used)} distractors in the full ground truth")
    print(f"  burden          : min {burden.min()}  median {int(burden.median())}  "
          f"max {burden.max()}")
    print(f"  signatures/sample: mean {(H > 0).sum(axis=1).mean():.2f} "
          f"(range {(H > 0).sum(axis=1).min()}-{(H > 0).sum(axis=1).max()})")
    if n_bad:
        print(f"  rounding        : {n_bad} samples lose >10% of their mutations, "
              "all low burden; see rounding_loss in sample_metadata.csv")


if __name__ == "__main__":
    main()
