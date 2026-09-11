"""
COSMIC v3.6 synthetic SBS benchmark in the Diaz-Gay layout: 12 PCAWG types,
300 tumours each, 3600 catalogues, fitted against the 101-signature panel.

What it is
----------
The same construction as make_diazgay_replica.py, which reproduces the
Diaz-Gay et al. 2023 SBS benchmark to the third decimal (see
PCAWG_Benchmark/README.md), with three changes:

    dictionary   COSMIC v3.6 (96 x 101, tests/data/COSMIC_v3.6_SBS_GRCh37.txt)
                 instead of the 2019 SigProfiler definitions.  W @ H is formed
                 with the v3.6 profiles and the panel handed to the fitters is
                 the same file, so generation and fitting share one
                 dictionary, as in every other synthetic set here.
    types        the nine Diaz-Gay types plus the three largest PCAWG cohorts
                 not among them: Liver-HCC (326 genomes), Prost-AdenoCA (286)
                 and Panc-AdenoCA (241).  Chosen by size because calibration
                 needs genomes; Cervix-AdenoCA, with two, is already the weak
                 point of the shipped set.
    size         12 x 300 = 3600 tumours against Diaz-Gay's 9 x 300 = 2700.

Everything else is SynSigGen's recipe on the PCAWG published attribution:

    per cancer type and signature
        prob   = fraction of that type's real genomes carrying the signature
        mean   = mean(log10 exposure) over those genomes
        stdev  = sd(log10 exposure) over those genomes
        signatures carried by fewer than two genomes of the type are dropped
    per synthetic tumour
        active ~ Bernoulli(prob), independently per signature
        e      = 10 ** Normal(mean, stdev) if active, else 0
    catalogue
        clean    = round(W @ H)
        noise-x% = floor(clean * (1 + (x/100) * Z)), one Z shared across levels

Where the v3.6 names come from
------------------------------
The PCAWG attribution is in COSMIC v3 names.  Every one of its 65 signatures
still exists in v3.6 under the same name except SBS22 and SBS40, which v3.4
split into SBS22a/b(/c) and SBS40a/b/c.  The script maps SBS22 -> SBS22a and
SBS40 -> SBS40a and checks the mapping against the profiles rather than
assuming it: for each attributed signature it computes the cosine between
its 2019 profile and every v3.6 column, and requires the declared target to
be the best match (SBS22a at 1.000, SBS40a at 0.913, the next candidates
SBS22b 0.842 and SBS40b 0.880).  Same-name signatures whose v3.6 profile was
revised are reported (SBS41 at 0.77 is the only one below 0.95).  Since W and
the fitting panel are both v3.6, a revised profile does not create a
generation/fitting mismatch; it only means the calibrated prevalence of that
name now refers to the revised profile.

What this changes for a fitter
------------------------------
The twelve types activate 31 signatures, so the full ground truth carries 70
distractors against the replica's 44.  Several of them are near-copies of
true signatures that did not exist in the 2019 panel: SBS40b and SBS40c
(cosine 0.88 and 0.86 to the old SBS40), SBS22b (0.84 to SBS22a), SBS10c and
SBS10d next to SBS10a/b, and the v3.4+ additions SBS86-SBS113.  With 101
columns on 96 channels the panel is rank-deficient, which is what
tests/test_cosmic_data.py exercises.

Outputs, in the Diaz-Gay layout
-------------------------------
    Samples.txt / Samples_noise5.txt / Samples_noise10.txt
    ground.truth.syn.exposures.csv        the 31 active signatures
    ground.truth.syn.exposures.full.csv   padded to all 101, distractors
    COSMIC_v3.6_SBS_GRCh37.txt            the panel, COSMIC channel order
    sample_metadata.csv, calibration.csv, signature_map.csv,
    generation_manifest.json

Usage
-----
    python make_diazgay_v36.py                                  # seed 0
    python make_diazgay_v36.py --seed 1 --out-dir PCAWG_Benchmark/diazgay-v36-seed1
    python make_diazgay_v36.py --types <the nine Diaz-Gay types> --per-type 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_diazgay_replica import (
    DG_GROUND_TRUTH,
    DG_MIN_TUMOURS,
    DG_TYPES,
    _type_sig_stats,
    calibrate,
    draw_activities,
)
from make_pcawg_parametric import load_published
from make_pcawg_synthetic import (
    DG_REFERENCE,
    PUBLISHED_DICT,
    SAMPLE_PREFIX,
    build_catalogues,
    load_cosmic_dictionary,
    load_published_dictionary,
)

# ── Layout ─────────────────────────────────────────────────────────────────────
EXTRA_TYPES = ["Liver-HCC", "Prost-AdenoCA", "Panc-AdenoCA"]
TYPES = DG_TYPES + EXTRA_TYPES
PER_TYPE = 300

V36_DICT = Path("tests/data/COSMIC_v3.6_SBS_GRCh37.txt")
DICT_NAME = "COSMIC_v3.6_SBS_GRCh37.txt"
DEFAULT_OUT = Path("PCAWG_Benchmark/diazgay-v36")

# v3 attribution name -> v3.6 panel name, for the two signatures v3.4 split.
RENAMES = {"SBS22": "SBS22a", "SBS40": "SBS40a"}
REVISED_WARN = 0.95  # same-name profiles below this cosine are reported


# ── v3 -> v3.6 signature map, checked against the profiles ─────────────────────
def map_signatures(
    W_old: pd.DataFrame, W_new: pd.DataFrame, used: list[str]
) -> pd.DataFrame:
    """One row per attributed signature: its v3.6 name and the evidence.

    The declared target must be the best-matching v3.6 column.  Same-name
    signatures are accepted whatever their cosine, since W and the panel are
    the same v3.6 file, but low values are printed so a revised profile is
    visible.
    """
    common = W_old.index.intersection(W_new.index)
    A = W_old.loc[common].values
    B = W_new.loc[common].values
    C = (A.T @ B) / np.outer(np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0))
    C = pd.DataFrame(C, index=W_old.columns, columns=W_new.columns)

    rows = []
    for s in used:
        target = RENAMES.get(s, s)
        if target not in W_new.columns:
            raise SystemExit(f"{s} -> {target}: not in the v3.6 panel")
        best = C.loc[s].idxmax()
        row = {
            "v3": s,
            "v36": target,
            "cosine": float(C.loc[s, target]),
            "best_match": best,
            "best_cosine": float(C.loc[s, best]),
        }
        if s in RENAMES and best != target:
            raise SystemExit(
                f"{s} -> {target} is not the best v3.6 match ({best},"
                f" {row['best_cosine']:.3f} vs {row['cosine']:.3f})"
            )
        rows.append(row)
    m = pd.DataFrame(rows)

    print("\n── v3 -> v3.6 signature map ──")
    for r in m.itertuples():
        if r.v3 != r.v36:
            alt = C.loc[r.v3, [c for c in W_new.columns if c.startswith(r.v3)]]
            print(
                f"  {r.v3:7s} -> {r.v36:7s} cosine {r.cosine:.3f}   alternatives"
                f" {', '.join(f'{k} {v:.3f}' for k, v in alt.drop(r.v36).items())}"
            )
        elif r.cosine < REVISED_WARN:
            print(
                f"  {r.v3:7s} kept, profile revised: cosine {r.cosine:.3f}"
                f" (closest other column {r.best_match} {r.best_cosine:.3f})"
            )
    n_same = int((m.v3 == m.v36).sum())
    print(f"  {n_same} same-name, {len(m) - n_same} renamed, min same-name cosine"
          f" {m[m.v3 == m.v36].cosine.min():.3f}")
    return m


# ── Verification ───────────────────────────────────────────────────────────────
def verify(
    H: pd.DataFrame,
    kinds: list[str],
    params: pd.DataFrame,
    catalogues: dict,
    expected: pd.DataFrame,
    cancer_types: list[str],
) -> dict:
    kinds = np.array(kinds)
    ours = _type_sig_stats(H, kinds, cancer_types)
    report: dict = {}

    print("\n── activities against the calibration targets ──")
    tgt = params.set_index(["cancer_type", "signature"])
    j = ours.join(tgt, rsuffix="_target", how="inner")
    for col in ("prob", "mean_log10", "sd_log10"):
        d = (j[col] - j[f"{col}_target"]).abs()
        print(
            f"  {col:10s} corr {j[col].corr(j[f'{col}_target']):.3f}"
            f"   mean |diff| {d.mean():.3f}   max {d.max():.3f}"
        )
        report[f"{col}_max_abs_diff_vs_target"] = float(d.max())

    n_per = (H > 0).sum(axis=1)
    burden = catalogues["clean"].sum(axis=0).values
    q = [0.1, 0.5, 0.9]
    print(f"  signatures per sample         : {n_per.mean():.2f}"
          f" (range {n_per.min()}-{n_per.max()})")
    print(f"  burden quantiles 10/50/90     : {np.quantile(burden, q).round(0)}")
    frac = H.div(H.sum(axis=1), axis=0)
    small = float(((frac > 0) & (frac < 0.01)).sum().sum() / (H > 0).sum().sum())
    print(f"  true exposures below 1% burden: {small:.3f}")
    report["signatures_per_sample"] = float(n_per.mean())
    report["frac_true_exposures_below_1pct"] = small

    shared = [t for t in cancer_types if t in DG_TYPES]
    if DG_GROUND_TRUTH.exists() and shared:
        dg = pd.read_csv(DG_GROUND_TRUTH, index_col=0).T.rename(columns=RENAMES)
        dg_kinds = np.array(
            [c.split("::")[0].replace(SAMPLE_PREFIX, "") for c in dg.index]
        )
        theirs = _type_sig_stats(
            dg.reindex(columns=H.columns).fillna(0.0), dg_kinds, shared
        )
        j = ours.join(theirs, rsuffix="_dg", how="inner")
        j = j[(j.prob > 0) | (j.prob_dg > 0)]
        print(f"\n── the {len(shared)} shared types against the shipped Diaz-Gay"
              " ground truth ──")
        for col in ("prob", "mean_log10", "sd_log10"):
            d = (j[col] - j[f"{col}_dg"]).abs().dropna()
            c = j[col].corr(j[f"{col}_dg"])
            print(f"  {col:10s} corr {c:.3f}   mean |diff| {d.mean():.3f}")
            report[f"{col}_corr_vs_diazgay_shared_types"] = float(c)
        extra = set(j.index[(j.prob > 0) & (j.prob_dg == 0)].get_level_values(1))
        print(f"  signatures active here but absent from Diaz-Gay on those types:"
              f" {sorted(extra) or 'none'}")

    print("\n── construction checks ──")
    clean = catalogues["clean"].values.astype(float)
    E = expected.values
    same = bool(np.array_equal(clean, np.round(E)))
    print(f"  clean == round(W @ H)       : {same}")
    report["clean_is_rounded_product"] = same
    for name, cat in catalogues.items():
        obs = cat.values.astype(float)
        chi2 = float(np.mean((obs - E) ** 2 / np.maximum(E, 1e-9)))
        norms = np.linalg.norm(obs, axis=0) * np.linalg.norm(E, axis=0)
        c = np.divide(
            (obs * E).sum(0), norms, out=np.full(obs.shape[1], np.nan), where=norms > 0
        )
        ref = DG_REFERENCE[name]
        print(
            f"  [{name:7s}] chi2/df {chi2:6.3f} (Diaz-Gay {ref['chi2_df']:.3f})"
            f"   cosine to truth {np.nanmedian(c):.4f} (Diaz-Gay {ref['cosine']:.4f})"
        )
        report[f"{name}_chi2_df"] = chi2
    return report


# ── Output ─────────────────────────────────────────────────────────────────────
def write_outputs(
    out_dir: Path,
    catalogues: dict,
    H: pd.DataFrame,
    W: pd.DataFrame,
    kinds: list[str],
    params: pd.DataFrame,
    sig_map: pd.DataFrame,
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "clean": "Samples.txt",
        "noise5": "Samples_noise5.txt",
        "noise10": "Samples_noise10.txt",
    }
    for name, cat in catalogues.items():
        cat.index.name = "Mutation Types"
        cat.to_csv(out_dir / files[name], sep="\t")

    used = [s for s in H.columns if (H[s] > 0).any()]
    gt = H[used].T
    gt.index.name = None
    gt.to_csv(out_dir / "ground.truth.syn.exposures.csv")
    full = H.T.reindex(W.columns).fillna(0.0)
    full.index.name = None
    full.to_csv(out_dir / "ground.truth.syn.exposures.full.csv")

    W.to_csv(out_dir / DICT_NAME, sep="\t")
    params.to_csv(out_dir / "calibration.csv", index=False)
    sig_map.to_csv(out_dir / "signature_map.csv", index=False)

    clean_burden = catalogues["clean"].sum(axis=0)
    act_burden = H.sum(axis=1)
    meta = pd.DataFrame(
        {
            "sample": H.index,
            "cancer_type": kinds,
            "activity_burden": act_burden.values,
            "clean_burden": clean_burden.values,
            "n_signatures": (H > 0).sum(axis=1).values,
            "rounding_loss": 1.0
            - clean_burden.values / np.maximum(act_burden.values, 1e-9),
        }
    )
    meta.to_csv(out_dir / "sample_metadata.csv", index=False)
    (out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--types",
        nargs="+",
        default=TYPES,
        help="PCAWG cancer types to draw (default: the nine Diaz-Gay types"
        f" plus {', '.join(EXTRA_TYPES)})",
    )
    p.add_argument("--per-type", type=int, default=PER_TYPE,
                   help=f"synthetic tumours per type (default {PER_TYPE})")
    p.add_argument("--dictionary", type=Path, default=V36_DICT,
                   help=f"COSMIC panel, tab-separated (default {V36_DICT})")
    p.add_argument("--round-exposures", action="store_true",
                   help="round drawn exposures to integers; Diaz-Gay's are not")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    W = load_cosmic_dictionary(args.dictionary)
    W_old = load_published_dictionary(PUBLISHED_DICT)
    H_real, types = load_published()
    unknown = [t for t in args.types if t not in set(types)]
    if unknown:
        raise SystemExit(f"not PCAWG cancer types: {unknown}")
    print(f"dictionary: {args.dictionary} ({W.shape[1]} signatures)")
    print(f"types     : {len(args.types)} x {args.per_type} = "
          f"{len(args.types) * args.per_type} tumours")

    params = calibrate(H_real, types, args.types)
    used_v3 = sorted(params.signature.unique(), key=lambda s: (len(s), s))
    print(
        f"calibration: {len(used_v3)} signatures active in >= {DG_MIN_TUMOURS}"
        f" genomes of a type: {used_v3}"
    )
    print(params.groupby("cancer_type").size().rename("signatures").to_string())

    sig_map = map_signatures(W_old, W, used_v3)
    rename = dict(zip(sig_map.v3, sig_map.v36))
    params["signature_v3"] = params.signature
    params["signature"] = params.signature.map(rename)
    sigs = [rename[s] for s in used_v3]

    rng = np.random.default_rng(args.seed)
    H, kinds = draw_activities(
        params, sigs, rng, args.per_type, args.round_exposures, args.types
    )
    H = H.reindex(columns=W.columns).fillna(0.0)  # zero rows for the distractors

    catalogues, expected = build_catalogues(W, H, args.seed, "multiplicative")
    checks = verify(H, kinds, params, catalogues, expected, args.types)

    manifest = {
        "source": "SynSigGen recipe (Islam et al. 2022 / Diaz-Gay et al. 2023)"
        " on the PCAWG SigProfiler attribution, fitted against COSMIC v3.6",
        "activities": "drawn",
        "construction": "H: Bernoulli(prob) x 10**Normal(mean_log10, sd_log10) per"
        " type and signature; clean = round(W @ H); noise = floor(clean *"
        " (1 + sigma * Z)), shared Z",
        "dictionary": DICT_NAME,
        "dictionary_source": str(args.dictionary),
        "seed": args.seed,
        "cancer_types": args.types,
        "per_type": args.per_type,
        "n_samples": int(len(H)),
        "n_signatures_used": len(sigs),
        "n_distractors": int(W.shape[1] - len(sigs)),
        "signatures": sigs,
        "signature_map": sig_map.to_dict(orient="records"),
        "min_tumours_per_signature": DG_MIN_TUMOURS,
        "round_exposures": bool(args.round_exposures),
        "checks": checks,
    }
    write_outputs(args.out_dir, catalogues, H, W, kinds, params, sig_map, manifest)
    print(f"\nwrote {len(H)} samples x {len(sigs)} signatures"
          f" ({W.shape[1] - len(sigs)} distractors) to {args.out_dir}/")


if __name__ == "__main__":
    main()
