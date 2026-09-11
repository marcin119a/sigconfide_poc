"""
Replica of the Diaz-Gay et al. 2023 synthetic SBS benchmark, rebuilt from the
PCAWG published attributions in PCAWG_Benchmark/published/.

Where the Diaz-Gay set comes from
---------------------------------
The 2700-sample SBS set shipped with Diaz-Gay et al. 2023 (Bioinformatics,
btad756) is the "realistic" synthetic dataset of Islam et al. 2022 (Cell
Genomics, SigProfilerExtractor): 300 synthetic tumours for each of nine PCAWG
cancer types, generated with SynSigGen (Rozen lab, github.com/steverozen/
SynSigGen) from the PCAWG SigProfiler attributions, which are the same
`PCAWG_sigProfiler_SBS_signatures_in_samples.csv` this repository already
uses as ground truth for the PCAWG synthetic arm.  The sample prefix
`SP.Syn.` is SynSigGen's marker for "drawn from the SigProfiler (SP)
attribution".

Measured on the shipped files, the construction is:

    per cancer type and signature (SynSigGen::GetSynSigParamsFromExposures)
        prob   = fraction of that type's real genomes with the signature
        mean   = mean(log10 exposure) over those genomes
        stdev  = sd(log10 exposure) over those genomes
        signatures active in fewer than two genomes of the type are dropped

    per synthetic tumour (SynSigGen::GenerateSyntheticExposures)
        active_s ~ Bernoulli(prob_s)            independently per signature
        e_s      = 10 ** Normal(mean_s, stdev_s) if active, else 0

    catalogue (the Diaz-Gay rule, shared with the other synthetic sets here)
        clean    = round(W @ H)
        noise-x% = floor(clean * (1 + (x/100) * Z)),  one Z shared by levels

Three facts pin this down against the shipped ground truth.  The nine types
and 300-per-type layout match.  Per type and signature, log10 mean and sd of
the shipped exposures agree with the PCAWG attribution to 0.03 and 0.02.  And
signatures are drawn independently: in the shipped breast samples SBS2 and
SBS13 co-occur at exactly the product of their prevalences (0.76), where the
real PCAWG breast genomes co-carry them well above independence (0.89 against
0.80).  The 21-signature set is exactly the set of signatures active in at
least two genomes of one of the nine types.

Why rebuild it
--------------
Scored on the PCAWG synthetic arm (real per-sample H, 37 types, 54
signatures) SigProfilerAssignment beats sigconfide by 0.02 to 0.07 F1, while
on the Diaz-Gay set sigconfide leads.  Two things differ between those sets:
the ground truth of the PCAWG arm *is* SigProfiler's own output, and the arm
carries 33 more signatures and 28 more types.  This script separates them.

    --activities drawn   the replica: 9 types x 300, 21 signatures, H drawn
                         from the calibrated distributions.  No tool produced
                         this H.
    --activities real    the same 9 types and 21 signatures, but H is the real
                         per-sample PCAWG attribution (741 genomes).  This is
                         SigProfiler's output.

Same dictionary, same construction, same scoring.  If SPA's lead survives on
the drawn set, it is not provenance.  If it appears only on the real set, it
is.

Dictionary
----------
Both arms use the SigProfiler 2019-05-22 definitions the attributions were
computed with (COSMIC v3), as the PCAWG synthetic arm does.  The file is
written under a `COSMIC_*` name so benchmark_isbs.py finds it.  The Diaz-Gay
benchmark itself fits with COSMIC v3.3; pass --dictionary cosmic33 to
reproduce that choice.

Outputs, in the Diaz-Gay layout
-------------------------------
    Samples.txt / Samples_noise5.txt / Samples_noise10.txt
    ground.truth.syn.exposures.csv        the 21 active signatures
    ground.truth.syn.exposures.full.csv   padded with zero rows, distractors
    COSMIC_<dictionary>.txt
    sample_metadata.csv, generation_manifest.json, calibration.csv

Usage
-----
    python make_diazgay_replica.py                      # drawn, seed 0
    python make_diazgay_replica.py --activities real
    python make_diazgay_replica.py --seed 1 --out-dir PCAWG_Benchmark/replica-seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_pcawg_parametric import load_published
from make_pcawg_synthetic import (
    DG_REFERENCE,
    FITTED_DICT,
    PUBLISHED_DICT,
    SAMPLE_PREFIX,
    build_catalogues,
    load_cosmic_dictionary,
    load_published_dictionary,
)

# ── The Diaz-Gay / Islam 2022 layout ──────────────────────────────────────────
DG_TYPES = [
    "Bladder-TCC",
    "Eso-AdenoCA",
    "Breast-AdenoCA",
    "Lung-SCC",
    "Kidney-RCC",
    "Ovary-AdenoCA",
    "Bone-Osteosarc",
    "Cervix-AdenoCA",
    "Stomach-AdenoCA",
]
DG_PER_TYPE = 300
DG_MIN_TUMOURS = 2  # SynSigGen drops signatures seen in < 2 tumours
DG_GROUND_TRUTH = Path(
    "Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/ground.truth.syn.exposures.csv"
)
DICTIONARIES = {
    "published": (PUBLISHED_DICT, "COSMIC_v3_sigProfiler_2019_05_22_SBS_GRCh37.txt"),
    "cosmic33": (FITTED_DICT, "COSMIC_v3.3_SBS_GRCh37.txt"),
}
DEFAULT_OUT = {
    "drawn": Path("PCAWG_Benchmark/diazgay-replica"),
    "real": Path("PCAWG_Benchmark/diazgay-realH"),
}


# ── Calibration ────────────────────────────────────────────────────────────────
def calibrate(
    H: pd.DataFrame, types: pd.Series, cancer_types: list[str] = DG_TYPES
) -> pd.DataFrame:
    """SynSigGen::GetSynSigParamsFromExposures, per cancer type.

    One row per (type, signature) with prob, mean and stdev of log10 exposure
    over the genomes where the signature is active; signatures active in fewer
    than DG_MIN_TUMOURS genomes of the type are dropped, which is what makes
    the signature set come out at 21.
    """
    rows = []
    for ct in cancer_types:
        sub = H[types == ct]
        if sub.empty:
            raise SystemExit(f"no PCAWG genomes of type {ct}")
        for sig in H.columns:
            x = sub[sig].values
            pos = x[x > 0]
            if len(pos) < DG_MIN_TUMOURS:
                continue
            lg = np.log10(pos)
            rows.append(
                {
                    "cancer_type": ct,
                    "signature": sig,
                    "n_genomes": len(sub),
                    "n_active": len(pos),
                    "prob": len(pos) / len(sub),
                    "mean_log10": lg.mean(),
                    "sd_log10": lg.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


# ── Activities ─────────────────────────────────────────────────────────────────
def draw_activities(
    params: pd.DataFrame,
    sigs: list[str],
    rng: np.random.Generator,
    per_type: int,
    round_exposures: bool,
    cancer_types: list[str] = DG_TYPES,
) -> tuple[pd.DataFrame, list[str]]:
    """SynSigGen::GenerateSyntheticExposures for each type in turn.

    Presence is Bernoulli(prob) independently per signature, magnitude is
    10 ** Normal(mean, sd).  A tumour that draws no signature at all is
    redrawn, since it has no catalogue; the shipped set has none.
    """
    blocks, names, kinds = [], [], []
    for ct in cancer_types:
        p = params[params.cancer_type == ct].set_index("signature")
        k = len(p)
        out = np.zeros((per_type, len(sigs)))
        cols = [sigs.index(s) for s in p.index]
        prob, mu, sd = p.prob.values, p.mean_log10.values, p.sd_log10.values
        for i in range(per_type):
            while True:
                active = rng.random(k) < prob
                if active.any():
                    break
            e = np.where(active, 10.0 ** rng.normal(mu, sd), 0.0)
            out[i, cols] = e
        blocks.append(out)
        names += [f"{SAMPLE_PREFIX}{ct}::S.{i + 1}" for i in range(per_type)]
        kinds += [ct] * per_type
    H = pd.DataFrame(
        np.vstack(blocks), index=pd.Index(names, name="sample"), columns=sigs
    )
    if round_exposures:
        H = H.round()
    return H, kinds


def real_activities(
    H: pd.DataFrame, types: pd.Series, sigs: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """The real PCAWG attribution of the nine types, restricted to `sigs`."""
    keep = types.isin(DG_TYPES)
    sub = H.loc[keep, sigs].copy()
    kinds = types[keep].tolist()
    dropped = [s for s in H.columns if s not in sigs and (H.loc[keep, s] > 0).any()]
    lost = int((H.loc[keep, dropped] > 0).sum().sum()) if dropped else 0
    print(
        f"  real H: {len(sub)} genomes; {lost} exposures on {len(dropped)}"
        f" singleton signatures zeroed: {dropped}"
    )
    empty = (sub > 0).sum(axis=1) == 0
    if empty.any():
        print(f"  dropping {int(empty.sum())} genomes left without a signature")
        sub, kinds = sub[~empty], [k for k, e in zip(kinds, empty) if not e]
    sub.index = pd.Index([f"{SAMPLE_PREFIX}{i}" for i in sub.index], name="sample")
    return sub, kinds


# ── Verification ───────────────────────────────────────────────────────────────
def _type_sig_stats(
    H: pd.DataFrame, kinds: np.ndarray, cancer_types: list[str] = DG_TYPES
) -> pd.DataFrame:
    rows = []
    for ct in cancer_types:
        sub = H[kinds == ct]
        for sig in H.columns:
            pos = sub[sig].values[sub[sig].values > 0]
            rows.append(
                {
                    "cancer_type": ct,
                    "signature": sig,
                    "prob": len(pos) / len(sub),
                    "mean_log10": np.log10(pos).mean() if len(pos) else np.nan,
                    "sd_log10": np.log10(pos).std(ddof=1) if len(pos) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows).set_index(["cancer_type", "signature"])


def verify(
    H: pd.DataFrame,
    kinds: list[str],
    params: pd.DataFrame,
    catalogues: dict,
    expected: pd.DataFrame,
) -> dict:
    kinds = np.array(kinds)
    ours = _type_sig_stats(H, kinds)
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

    if DG_GROUND_TRUTH.exists():
        dg = pd.read_csv(DG_GROUND_TRUTH, index_col=0).T
        dg_kinds = np.array(
            [c.split("::")[0].replace(SAMPLE_PREFIX, "") for c in dg.index]
        )
        same = sorted(dg.columns) == sorted(H.columns[(H > 0).any()])
        print(f"\n── against the shipped Diaz-Gay ground truth ({len(dg)} samples) ──")
        print(f"  same 21 signatures            : {same}")
        theirs = _type_sig_stats(dg.reindex(columns=H.columns).fillna(0.0), dg_kinds)
        j = ours.join(theirs, rsuffix="_dg", how="inner")
        j = j[(j.prob > 0) | (j.prob_dg > 0)]
        for col in ("prob", "mean_log10", "sd_log10"):
            d = (j[col] - j[f"{col}_dg"]).abs().dropna()
            print(
                f"  {col:10s} corr {j[col].corr(j[f'{col}_dg']):.3f}"
                f"   mean |diff| {d.mean():.3f}   max {d.max():.3f}"
            )
            report[f"{col}_corr_vs_diazgay"] = float(j[col].corr(j[f"{col}_dg"]))
        n_ours, n_dg = (H > 0).sum(axis=1), (dg > 0).sum(axis=1)
        print(
            f"  signatures per sample         : {n_ours.mean():.2f} ours,"
            f" {n_dg.mean():.2f} Diaz-Gay"
        )
        b_ours = catalogues["clean"].sum(axis=0).values
        b_dg = dg.sum(axis=1).values
        q = [0.1, 0.5, 0.9]
        print(
            f"  burden quantiles 10/50/90     : ours {np.quantile(b_ours, q).round(0)},"
            f" Diaz-Gay {np.quantile(b_dg, q).round(0)}"
        )
        # Independence of signature presence: SBS2/SBS13 in breast.
        for name, M, K in (("ours", H, kinds), ("Diaz-Gay", dg, dg_kinds)):
            b = M[K == "Breast-AdenoCA"]
            p2, p13 = (b["SBS2"] > 0).mean(), (b["SBS13"] > 0).mean()
            both = ((b["SBS2"] > 0) & (b["SBS13"] > 0)).mean()
            print(
                f"  breast SBS2/SBS13 co-active   : {name:8s} {both:.3f}"
                f"  (independence {p2 * p13:.3f})"
            )
        report["same_signature_set_as_diazgay"] = bool(same)

    print("\n── construction checks ──")
    clean = catalogues["clean"].values.astype(float)
    E = expected.values
    print(f"  clean == round(W @ H)       : {np.array_equal(clean, np.round(E))}")
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
    dict_name: str,
    kinds: list[str],
    params: pd.DataFrame,
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

    W.to_csv(out_dir / dict_name, sep="\t")
    params.to_csv(out_dir / "calibration.csv", index=False)

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
        "--activities",
        choices=["drawn", "real"],
        default="drawn",
        help="drawn: SynSigGen-style replica (default); real: the PCAWG"
        " attribution of the same nine types and 21 signatures",
    )
    p.add_argument(
        "--dictionary",
        choices=list(DICTIONARIES),
        default="published",
        help="published: SigProfiler 2019-05-22 / COSMIC v3 (default);"
        " cosmic33: the COSMIC v3.3 file Diaz-Gay fit with",
    )
    p.add_argument(
        "--per-type",
        type=int,
        default=DG_PER_TYPE,
        help=f"synthetic tumours per type (default {DG_PER_TYPE})",
    )
    p.add_argument(
        "--round-exposures",
        action="store_true",
        help="round drawn exposures to integers (SynSigGen >= 1.1.1);"
        " the Diaz-Gay files are unrounded",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or DEFAULT_OUT[args.activities]
    dict_path, dict_name = DICTIONARIES[args.dictionary]
    W = (
        load_published_dictionary(dict_path)
        if args.dictionary == "published"
        else load_cosmic_dictionary(dict_path)
    )
    H_real, types = load_published()
    H_real = H_real[[s for s in H_real.columns if s in W.columns]]
    missing = [s for s in H_real.columns if s not in W.columns]
    if missing:
        raise SystemExit(f"attribution signatures absent from dictionary: {missing}")

    params = calibrate(H_real, types)
    sigs = sorted(params.signature.unique(), key=lambda s: (len(s), s))
    print(
        f"calibration: {len(DG_TYPES)} types, {len(sigs)} signatures active in"
        f" >= {DG_MIN_TUMOURS} genomes of a type: {sigs}"
    )
    print(params.groupby("cancer_type").size().rename("signatures").to_string())

    rng = np.random.default_rng(args.seed)
    if args.activities == "drawn":
        H, kinds = draw_activities(
            params, sigs, rng, args.per_type, args.round_exposures
        )
    else:
        H, kinds = real_activities(H_real, types, sigs)
    H = H.reindex(columns=W.columns).fillna(0.0)  # zero rows for the distractors

    catalogues, expected = build_catalogues(W, H, args.seed, "multiplicative")
    checks = verify(H, kinds, params, catalogues, expected)

    manifest = {
        "source": "Diaz-Gay et al. 2023 SBS benchmark = Islam et al. 2022 realistic"
        " set, SynSigGen on PCAWG SigProfiler attributions",
        "activities": args.activities,
        "construction": "H: Bernoulli(prob) x 10**Normal(mean_log10, sd_log10) per type"
        " and signature; clean = round(W @ H); noise = floor(clean *"
        " (1 + sigma * Z)), shared Z"
        if args.activities == "drawn"
        else "H: PCAWG published attribution restricted to the nine types and"
        " the 21 signatures; clean = round(W @ H); noise = floor(clean *"
        " (1 + sigma * Z)), shared Z",
        "dictionary": dict_name,
        "seed": args.seed,
        "cancer_types": DG_TYPES,
        "per_type": args.per_type if args.activities == "drawn" else None,
        "n_samples": int(len(H)),
        "n_signatures_used": len(sigs),
        "signatures": sigs,
        "min_tumours_per_signature": DG_MIN_TUMOURS,
        "round_exposures": bool(args.round_exposures),
        "checks": checks,
    }
    write_outputs(out_dir, catalogues, H, W, dict_name, kinds, params, manifest)
    print(f"\nwrote {len(H)} samples x {len(sigs)} signatures to {out_dir}/")


if __name__ == "__main__":
    main()
