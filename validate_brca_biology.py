"""
Biological validation of sigconfide on real breast cancers (ICGC-BRCA / 560 WGS).

Unlike the Diaz-Gay benchmark, this cohort has no synthetic ground truth.  The
reference used here is Supplementary Table 21 of Nik-Zainal et al. (2016): the
original authors' own de-novo NNMF assignment of twelve COSMIC v2 signatures
across the very same 560 genomes, sample by sample.  Because it is per-sample
and same-cohort, it constrains not only how often a signature should be called
but in which tumours, which makes paired statistics possible.

Two reference dictionaries are supported, and the choice matters:

  --catalog v2    (default, the primary analysis)
      The 30 COSMIC v2 signatures - the dictionary Table 21 itself was built
      in.  Every reference signature maps one-to-one onto a fitted signature,
      so there are no v2->v3 splits, no cross-dictionary transfer slack, and
      the prevalence comparison becomes an exact paired test (McNemar) rather
      than a band check.  SigMA's shipped Signature 3 calls are also v2 calls,
      so the external comparison is like-for-like too.

  --catalog v3.3  (cross-dictionary robustness check)
      The 78 COSMIC v3.3 signatures.  Table 21's Signature 5 splits into
      SBS5/SBS40 and Signature 17 into SBS17a/SBS17b, so those are only pinned
      as unions, and prevalence is checked against Clopper-Pearson intervals
      widened by an identifiability-scaled slack.

Three further sources of evidence are used in both arms:

  1. TISSUE KNOWLEDGE  - which signatures are known to operate in breast tissue.
     Used both as a tissue-informed panel and, on the unrestricted reference, as
     a specificity check: signatures whose aetiology is incompatible with a
     treatment-naive primary breast tumour (UV, tobacco, aflatoxin, ...) are
     counted as false positives.

  2. SigMA (Gulhan et al. 2019) - a panel-oriented Signature 3 detector whose
     published calls on the panel-downsampled version of this cohort ship with
     the data (out-sigma-brca-panel.tsv).

  3. HRDetect (Davies et al. 2017) - reported only as a data-integrity
     diagnostic: the shipped labels do not line up with these catalogues (see
     hrdetect_integrity), so they are not used as truth anywhere.

Datasets: full WGS catalogues, a panel simulation (~5 SNVs/sample) and a
downsampling ladder (3, 6, 9, 12, 15, 18 mutations per sample), so the same
biology can be probed as a function of mutation burden.

Usage
-----
    python validate_brca_biology.py --stage fit      # run sigconfide + SPA
    python validate_brca_biology.py --stage report   # metrics, figures, LaTeX
    python validate_brca_biology.py --stage all
    python validate_brca_biology.py --catalog v3.3 --stage report
    python validate_brca_biology.py --stage fit --datasets wgs --max-samples 20
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sigconfide.estimates.selection import hybrid_stepwise_selection  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("ICGC-BRCA")
WGS_COUNTS = DATA_DIR / "counts.ICGC-BRCA-EU_BRCA_22.WGS.SBS-96.tsv"
PANEL_COUNTS = DATA_DIR / "panel-BRCA-WGS_counts.tsv"
HRD_FILE = DATA_DIR / "icgc_brca_hrd.tsv"
SIGMA_FILE = DATA_DIR / "out-sigma-brca-panel.tsv"
TABLE21_FILE = DATA_DIR / "Supplementary.Table.21.Signatures.v3.xlsx"
DOWNSAMPLE_DIR = DATA_DIR / "downsampled"
DOWNSAMPLE_LEVELS = [3, 6, 9, 12, 15, 18]
FIG_DIR = Path("SigConfide---Application-note")

# ── sigconfide parameters (identical to the Diaz-Gay benchmark) ────────────────
R = 100
PRE_FILTER = 0.001
MAX_WORKERS = None
N_BOOT = 2000     # paired-bootstrap resamples for the concordance intervals
BOOT_SEED = 0     # fixed so the published intervals are reproducible


# ── Nik-Zainal 2016 Supplementary Table 21 ─────────────────────────────────────
# Sheet B is the authors' own NNMF assignment of 12 de-novo COSMIC v2 signatures
# across the same 560 genomes we analyse.  Non-zero counts out of 560 as
# published; hardcoded so load_table21() can assert the workbook still matches.
TABLE21_NONZERO = {
    "Signature 1": 523, "Signature 2": 459, "Signature 3": 158,
    "Signature 5": 482, "Signature 6": 9, "Signature 8": 368,
    "Signature 13": 426, "Signature 17": 33, "Signature 18": 97,
    "Signature 20": 3, "Signature 26": 10, "Signature 30": 1,
}
TABLE21_N = 560


# ══ COSMIC v2: the dictionary Table 21 was built in ════════════════════════════
# Signatures reported in breast carcinoma by COSMIC v2 / Nik-Zainal 2016.  They
# are exactly the twelve Table 21 uses, which is the point: with the same
# dictionary and the same tissue restriction, the only thing that differs
# between the reference and a fit is the algorithm.
V2_BREAST = [
    "Signature_1",   # clock-like, 5mC deamination
    "Signature_2",   # APOBEC
    "Signature_3",   # HRD / BRCA1-BRCA2 loss
    "Signature_5",   # clock-like
    "Signature_6",   # mismatch-repair deficiency
    "Signature_8",   # unknown aetiology, common in breast
    "Signature_13",  # APOBEC
    "Signature_17",  # unknown, rare in breast
    "Signature_18",  # reactive oxygen species
    "Signature_20",  # mismatch-repair deficiency
    "Signature_26",  # mismatch-repair deficiency
    "Signature_30",  # base-excision repair deficiency (NTHL1)
]

# Aetiologies that cannot operate in a treatment-naive primary breast tumour.
# Restricted to signatures with a hard, documented cause; the merely
# organ-restricted ones of unknown aetiology are left out, so this is a
# conservative count of biologically impossible calls.
V2_IMPLAUSIBLE = {
    "Signature_4": "tobacco smoking (lung, head and neck)",
    "Signature_7": "ultraviolet light (skin)",
    "Signature_9": "polymerase eta, IGHV-mutated lymphoid",
    "Signature_10": "POLE proofreading deficiency",
    "Signature_11": "temozolomide treatment",
    "Signature_22": "aristolochic acid (urothelial, liver)",
    "Signature_24": "aflatoxin (liver)",
    "Signature_25": "Hodgkin lymphoma",
    "Signature_29": "tobacco chewing (oral)",
}

# Table 21 signature -> fitted signature(s).  One-to-one in v2, by construction.
V2_REF_MAP = {v2: [v2.replace(" ", "_")] for v2 in TABLE21_NONZERO}


# ══ COSMIC v3.3: the cross-dictionary arm ══════════════════════════════════════
V33_BREAST = [
    "SBS1",    # clock-like, 5mC deamination
    "SBS2",    # APOBEC
    "SBS3",    # HRD / BRCA1-BRCA2 loss
    "SBS5",    # clock-like
    "SBS8",    # unknown aetiology, common in breast
    "SBS13",   # APOBEC
    "SBS17a",  # rare in breast
    "SBS17b",  # rare in breast
    "SBS18",   # reactive oxygen species
    "SBS30",   # base-excision repair deficiency (NTHL1)
    "SBS40",   # unknown aetiology, correlates with age
]

V33_IMPLAUSIBLE = {
    "SBS4": "tobacco smoking (lung)",
    "SBS7a": "ultraviolet light (skin)",
    "SBS7b": "ultraviolet light (skin)",
    "SBS7c": "ultraviolet light (skin)",
    "SBS7d": "ultraviolet light (skin)",
    "SBS9": "polymerase eta / lymphoid",
    "SBS10a": "POLE proofreading deficiency",
    "SBS10b": "POLE proofreading deficiency",
    "SBS10c": "POLD1 proofreading deficiency",
    "SBS10d": "POLD1 proofreading deficiency",
    "SBS11": "temozolomide treatment",
    "SBS22": "aristolochic acid",
    "SBS24": "aflatoxin",
    "SBS29": "tobacco chewing",
    "SBS31": "platinum chemotherapy",
    "SBS32": "azathioprine treatment",
    "SBS35": "platinum chemotherapy",
    "SBS42": "haloalkane exposure",
    "SBS86": "unknown chemotherapy",
    "SBS87": "thiopurine chemotherapy",
    "SBS88": "colibactin (E. coli, colorectal)",
    "SBS90": "duocarmycin exposure",
    "SBS99": "unknown treatment-related",
}

# v2 -> v3 is not one-to-one: Signature 5 was split into SBS5 + SBS40 and
# Signature 17 into SBS17a + SBS17b, so those two are compared against the
# union of their v3 descendants.
V33_REF_MAP = {
    "Signature 1": ["SBS1"],
    "Signature 2": ["SBS2"],
    "Signature 3": ["SBS3"],
    "Signature 5": ["SBS5", "SBS40"],
    "Signature 6": ["SBS6"],
    "Signature 8": ["SBS8"],
    "Signature 13": ["SBS13"],
    "Signature 17": ["SBS17a", "SBS17b"],
    "Signature 18": ["SBS18"],
    "Signature 20": ["SBS20"],
    "Signature 26": ["SBS26"],
    "Signature 30": ["SBS30"],
}

# Prevalence bands for the v3.3 arm only.  Two sources of uncertainty are kept
# separate:
#
#  * SAMPLING.  Each Table 21 rate gets an exact Clopper-Pearson 95% interval.
#    This is *not* the uncertainty of the comparison itself - Table 21 covers
#    the identical 560 tumours - so the bands are a coarse plausibility screen
#    and the paired statistics are the actual test.
#
#  * TRANSFER.  Table 21 is a 12-signature de-novo NNMF in COSMIC v2 while this
#    arm refits 78 reference signatures in v3.3.  A prevalence transfers across
#    dictionaries only as well as the signature is identifiable, so the slack is
#    tiered by each signature's maximum cosine similarity to any other COSMIC
#    v3.3 signature:
#
#      SBS13 0.52  SBS17b 0.48  SBS17a 0.67  SBS2 0.71  SBS1 0.77  SBS30 0.77
#        -> distinct, slack 0.10
#      SBS8 0.82  SBS40 0.88  SBS3 0.88 (vs SBS40)  SBS5 0.88  SBS18 0.91
#        -> confusable, slack 0.20
#
# The v2 arm needs none of this: same dictionary, same cohort, so it is scored
# by paired tests against the exact reference rate instead.
SLACK_DISTINCT = 0.10    # max cosine to any other COSMIC v3.3 signature < 0.80
SLACK_CONFUSABLE = 0.20  # max cosine >= 0.80

V33_BAND_SPEC = {
    "SBS1": ("Signature 1", SLACK_DISTINCT, "ubiquitous, clock-like"),
    "SBS2": ("Signature 2", SLACK_DISTINCT, "APOBEC"),
    "SBS13": ("Signature 13", SLACK_DISTINCT, "APOBEC"),
    "SBS30": ("Signature 30", SLACK_DISTINCT, "rare (NTHL1 deficiency)"),
    "SBS3": ("Signature 3", SLACK_CONFUSABLE, "HRD; flat, confusable with SBS40"),
    "SBS8": ("Signature 8", SLACK_CONFUSABLE, "common in breast"),
    "SBS18": ("Signature 18", SLACK_CONFUSABLE, "reactive oxygen species"),
}
V33_UNION_SPEC = {
    ("SBS5", "SBS40"): ("Signature 5", SLACK_CONFUSABLE, "clock-like + flat"),
    ("SBS17a", "SBS17b"): ("Signature 17", SLACK_DISTINCT, "rare in breast"),
}


# ── Catalogue configuration ────────────────────────────────────────────────────
CATALOGS = {
    "v2": dict(
        cosmic_file=DATA_DIR / "COSMIC_v2_SBS_GRCh37.txt",
        label="COSMIC v2 (30 signatures)",
        breast=V2_BREAST,
        implausible=V2_IMPLAUSIBLE,
        ref_map=V2_REF_MAP,
        band_spec={},        # same dictionary as the reference: no bands needed
        union_spec={},       # and no v2->v3 splits to worry about
        mandatory=["Signature_1", "Signature_5"],
        apobec=("Signature_2", "Signature_13"),
        sig3="Signature_3",
        out_dir=DATA_DIR / "validation-v2",
        fig_prefix="brca_v2_",
    ),
    "v3.3": dict(
        cosmic_file=Path("Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/"
                         "COSMIC_v3.3_SBS_GRCh37.txt"),
        label="COSMIC v3.3 (78 signatures)",
        breast=V33_BREAST,
        implausible=V33_IMPLAUSIBLE,
        ref_map=V33_REF_MAP,
        band_spec=V33_BAND_SPEC,
        union_spec=V33_UNION_SPEC,
        mandatory=["SBS1", "SBS5"],
        apobec=("SBS2", "SBS13"),
        sig3="SBS3",
        out_dir=DATA_DIR / "validation-v3.3",
        fig_prefix="brca_v33_",
    ),
}

# Globals filled in by configure(); every downstream function reads these.
CATALOG = COSMIC_FILE = OUT_DIR = ASSIGN_CSV = FIG_PREFIX = None
BREAST_SIGS = MANDATORY = None
IMPLAUSIBLE_SIGS = REF_MAP = PANELS = None
EXPECTED_PREVALENCE = EXPECTED_UNION_PREVALENCE = None
APOBEC_PAIR = SIG3 = None
N_BANDED = 0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval for k successes/n."""
    from scipy.stats import beta

    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def configure(name: str):
    """Point the module at one reference dictionary."""
    global CATALOG, COSMIC_FILE, OUT_DIR, ASSIGN_CSV, FIG_PREFIX
    global BREAST_SIGS, MANDATORY, IMPLAUSIBLE_SIGS, REF_MAP, PANELS
    global EXPECTED_PREVALENCE, EXPECTED_UNION_PREVALENCE, N_BANDED
    global APOBEC_PAIR, SIG3

    cfg = CATALOGS[name]
    CATALOG = name
    COSMIC_FILE = cfg["cosmic_file"]
    OUT_DIR = cfg["out_dir"]
    ASSIGN_CSV = OUT_DIR / "assignments.csv"
    FIG_PREFIX = cfg["fig_prefix"]
    BREAST_SIGS = cfg["breast"]
    MANDATORY = cfg["mandatory"]
    IMPLAUSIBLE_SIGS = cfg["implausible"]
    REF_MAP = cfg["ref_map"]
    APOBEC_PAIR = cfg["apobec"]
    SIG3 = cfg["sig3"]
    PANELS = {"full": None, "breast": BREAST_SIGS}

    def band(v2, slack):
        lo, hi = clopper_pearson(TABLE21_NONZERO[v2], TABLE21_N)
        return round(max(0.0, lo - slack), 2), round(min(1.0, hi + slack), 2)

    EXPECTED_PREVALENCE = {
        sig: (*band(v2, slack), f"{why}; Table 21 {TABLE21_NONZERO[v2]}/{TABLE21_N}")
        for sig, (v2, slack, why) in cfg["band_spec"].items()
    }
    # members of a split: individually unconstrained, checked as a union
    EXPECTED_PREVALENCE.update({
        sig: (np.nan, np.nan,
              f"{why}; only the {'+'.join(pair)} union is pinned by Table 21 {v2}")
        for pair, (v2, _, why) in cfg["union_spec"].items() for sig in pair
    })
    EXPECTED_UNION_PREVALENCE = {
        pair: (*band(v2, slack),
               f"{why}; Table 21 {v2} {TABLE21_NONZERO[v2]}/{TABLE21_N}")
        for pair, (v2, slack, why) in cfg["union_spec"].items()
    }
    N_BANDED = len(cfg["band_spec"])


def reference_rate(sig: str) -> float:
    """Table 21 prevalence of the reference signature that maps onto `sig`."""
    for v2, members in REF_MAP.items():
        if sig in members:
            return TABLE21_NONZERO[v2] / TABLE21_N
    return np.nan


def pretty(sig: str) -> str:
    return sig.replace("Signature_", "Sig")


def pretty_ref(v2: str) -> str:
    return v2.replace("Signature ", "Sig")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_cosmic() -> pd.DataFrame:
    return pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)


def load_datasets(which=None) -> dict[str, pd.DataFrame]:
    """Return {name: DataFrame(samples x 96 contexts)}."""
    wgs = pd.read_csv(WGS_COUNTS, sep="\t", index_col=0)
    contexts = list(wgs.columns)
    sample_order = list(wgs.index)

    datasets = {"wgs": wgs}

    panel = pd.read_csv(PANEL_COUNTS, sep="\t", index_col=0)
    datasets["panel"] = panel[contexts]

    for lvl in DOWNSAMPLE_LEVELS:
        npy = DOWNSAMPLE_DIR / f"brca-downsize{lvl:03d}_counts.npy"
        ids = DOWNSAMPLE_DIR / f"brca-downsize{lvl:03d}_sample_id.csv"
        if not npy.exists():
            continue
        mat = np.load(npy)
        names = pd.read_csv(ids, header=None).iloc[:, 0].tolist()
        # row order of the downsampled matrices follows the sample-id file,
        # which is identical to the WGS count table order (verified).
        assert names == sample_order, "downsampled sample order mismatch"
        datasets[f"ds{lvl:02d}"] = pd.DataFrame(mat, index=names, columns=contexts)

    if which:
        datasets = {k: v for k, v in datasets.items() if k in which}
    return datasets


def load_hrd() -> pd.DataFrame:
    hrd = pd.read_csv(HRD_FILE, sep="\t")
    return hrd.set_index("ICGC_ID")


def load_sigma() -> pd.DataFrame:
    return pd.read_csv(SIGMA_FILE).set_index("tumor")


def load_table21() -> pd.DataFrame:
    """Table 21 sheet B as relative exposures, indexed by ICGC sample ID.

    The workbook keys samples by Sanger PD-code; icgc_brca_hrd.tsv carries both
    that code (Original_ID) and the ICGC_ID used everywhere else here.  One
    sample (PD9582c) differs by a trailing lowercase letter, so the join is on
    the letter-stripped stem, which resolves all 560.
    """
    raw = pd.read_excel(TABLE21_FILE, sheet_name="B.ContributionBySample",
                        header=1).iloc[:, 1:]
    raw = raw.dropna(subset=["Sample Name"])
    sigs = [c for c in raw.columns if str(c).startswith("Signature ")]
    counts = raw[sigs].astype(float)

    observed = {s: int((counts[s] > 0).sum()) for s in sigs}
    assert len(raw) == TABLE21_N, f"expected {TABLE21_N} samples, got {len(raw)}"
    assert observed == TABLE21_NONZERO, (
        f"Table 21 non-zero counts changed: {observed} != {TABLE21_NONZERO}")

    stem = raw["Sample Name"].astype(str).str.strip().str.replace(r"[a-z]$", "",
                                                                  regex=True)
    hrd = pd.read_csv(HRD_FILE, sep="\t")
    hrd["stem"] = hrd["Original_ID"].astype(str).str.strip().str.replace(
        r"[a-z]$", "", regex=True)
    id_map = dict(zip(hrd["stem"], hrd["ICGC_ID"]))

    counts.index = [id_map.get(s) for s in stem]
    missing = counts.index.isna().sum()
    assert missing == 0, f"{missing} Table 21 samples did not map to an ICGC ID"
    assert counts.index.is_unique, "Table 21 PD-code stems collided on ICGC IDs"
    return counts.div(counts.sum(axis=1), axis=0)


# ── sigconfide ─────────────────────────────────────────────────────────────────
def _sc_worker(args):
    sample, m, P, mandatory_idx = args
    if m.sum() <= 0:
        return sample, np.array([], dtype=int), np.array([])
    sel, exposures, _ = hybrid_stepwise_selection(
        m,
        P,
        R=R,
        pre_filter_threshold=PRE_FILTER,
        mandatory_indices=mandatory_idx or None,
    )
    return sample, sel, exposures


def run_sigconfide(counts: pd.DataFrame, P_df: pd.DataFrame, tag: str):
    if list(P_df.index) != list(counts.columns):
        raise ValueError("signature panel rows are not aligned with the catalogue")
    sig_names = np.array(P_df.columns)
    P = P_df.values.astype(float)
    mandatory_idx = [
        list(sig_names).index(s) for s in MANDATORY if s in list(sig_names)
    ]

    tasks = [
        (s, counts.loc[s].values.astype(float), P, mandatory_idx) for s in counts.index
    ]
    t0 = time.time()
    rows = []
    done = 0
    step = max(1, len(tasks) // 4)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_sc_worker, t) for t in tasks]
        for fut in as_completed(futs):
            sample, sel, exposures = fut.result()
            n_mut = int(counts.loc[sample].sum())
            for j, e in zip(sel, exposures):
                if e > 0:
                    rows.append(
                        dict(
                            sample=sample,
                            signature=sig_names[j],
                            exposure=float(e),
                            n_mut=n_mut,
                        )
                    )
            done += 1
            if done % step == 0 or done == len(tasks):
                print(f"    [sigconfide {tag}] {done}/{len(tasks)} "
                      f"({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows), round(time.time() - t0, 1)


# ── SigProfilerAssignment ──────────────────────────────────────────────────────
def run_spa(counts: pd.DataFrame, P_df: pd.DataFrame, tag: str, tmpdir: str):
    from SigProfilerAssignment import Analyzer as SPA

    samp_path = os.path.join(tmpdir, f"samples_{tag}.txt")
    sig_path = os.path.join(tmpdir, f"sigs_{tag}.txt")
    out_path = os.path.join(tmpdir, f"spa_{tag}")

    M = counts.T.copy()  # contexts x samples
    M.index.name = "MutationType"  # SPA requires a named mutation-type index
    M.to_csv(samp_path, sep="\t")
    P_df.to_csv(sig_path, sep="\t")

    t0 = time.time()
    try:
        SPA.cosmic_fit(
            samples=samp_path,
            output=out_path,
            signature_database=sig_path,
            collapse_to_SBS96=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    [SPA {tag}] ERROR: {e}")
        return pd.DataFrame(), None

    act_path = os.path.join(
        out_path,
        "Assignment_Solution",
        "Activities",
        "Assignment_Solution_Activities.txt",
    )
    if not os.path.exists(act_path):
        print(f"    [SPA {tag}] activities file missing")
        return pd.DataFrame(), None

    act = pd.read_csv(act_path, sep="\t", index_col=0)
    act = act.reindex(counts.index).fillna(0.0)
    tot = act.sum(axis=1).replace(0, np.nan)
    rel = act.div(tot, axis=0).fillna(0.0)

    rows = []
    for sample in act.index:
        n_mut = int(counts.loc[sample].sum())
        for sig in act.columns:
            if act.loc[sample, sig] > 0:
                rows.append(
                    dict(
                        sample=sample,
                        signature=sig,
                        exposure=float(rel.loc[sample, sig]),
                        n_mut=n_mut,
                    )
                )
    print(f"    [SPA {tag}] done ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows), round(time.time() - t0, 1)


# ── Fit stage ──────────────────────────────────────────────────────────────────
def stage_fit(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cosmic = load_cosmic()
    datasets = load_datasets(args.datasets)

    # The COSMIC panel and the ICGC count tables list the 96 contexts in
    # different orders (COSMIC groups by 5' base, the catalogues by mutation
    # class).  sigconfide multiplies the two row-wise, so they must agree; and
    # SigProfilerAssignment reads a custom signature database positionally,
    # i.e. it only produces correct assignments when both files are in COSMIC's
    # own order.  Everything is therefore reindexed onto COSMIC's order.
    contexts = list(cosmic.index)
    for name, df in datasets.items():
        missing = [ctx for ctx in contexts if ctx not in df.columns]
        if missing:
            raise ValueError(
                f"{name}: contexts absent from the catalogue: {missing[:5]}"
            )
    datasets = {name: df[contexts] for name, df in datasets.items()}

    all_rows, timings = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        for ds_name, counts in datasets.items():
            if args.max_samples:
                counts = counts.iloc[: args.max_samples]
            for panel_name in args.panels:
                sigs = PANELS[panel_name]
                P_df = cosmic if sigs is None else cosmic[sigs]
                for method in args.methods:
                    tag = f"{ds_name}_{panel_name}_{method}"
                    print(f"\n═══ {tag}  ({len(counts)} samples, "
                          f"{P_df.shape[1]} signatures) ═══")
                    if method == "sigconfide":
                        df, secs = run_sigconfide(counts, P_df, tag)
                    else:
                        df, secs = run_spa(counts, P_df, tag, tmpdir)
                    if df.empty:
                        continue
                    df.insert(0, "method", method)
                    df.insert(0, "panel", panel_name)
                    df.insert(0, "dataset", ds_name)
                    all_rows.append(df)
                    timings.append(
                        dict(dataset=ds_name, panel=panel_name, method=method,
                             n_samples=len(counts), seconds=secs)
                    )

    out = pd.concat(all_rows, ignore_index=True)
    if args.append and ASSIGN_CSV.exists():
        prev = pd.read_csv(ASSIGN_CSV)
        key = ["dataset", "panel", "method"]
        done = set(map(tuple, out[key].drop_duplicates().values.tolist()))
        prev = prev[~prev[key].apply(tuple, axis=1).isin(done)]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(ASSIGN_CSV, index=False)
    pd.DataFrame(timings).to_csv(OUT_DIR / "timings.csv", index=False)
    print(f"\nAssignments → {ASSIGN_CSV}  ({len(out)} rows)")


# ── Metrics ────────────────────────────────────────────────────────────────────
def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and ppv + sens) else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else np.nan
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=spec,
        ppv=ppv, npv=npv, f1=f1, mcc=mcc,
        balanced_acc=np.nanmean([sens, spec]), detection_rate=pred.mean(),
    )


def auc_score(scores: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), ties handled by average ranks."""
    truth = truth.astype(bool)
    n1, n0 = truth.sum(), (~truth).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(scores).rank().values
    return float((ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fisher_or_p(pred: np.ndarray, truth: np.ndarray):
    from scipy.stats import fisher_exact

    m = binary_metrics(pred, truth)
    table = [[m["tp"], m["fp"]], [m["fn"], m["tn"]]]
    or_, p = fisher_exact(table)
    return or_, p


def wide_matrices(assign: pd.DataFrame, dataset: str, panel: str, method: str,
                  samples: list[str]):
    """(binary detection DataFrame, relative-exposure DataFrame) samples x sigs."""
    sub = assign[
        (assign.dataset == dataset)
        & (assign.panel == panel)
        & (assign.method == method)
    ]
    if sub.empty:
        return None, None
    exp = sub.pivot_table(index="sample", columns="signature", values="exposure",
                          aggfunc="first").reindex(samples).fillna(0.0)
    return (exp > 0), exp


# ── Report stage ───────────────────────────────────────────────────────────────
def stage_report(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assign = pd.read_csv(ASSIGN_CSV)
    hrd = load_hrd()
    sigma = load_sigma()
    datasets = load_datasets()
    try:
        t21 = load_table21()
    except (FileNotFoundError, ImportError) as e:
        print(f"[warn] Table 21 unavailable ({e}); skipping per-sample "
              f"concordance against Nik-Zainal 2016")
        t21 = None

    combos = list(map(tuple, assign[["dataset", "panel", "method"]]
                      .drop_duplicates().values.tolist()))

    det_rows, imp_rows, apo_rows, uni_rows = [], [], [], []
    det_cache: dict[tuple, pd.DataFrame] = {}
    exp_cache: dict[tuple, pd.DataFrame] = {}

    for ds_name, panel_name, method in combos:
        samples = list(datasets[ds_name].index)
        det, exp = wide_matrices(assign, ds_name, panel_name, method, samples)
        if det is None:
            continue
        det_cache[(ds_name, panel_name, method)] = det
        exp_cache[(ds_name, panel_name, method)] = exp
        n = len(samples)

        # 1. per-signature detection rate, against the reference prevalence
        for sig in det.columns:
            lo, hi, why = EXPECTED_PREVALENCE.get(sig, (np.nan, np.nan, ""))
            rate = float(det[sig].mean())
            # the rate's own precision as an estimate for breast cancer at
            # large; agreement with the reference is a *paired* question and is
            # answered by prevalence_diff in table21_concordance(), not by
            # whether this interval covers the reference rate
            ci_lo, ci_hi = _wilson(int(det[sig].sum()), n)
            det_rows.append(dict(
                dataset=ds_name, panel=panel_name, method=method, signature=sig,
                n_samples=n, n_detected=int(det[sig].sum()), detection_rate=rate,
                rate_lo=ci_lo, rate_hi=ci_hi,
                table21_rate=reference_rate(sig),
                expected_lo=lo, expected_hi=hi, expectation=why,
                in_expected_band=(bool(lo <= rate <= hi) if np.isfinite(lo) else None),
                mean_exposure_when_called=(float(exp.loc[det[sig], sig].mean())
                                           if det[sig].any() else 0.0),
                implausible=sig in IMPLAUSIBLE_SIGS,
            ))

        # 1b. split pairs, where the reference pins only the union
        for pair, (lo, hi, why) in EXPECTED_UNION_PREVALENCE.items():
            present = [s for s in pair if s in det.columns]
            if not present:
                continue
            rate = float(det[present].any(axis=1).mean())
            uni_rows.append(dict(
                dataset=ds_name, panel=panel_name, method=method,
                pair="+".join(pair), members_present="+".join(present),
                n_samples=n, detection_rate=rate, expected_lo=lo, expected_hi=hi,
                expectation=why, in_expected_band=bool(lo <= rate <= hi),
            ))

        # 2. APOBEC coherence: the two APOBEC signatures come from the same
        #    enzymes, so a method that reports one without the other is
        #    internally inconsistent.
        a, b = APOBEC_PAIR
        if a in det.columns and b in det.columns:
            either = det[a] | det[b]
            both = det[a] & det[b]
            xor = either & ~both
            from scipy.stats import spearmanr

            rho = np.nan
            if both.sum() > 5:
                rho = spearmanr(exp.loc[both, a], exp.loc[both, b]).statistic
            apo_rows.append(dict(
                dataset=ds_name, panel=panel_name, method=method,
                n_either=int(either.sum()), n_both=int(both.sum()),
                discordance=float(xor.sum() / either.sum()) if either.any() else np.nan,
                exposure_rho_when_both=rho,
            ))

        # 3. tissue-implausible calls (only meaningful on the full panel)
        if panel_name == "full":
            imp_cols = [c for c in det.columns if c in IMPLAUSIBLE_SIGS]
            imp = det[imp_cols]
            n_imp = int((imp.sum(axis=1) > 0).sum())
            imp_lo, imp_hi = _wilson(n_imp, n)
            imp_rows.append(dict(
                dataset=ds_name, method=method, n_samples=n,
                median_mut=float(np.median(datasets[ds_name].sum(axis=1))),
                mean_implausible_per_sample=float(imp.sum(axis=1).mean()),
                pct_samples_with_implausible=float(n_imp / n),
                pct_implausible_lo=imp_lo, pct_implausible_hi=imp_hi,
                mean_implausible_exposure=float(exp[imp_cols].sum(axis=1).mean()),
                mean_sigs_per_sample=float(det.sum(axis=1).mean()),
                pct_offpanel_calls=float(
                    det[[c for c in det.columns if c not in BREAST_SIGS]].sum().sum()
                    / max(det.sum().sum(), 1)),
            ))

    # ── 4. Low-burden recovery ────────────────────────────────────────────────
    rec_rows = recovery_vs_wgs_consensus(det_cache, datasets, sigma)
    t21_rec = (recovery_vs_table21(det_cache, datasets, t21, sigma)
               if t21 is not None else [])

    # ── 5. HRDetect data-integrity diagnostic ────────────────────────────────
    integ_rows = hrdetect_integrity(det_cache, exp_cache, datasets, hrd, sigma)

    # ── 6. Per-sample agreement with Nik-Zainal Table 21 ─────────────────────
    if t21 is not None:
        conc_df, prof_df = table21_concordance(det_cache, exp_cache, datasets, t21)
    else:
        conc_df, prof_df = pd.DataFrame(), pd.DataFrame()

    det_df = pd.DataFrame(det_rows)
    imp_df = pd.DataFrame(imp_rows)
    apo_df = pd.DataFrame(apo_rows)
    rec_df = pd.DataFrame(rec_rows)
    t21rec_df = pd.DataFrame(t21_rec)
    integ_df = pd.DataFrame(integ_rows)
    uni_df = pd.DataFrame(uni_rows)

    det_df.to_csv(OUT_DIR / "detection_rates.csv", index=False)
    imp_df.to_csv(OUT_DIR / "implausible_calls.csv", index=False)
    apo_df.to_csv(OUT_DIR / "apobec_coherence.csv", index=False)
    rec_df.to_csv(OUT_DIR / "low_burden_recovery.csv", index=False)
    t21rec_df.to_csv(OUT_DIR / "low_burden_recovery_table21.csv", index=False)
    integ_df.to_csv(OUT_DIR / "hrdetect_integrity.csv", index=False)
    uni_df.to_csv(OUT_DIR / "union_prevalence.csv", index=False)
    conc_df.to_csv(OUT_DIR / "table21_concordance.csv", index=False)
    prof_df.to_csv(OUT_DIR / "table21_profile_similarity.csv", index=False)

    _print_report(det_df, imp_df, apo_df, rec_df, integ_df, uni_df, conc_df,
                  prof_df, t21rec_df)
    if not args.no_figures:
        make_figures(det_df, imp_df, rec_df, det_cache, exp_cache, datasets,
                     conc_df, t21rec_df)
    make_latex(det_df, imp_df, apo_df, rec_df, uni_df, conc_df, t21rec_df)
    return (det_df, imp_df, apo_df, rec_df, integ_df, uni_df, conc_df, prof_df,
            t21rec_df)


def _wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k successes out of n.

    Used for rates over tumours, where the only randomness is which tumours the
    cohort happens to contain.  Preferred to the normal approximation because
    several of these rates sit against 0 or 1, where that approximation runs
    outside the unit interval.
    """
    from scipy.stats import norm

    if n <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _as_matrix(a) -> np.ndarray:
    """A (tumours x signatures) boolean matrix, from either shape."""
    a = np.asarray(a, dtype=bool)
    return a[:, None] if a.ndim == 1 else a


def _confusion_scores(pred, mask_pos, mask_neg, n_boot=0, rng=None) -> dict:
    """Sensitivity, specificity, PPV, F1 and MCC, with optional 95% intervals.

    Inputs are (tumours x signatures) boolean matrices.  The bootstrap resamples
    whole tumours rather than individual calls: one tumour contributes a row of
    signature calls that rise and fall together, so resampling calls
    independently would understate the interval on the pooled rows.
    """
    pred, mask_pos, mask_neg = map(_as_matrix, (pred, mask_pos, mask_neg))
    tp = int((pred & mask_pos).sum())
    fn = int((~pred & mask_pos).sum())
    fp = int((pred & mask_neg).sum())
    tn = int((~pred & mask_neg).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and ppv + sens) else 0.0
    mcc, _ = _mcc_jaccard(tp, fp, fn, tn)
    out = dict(n_positive=tp + fn, n_negative=tn + fp, sensitivity=sens,
               specificity=spec, ppv=ppv, f1=f1, mcc=float(mcc))

    lo_hi = {k: (np.nan, np.nan) for k in ("sensitivity", "specificity", "mcc")}
    if n_boot and rng is not None:
        n = pred.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        P, Po, Ne = pred[idx], mask_pos[idx], mask_neg[idx]
        btp = (P & Po).sum((1, 2))
        bfn = (~P & Po).sum((1, 2))
        bfp = (P & Ne).sum((1, 2))
        btn = (~P & Ne).sum((1, 2))
        bmcc, _ = _mcc_jaccard(btp, bfp, bfn, btn)
        with np.errstate(invalid="ignore", divide="ignore"):
            pos_n, neg_n = btp + bfn, btn + bfp
            bsens = np.where(pos_n > 0, btp / np.where(pos_n > 0, pos_n, 1), np.nan)
            bspec = np.where(neg_n > 0, btn / np.where(neg_n > 0, neg_n, 1), np.nan)
        for key, draws in (("sensitivity", bsens), ("specificity", bspec),
                           ("mcc", bmcc)):
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])
            lo_hi[key] = (float(lo), float(hi))
    for key, (lo, hi) in lo_hi.items():
        out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi
    return out


def recovery_vs_table21(det_cache, datasets, t21, sigma,
                        n_boot=N_BOOT, seed=BOOT_SEED + 1):
    """How much of the authors' own per-tumour assignment survives at low burden?

    Truth is external here: a signature is present in a tumour when Table 21
    assigns it there, absent otherwise.  The same 560 tumours are then refit
    from progressively thinner catalogues (whole genome, then 18 down to 3
    mutations, plus the panel simulation), so the curve reads as "how much of
    the published assignment can still be recovered from n mutations".

    The pooled 'all' scope excludes the mandatory signatures, whose detection is
    forced in sigconfide and would otherwise inflate its sensitivity.

    Every rate carries a 95% interval from a bootstrap over tumours (its own
    generator, so the concordance intervals are unaffected).
    """
    rng = np.random.default_rng(seed)
    rows = []
    scored = [v2 for v2, members in REF_MAP.items()
              if any(m in BREAST_SIGS for m in members)]
    pooled = [v2 for v2 in scored
              if not set(REF_MAP[v2]) & set(MANDATORY)]

    for (ds_name, panel_name, method), det in sorted(det_cache.items()):
        if panel_name != "breast":
            continue
        samples = [s for s in datasets[ds_name].index if s in t21.index]
        ref = t21.loc[samples]
        det = det.reindex(samples).fillna(False)
        burden = float(np.median(datasets[ds_name].sum(axis=1).reindex(samples)))

        def call(v2):
            members = [m for m in REF_MAP[v2] if m in det.columns]
            if not members:
                return np.zeros(len(samples), dtype=bool)
            return det[members].any(axis=1).values.astype(bool)

        pool_pred, pool_truth = [], []
        for v2 in scored:
            pred = call(v2)
            truth = (ref[v2] > 0).values.astype(bool)
            rows.append(dict(dataset=ds_name, method=method,
                             scope=pretty_ref(v2), median_mut=burden,
                             **_confusion_scores(pred, truth, ~truth,
                                                 n_boot, rng)))
            if v2 in pooled:
                pool_pred.append(pred)
                pool_truth.append(truth)

        # stacked as columns, not concatenated: the bootstrap has to resample
        # tumours with all their signatures attached
        pred = np.column_stack(pool_pred)
        truth = np.column_stack(pool_truth)
        rows.append(dict(dataset=ds_name, method=method, scope="all",
                         median_mut=burden, n_signatures=len(pooled),
                         **_confusion_scores(pred, truth, ~truth, n_boot, rng)))

    # SigMA, run independently on the panel version of the same tumours
    common = [s for s in sigma.index if s in t21.index]
    sm = sigma.loc[common]
    ref = t21.loc[common]
    burden = float(np.median(sm["total_snvs"]))
    for v2, pred in (
        ("Signature 3",
         sm["categ"].isin(["Signature_3_hc", "Signature_3_lc"]).values),
        ("Signature 2", (sm["categ"] == "Signature_APOBEC").values),
        ("Signature 13", (sm["categ"] == "Signature_APOBEC").values),
    ):
        truth = (ref[v2] > 0).values.astype(bool)
        rows.append(dict(dataset="panel", method="SigMA", scope=pretty_ref(v2),
                         median_mut=burden,
                         **_confusion_scores(pred, truth, ~truth, n_boot, rng)))
    return rows


def recovery_vs_wgs_consensus(det_cache, datasets, sigma,
                              n_boot=N_BOOT, seed=BOOT_SEED + 2):
    """How much of the whole-genome signal survives at panel-like burden?

    A method-internal companion to recovery_vs_table21: a signature counts as
    truly present in a tumour when *both* sigconfide and SPA call it on the full
    whole-genome catalogue of that same tumour, and truly absent when neither
    does.  Signatures the two methods disagree about at WGS depth are excluded,
    so neither method is scored against its own opinion.
    """
    rng = np.random.default_rng(seed)
    sc_key, spa_key = ("wgs", "breast", "sigconfide"), ("wgs", "breast", "spa")
    if sc_key not in det_cache or spa_key not in det_cache:
        return []
    sc, spa = det_cache[sc_key], det_cache[spa_key]
    sigs = [s for s in BREAST_SIGS if s in sc.columns and s in spa.columns]
    sc, spa = sc[sigs], spa[sigs]
    pos = sc & spa
    neg = ~sc & ~spa

    rows = []
    for (ds_name, panel_name, method), det in sorted(det_cache.items()):
        if ds_name == "wgs" or panel_name != "breast":
            continue
        det = det.reindex(columns=sigs, fill_value=False)
        common = [s for s in det.index if s in pos.index]
        d, p, n = det.loc[common], pos.loc[common], neg.loc[common]
        burden = float(np.median(datasets[ds_name].sum(axis=1).reindex(common)))

        for scope in ["all"] + sigs:
            if scope == "all":
                mask_p, mask_n, pred = p.values, n.values, d.values
            else:
                mask_p = p[[scope]].values
                mask_n = n[[scope]].values
                pred = d[[scope]].values
            rows.append(dict(dataset=ds_name, method=method, scope=scope,
                             median_mut=burden,
                             **_confusion_scores(pred, mask_p, mask_n,
                                                 n_boot, rng)))

    # SigMA on the same panel simulation, for Signature 3 and APOBEC
    common = [s for s in sigma.index if s in pos.index]
    sm = sigma.loc[common]
    burden = float(np.median(sm["total_snvs"]))
    for scope, pred in (
        (SIG3, sm["categ"].isin(["Signature_3_hc", "Signature_3_lc"]).values),
        (APOBEC_PAIR[0], (sm["categ"] == "Signature_APOBEC").values),
        (APOBEC_PAIR[1], (sm["categ"] == "Signature_APOBEC").values),
    ):
        if scope not in pos.columns:
            continue
        rows.append(dict(dataset="panel", method="SigMA", scope=scope,
                         median_mut=burden,
                         **_confusion_scores(pred, pos.loc[common, scope].values,
                                             neg.loc[common, scope].values,
                                             n_boot, rng)))
    return rows


def hrdetect_integrity(det_cache, exp_cache, datasets, hrd, sigma):
    """Check whether the shipped HRDetect labels line up with these catalogues.

    Signature 3 is the HRD signature, so on correctly paired data the
    whole-genome Signature 3 exposure separates HRDetect-positive from
    HRDetect-negative tumours sharply.  Two positive controls are computed
    alongside: SigMA (run independently on the panel version of the same
    tumours) against the same WGS exposures, and a scan over all 96 mutation
    contexts.
    """
    rows = []
    key = ("wgs", "breast", "sigconfide")
    if key not in exp_cache:
        return rows
    wgs = datasets["wgs"]
    ids = list(wgs.index)
    y = hrd.reindex(ids)["hrdetect_status"].values.astype(bool)

    for method in ("sigconfide", "spa"):
        k = ("wgs", "breast", method)
        if k not in exp_cache:
            continue
        sbs3 = exp_cache[k].reindex(ids)[SIG3].fillna(0.0).values
        rows.append(dict(
            comparison=f"WGS {SIG3} exposure ({method}) vs HRDetect label",
            auc=auc_score(sbs3, y), n=len(ids), expected="~0.85-0.95 if aligned",
        ))
        # positive control: an independently run tool on the same tumours
        common = [s for s in sigma.index if s in ids]
        sm_hc = (sigma.loc[common, "categ"] == "Signature_3_hc").values
        sbs3_c = exp_cache[k].reindex(common)[SIG3].fillna(0.0).values
        rows.append(dict(
            comparison=f"WGS {SIG3} exposure ({method}) vs SigMA Sig3 "
                       f"high-confidence",
            auc=auc_score(sbs3_c, sm_hc), n=len(common),
            expected="high if the catalogues and SigMA agree",
        ))

    # scan every mutation context for any association with the label at all
    freq = wgs.div(wgs.sum(axis=1), axis=0)
    aucs = np.array([auc_score(freq[c].values, y) for c in wgs.columns])
    worst = int(np.argmax(np.abs(aucs - 0.5)))
    rows.append(dict(
        comparison=f"best single mutation context vs HRDetect ({wgs.columns[worst]})",
        auc=float(aucs[worst]), n=len(ids),
        expected="several contexts well away from 0.5 if aligned",
    ))
    return rows


def _spearman_ci(rho: float, n: int, alpha: float = 0.05):
    """Fisher z interval for a rank correlation, Bonett-Wright standard error.

    Spearman's rho is not Fisher-normal with se = 1/sqrt(n-3); the accepted
    correction inflates it by sqrt(1 + rho^2/2), which is what Bonett & Wright
    (2000) recommend and what is used here.
    """
    from scipy.stats import norm

    if not np.isfinite(rho) or n < 10 or abs(rho) >= 1:
        return np.nan, np.nan
    se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))
    z = np.arctanh(rho)
    crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def _mcc_jaccard(tp, fp, fn, tn):
    """Vectorised MCC and Jaccard from confusion counts (arrays or scalars)."""
    tp, fp, fn, tn = (np.asarray(v, dtype=float) for v in (tp, fp, fn, tn))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1),
                       np.nan)
        union = tp + fp + fn
        jac = np.where(union > 0, tp / np.where(union > 0, union, 1), np.nan)
    return mcc, jac


def _paired_bootstrap(pred: np.ndarray, truth: np.ndarray, n_boot: int, rng):
    """Percentile CIs for MCC, Jaccard and the prevalence difference.

    Samples are resampled as pairs, so the interval reflects the only randomness
    that exists here: which tumours ended up in the cohort.  There is no
    between-cohort sampling error to model, because Table 21 scores the identical
    560 tumours.
    """
    n = len(pred)
    idx = rng.integers(0, n, size=(n_boot, n))
    p, t = pred[idx], truth[idx]
    tp = (p & t).sum(1)
    fp = (p & ~t).sum(1)
    fn = (~p & t).sum(1)
    tn = (~p & ~t).sum(1)
    mcc, jac = _mcc_jaccard(tp, fp, fn, tn)
    diff = p.mean(1) - t.mean(1)
    q = lambda a: tuple(np.nanpercentile(a, [2.5, 97.5]))  # noqa: E731
    return q(mcc), q(jac), q(diff)


def table21_concordance(det_cache, exp_cache, datasets, t21,
                        n_boot=N_BOOT, seed=BOOT_SEED):
    """Per-sample agreement with Nik-Zainal Table 21 on the same 560 tumours.

    Prevalence alone only asks whether a method calls a signature at roughly the
    right *rate*; it is satisfied by a method that calls it in entirely the wrong
    tumours.  Because Table 21 assigns signatures sample by sample over the
    identical cohort, agreement can be scored per tumour instead:

      * detection  - confusion counts against Table 21 non-zero, summarised by
        MCC and Jaccard (both degrade if the right rate is reached in the wrong
        samples), with paired-bootstrap 95% intervals;
      * bias       - McNemar's exact test on the discordant pairs, which asks
        whether over- or under-calling is systematic rather than balanced.  In
        the v2 arm this is also the exact test of the prevalence difference,
        since reference and fit live in the same dictionary;
      * exposure   - Spearman correlation of relative exposure across tumours,
        with a Fisher-z interval.  Rank correlation is used because the two
        normalisations differ (12 reference signatures vs the fitted panel).

    In the v3.3 arm a reference signature that split is compared against the
    union of its descendants: detected when any member is called, exposures
    summed over members.
    """
    from scipy.stats import binomtest, spearmanr

    rng = np.random.default_rng(seed)
    rows, prof_rows = [], []

    for (ds_name, panel_name, method), det in det_cache.items():
        exp = exp_cache[(ds_name, panel_name, method)]
        samples = [s for s in datasets[ds_name].index if s in t21.index]
        if not samples:
            continue
        ref = t21.loc[samples]
        det = det.reindex(samples).fillna(False)
        exp = exp.reindex(samples).fillna(0.0)

        pred_prof, truth_prof, used_v2 = [], [], []
        for v2, members in REF_MAP.items():
            present = [m for m in members if m in det.columns]
            if not present:
                continue
            pred = det[present].any(axis=1).values.astype(bool)
            truth = (ref[v2] > 0).values.astype(bool)
            pred_exp = exp[present].sum(axis=1).values
            truth_exp = ref[v2].values
            used_v2.append(v2)
            pred_prof.append(pred_exp)
            truth_prof.append(truth_exp)

            m = binary_metrics(pred, truth)
            mcc, jac = _mcc_jaccard(m["tp"], m["fp"], m["fn"], m["tn"])
            b, c = m["fp"], m["fn"]  # discordant pairs
            p_mcnemar = (binomtest(b, b + c, 0.5).pvalue if b + c else np.nan)
            rho = spearmanr(pred_exp, truth_exp).statistic
            rho_lo, rho_hi = _spearman_ci(rho, len(samples))
            # rho over every tumour counts "both say absent" as agreement, which
            # dominates for rare signatures; restricting to co-detected tumours
            # isolates whether the *amount* tracks.  Needs enough co-detections
            # to mean anything, hence the floor.
            both = pred & truth
            n_both = int(both.sum())
            rho_both = (spearmanr(pred_exp[both], truth_exp[both]).statistic
                        if n_both > 5 else np.nan)
            rb_lo, rb_hi = _spearman_ci(rho_both, n_both)
            (mcc_lo, mcc_hi), (j_lo, j_hi), (d_lo, d_hi) = _paired_bootstrap(
                pred, truth, n_boot, rng)

            rows.append(dict(
                dataset=ds_name, panel=panel_name, method=method,
                table21_signature=v2, fitted_signatures="+".join(present),
                n=len(samples), n_truth_pos=int(truth.sum()),
                n_pred_pos=int(pred.sum()),
                tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                sensitivity=m["sensitivity"], specificity=m["specificity"],
                ppv=m["ppv"], mcc=float(mcc), mcc_lo=mcc_lo, mcc_hi=mcc_hi,
                jaccard=float(jac), jaccard_lo=j_lo, jaccard_hi=j_hi,
                prevalence_diff=float(pred.mean() - truth.mean()),
                prevalence_diff_lo=d_lo, prevalence_diff_hi=d_hi,
                mcnemar_p=float(p_mcnemar) if np.isfinite(p_mcnemar) else np.nan,
                exposure_rho=float(rho) if np.isfinite(rho) else np.nan,
                exposure_rho_lo=rho_lo, exposure_rho_hi=rho_hi,
                # MANDATORY signatures are forced into every fit, so their
                # detection is constant and MCC is undefined by construction;
                # only their exposure correlation carries information.
                mandatory_forced=bool(set(present) & set(MANDATORY)),
                exposure_rho_when_both=(float(rho_both)
                                        if np.isfinite(rho_both) else np.nan),
                exposure_rho_when_both_lo=rb_lo,
                exposure_rho_when_both_hi=rb_hi,
                n_both=n_both,
            ))

        # per-sample cosine between the mapped exposure profiles
        if pred_prof:
            A = np.vstack(pred_prof).T   # samples x mapped reference signatures
            B = np.vstack(truth_prof).T
            na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
            ok = (na > 0) & (nb > 0)
            cos = np.full(len(samples), np.nan)
            cos[ok] = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
            prof_rows.append(dict(
                dataset=ds_name, panel=panel_name, method=method,
                n=int(ok.sum()), n_mapped_signatures=len(used_v2),
                cosine_median=float(np.nanmedian(cos)),
                cosine_q25=float(np.nanpercentile(cos[ok], 25)),
                cosine_q75=float(np.nanpercentile(cos[ok], 75)),
                cosine_frac_above_0_9=float((cos[ok] >= 0.9).mean()),
                mean_sigs_called=float(det.sum(axis=1).mean()),
                mean_sigs_table21=float((ref > 0).sum(axis=1).mean()),
            ))

    return pd.DataFrame(rows), pd.DataFrame(prof_rows)


# ── Figures ────────────────────────────────────────────────────────────────────
COLORS = {"sigconfide": "#1f77b4", "spa": "#ff7f0e", "SigMA": "#2ca02c"}
LABELS = {"sigconfide": "sigconfide", "spa": "SPA", "SigMA": "SigMA"}


def _burden_figure(plt, df, scopes, title, path):
    """Sensitivity and specificity against mutation burden, one column per scope.

    Burden is drawn on evenly spaced positions rather than a log axis: the
    whole-genome point sits three orders of magnitude from the thinned ones, and
    on a real axis it squeezes the 3-18 mutation region - the part the figure is
    about - into a sliver.
    """
    burdens = sorted(df.median_mut.unique())
    pos = {b: i for i, b in enumerate(burdens)}
    labels_x = [("WGS" if b > 100 else f"{b:.0f}") for b in burdens]
    shown = df[df.scope.isin([s for s, _ in scopes])]
    spec_col = ("specificity_lo" if "specificity_lo" in df.columns
                else "specificity")
    spec_min = float(np.nanmin(shown[spec_col]))

    fig, axes = plt.subplots(2, len(scopes), figsize=(13, 6.4), sharex=True)
    for col, (scope, heading) in enumerate(scopes):
        for row, metric in enumerate(["sensitivity", "specificity"]):
            ax = axes[row, col]
            lo_c, hi_c = f"{metric}_lo", f"{metric}_hi"
            has_ci = lo_c in df.columns
            for meth in ("sigconfide", "spa"):
                sub = df[(df.method == meth) & (df.scope == scope)]
                sub = sub.sort_values("median_mut")
                if sub.empty:
                    continue
                x = [pos[b] for b in sub.median_mut]
                if has_ci:
                    ax.fill_between(x, sub[lo_c], sub[hi_c], color=COLORS[meth],
                                    alpha=0.18, linewidth=0)
                ax.plot(x, sub[metric], "o-", color=COLORS[meth],
                        label=LABELS[meth])
            sm = df[(df.method == "SigMA") & (df.scope == scope)]
            if not sm.empty:
                x = [pos[b] for b in sm.median_mut]
                if has_ci:
                    ax.errorbar(x, sm[metric],
                                yerr=[sm[metric] - sm[lo_c], sm[hi_c] - sm[metric]],
                                fmt="none", ecolor=COLORS["SigMA"], elinewidth=1.2,
                                capsize=3)
                ax.plot(x, sm[metric], "*", ms=14, color=COLORS["SigMA"],
                        label="SigMA")
            ax.set_xticks(range(len(burdens)))
            ax.set_xticklabels(labels_x, fontsize=8)
            if metric == "sensitivity":
                ax.set_ylim(-0.03, 1.03)
            else:
                ax.set_ylim(min(0.55, spec_min - 0.05), 1.02)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(heading, fontsize=12)
            if row == 1:
                ax.set_xlabel("Median mutations per sample")
            if col == 0:
                ax.set_ylabel(metric.capitalize())
    handles, labels = [], []
    for ax in axes.ravel():
        for hh, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(hh)
                labels.append(lab)
    axes[0, 0].legend(handles, labels, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_figures(det_df, imp_df, rec_df, det_cache, exp_cache, datasets,
                 conc_df=None, t21rec_df=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    out = []
    banded = bool(EXPECTED_PREVALENCE)

    # ── Fig 1: detection rate vs the prevalence expected in breast tissue ─────
    t = det_df[(det_df.dataset == "wgs") & (det_df.panel == "breast")]
    piv = t.pivot_table(index="signature", columns="method", values="detection_rate")
    order = [s for s in BREAST_SIGS if s in piv.index][::-1]
    piv = piv.reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(piv) + 1.8))
    y = np.arange(len(piv))
    if banded:
        # members of a split carry no individual band; show the pair's union
        # band hatched instead, so a blank row reads as "not individually
        # pinned" rather than as a missing expectation.
        union_of = {s: pair for pair in EXPECTED_UNION_PREVALENCE for s in pair}
        drawn_band = drawn_union = False
        for i, sig in enumerate(piv.index):
            lo, hi, _ = EXPECTED_PREVALENCE.get(sig, (np.nan, np.nan, ""))
            if np.isfinite(lo):
                ax.barh(i, (hi - lo) * 100, left=lo * 100, height=0.82,
                        color="#d9d9d9", zorder=0,
                        label=None if drawn_band else "expected in breast")
                drawn_band = True
            elif sig in union_of:
                ulo, uhi, _ = EXPECTED_UNION_PREVALENCE[union_of[sig]]
                ax.barh(i, (uhi - ulo) * 100, left=ulo * 100, height=0.82,
                        facecolor="none", edgecolor="#bdbdbd", hatch="//",
                        linewidth=0.8, zorder=0,
                        label=(None if drawn_union else
                               "expected for the pair (union only)"))
                drawn_union = True
    else:
        union_of = {}
        # same dictionary as the reference: the Table 21 rate is an exact
        # per-tumour figure on these very genomes, so mark the value itself
        for i, sig in enumerate(piv.index):
            rate = reference_rate(sig)
            if not np.isfinite(rate):
                continue
            ax.plot([rate * 100, rate * 100], [i - 0.42, i + 0.42], "-",
                    color="#333333", lw=2.2, zorder=4,
                    label=None if i else "Nik-Zainal Table 21")
    h = 0.34
    err = {m: t[t.method == m].set_index("signature").reindex(piv.index)
           for m in piv.columns}
    for k, meth in enumerate([m for m in ("sigconfide", "spa") if m in piv.columns]):
        yy = y + (0.5 - k) * h
        ax.barh(yy, piv[meth] * 100, height=h * 0.85,
                color=COLORS[meth], label=LABELS[meth], zorder=2)
        e = err[meth]
        if "rate_lo" in e.columns:
            ax.errorbar(piv[meth] * 100, yy,
                        xerr=[(piv[meth] - e["rate_lo"]) * 100,
                              (e["rate_hi"] - piv[meth]) * 100],
                        fmt="none", ecolor="#3f3f3f", elinewidth=0.9, capsize=2,
                        zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{pretty(s)}\n(w/ {pretty([m for m in union_of[s] if m != s][0])})"
        if s in union_of else pretty(s)
        for s in piv.index
    ])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Detection rate (% of 560 breast whole genomes)")
    ax.set_title("Signature prevalence vs. the Nik-Zainal assignment of the same "
                 f"genomes\n(tissue-informed {len(BREAST_SIGS)}-signature panel, "
                 f"{CATALOGS[CATALOG]['label']} reference)", fontsize=11)
    ax.set_xlim(0, 108)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=8, loc="center right",
              bbox_to_anchor=(1.0, 0.30))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = FIG_DIR / f"{FIG_PREFIX}detection_rates.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out.append(p)

    # ── Fig 2: recovery of the Table 21 assignment as burden falls ────────────
    if t21rec_df is not None and not t21rec_df.empty:
        p = _burden_figure(
            plt, t21rec_df,
            scopes=[("all", "All scored signatures"), ("Sig3", "Sig3 (HRD)"),
                    ("Sig2", "Sig2 (APOBEC)")],
            title="Recovery of the Nik-Zainal per-tumour assignment as the "
                  "catalogue is thinned\n(truth: Supplementary Table 21 on "
                  "these same 560 genomes; the pooled panel excludes the two "
                  "signatures forced into every fit)",
            path=FIG_DIR / f"{FIG_PREFIX}burden_table21.png")
        out.append(p)

    # ── Fig 3: recovery of the whole-genome consensus at panel burden ─────────
    p = _burden_figure(
        plt, rec_df,
        scopes=[("all", "All breast signatures"), (SIG3, f"{pretty(SIG3)} (HRD)"),
                (APOBEC_PAIR[0], f"{pretty(APOBEC_PAIR[0])} (APOBEC)")],
        title="Recovery of the whole-genome signature calls as the catalogue is "
              "thinned to panel size\n(reference: signatures on which sigconfide "
              "and SPA agree at whole-genome depth)",
        path=FIG_DIR / f"{FIG_PREFIX}burden_consensus.png")
    out.append(p)

    # ── Fig 4: calls incompatible with breast biology (full panel) ────────────
    if not imp_df.empty:
        burden = {d: float(np.median(datasets[d].sum(axis=1))) for d in imp_df.dataset}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for meth in ("sigconfide", "spa"):
            sub = imp_df[imp_df.method == meth].copy()
            if sub.empty:
                continue
            sub["burden"] = sub.dataset.map(burden)
            sub = sub.sort_values("burden")
            axes[0].plot(sub.burden, sub.pct_samples_with_implausible * 100, "o-",
                         color=COLORS[meth], label=LABELS[meth])
            axes[1].plot(sub.burden, sub.mean_implausible_per_sample, "o-",
                         color=COLORS[meth], label=LABELS[meth])
        axes[0].set_ylabel("% samples with $\\geq$1 implausible signature")
        axes[1].set_ylabel("Implausible signatures per sample")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("Median mutations per sample")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        n_full = len(load_cosmic().columns)
        fig.suptitle(f"Calls incompatible with breast biology "
                     f"(unrestricted {n_full}-signature reference, no tissue "
                     f"knowledge)", fontsize=11)
        fig.tight_layout()
        p = FIG_DIR / f"{FIG_PREFIX}implausible.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out.append(p)

    # ── Fig 5: APOBEC coupling ────────────────────────────────────────────────
    a, b = APOBEC_PAIR
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, meth in zip(axes, ("sigconfide", "spa")):
        k = ("wgs", "breast", meth)
        if k not in exp_cache:
            continue
        e = exp_cache[k]
        ax.plot(e[a], e[b], ".", ms=4, color=COLORS[meth], alpha=0.6)
        ax.set_title(LABELS[meth], fontsize=10)
        ax.set_xlabel(f"{pretty(a)} exposure")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"{pretty(b)} exposure")
    fig.suptitle(f"APOBEC coupling: {pretty(a)} and {pretty(b)} arise from the "
                 f"same enzymes", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / f"{FIG_PREFIX}apobec.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out.append(p)

    # ── Fig 6: per-sample agreement with Table 21 ─────────────────────────────
    if conc_df is not None and not conc_df.empty:
        sub = conc_df[(conc_df.dataset == "wgs") & (conc_df.panel == "breast")]
        sigs = [s for s in REF_MAP if s in set(sub.table21_signature)]
        # the panel heading says what is being compared, the axis label names
        # the statistic on it; what each one *means* belongs in the caption
        panels = [
            ("mcc", "mcc_lo", "mcc_hi", "Detection agreement",
             "Matthews correlation coefficient"),
            ("exposure_rho", "exposure_rho_lo", "exposure_rho_hi",
             "Exposure agreement, all tumours", "Spearman $\\rho$"),
            ("exposure_rho_when_both", "exposure_rho_when_both_lo",
             "exposure_rho_when_both_hi",
             "Exposure agreement, co-detected only", "Spearman $\\rho$"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.42 * len(sigs) + 2.6),
                                 sharey=True)
        yy = np.arange(len(sigs))
        for ax, (metric, lo_c, hi_c, title, xlab) in zip(axes, panels):
            for k, meth in enumerate(("sigconfide", "spa")):
                m = sub[sub.method == meth].set_index("table21_signature")
                m = m.reindex(sigs)
                off = (0.5 - k) * 0.3
                v, lo, hi = m[metric].values, m[lo_c].values, m[hi_c].values
                ax.errorbar(v, yy + off,
                            xerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)],
                            fmt="o", ms=5, capsize=3, lw=1.2,
                            color=COLORS[meth], label=LABELS[meth])
                # a blank row is never "no data": say which reason applies
                for j in range(len(sigs)):
                    nb = int(m["n_both"].iloc[j])
                    if metric == "exposure_rho_when_both":
                        # co-detection count differs per method and drives how
                        # much this panel can be trusted, so label both
                        ax.annotate(f"n={nb}", (1.12, yy[j] + off), fontsize=8.5,
                                    color=COLORS[meth], va="center", ha="left")
                    if np.isfinite(v[j]):
                        continue
                    if metric == "mcc" and m["mandatory_forced"].iloc[j]:
                        note = "forced (called in all)"
                    elif metric == "exposure_rho_when_both":
                        note = f"too few ({nb})"
                    else:
                        continue
                    ax.annotate(note, (0.04, yy[j] + off), fontsize=8.5,
                                color="#888", va="center")
            ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlab, fontsize=10)
            ax.tick_params(labelsize=10)
            # the third panel reserves right-hand margin for the n labels, but
            # a correlation axis must not appear to run past 1
            ax.set_xlim(-0.35,
                        1.34 if metric == "exposure_rho_when_both" else 1.08)
            ax.set_xticks(np.arange(0.0, 1.01, 0.2))
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(yy)
        axes[0].set_yticklabels([
            f"{pretty_ref(s)}\n({'+'.join(pretty(m) for m in REF_MAP[s])})"
            if REF_MAP[s] != [s.replace(" ", "_")] else pretty_ref(s)
            for s in sigs], fontsize=9.5)
        # the left strip of the middle panel is empty at every row (all rho are
        # positive), so the legend goes there rather than over a data row
        axes[1].legend(frameon=False, fontsize=9.5, loc="center left")
        fig.suptitle("Per-sample agreement with Nik-Zainal 2016 Table 21 "
                     "on the same 560 genomes\n"
                     "(95% CI: paired bootstrap for MCC, Fisher-z for $\\rho$; "
                     "n = tumours where that method and Table 21 both "
                     "call the signature)",
                     fontsize=12)
        fig.tight_layout()
        p = FIG_DIR / f"{FIG_PREFIX}table21_concordance.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out.append(p)

    print("\nFigures written:")
    for p in out:
        print(f"  {p}")


# ── LaTeX tables ───────────────────────────────────────────────────────────────
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def _ci(v, lo, hi, nd=2, signed=False):
    """"0.68 [0.63, 0.73]", degrading to the point estimate if no interval."""
    if v is None or not np.isfinite(v):
        return "--"
    fmt = f"{{:+.{nd}f}}" if signed else f"{{:.{nd}f}}"
    if lo is None or not np.isfinite(lo) or not np.isfinite(hi):
        return fmt.format(v)
    return f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _pct_ci(v, lo, hi, nd=1):
    """A rate given as a fraction, rendered as "45.9\\% [41.8, 50.1]"."""
    if v is None or not np.isfinite(v):
        return "--"
    if lo is None or not np.isfinite(lo):
        return f"{v*100:.{nd}f}\\%"
    return f"{v*100:.{nd}f}\\% [{lo*100:.{nd}f}, {hi*100:.{nd}f}]"


def _p_str(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def make_latex(det_df, imp_df, apo_df, rec_df, uni_df=None, conc_df=None,
               t21rec_df=None):
    """Manuscript tables for the current catalogue.

    The remaining numbers (APOBEC discordance, implausible calls) are short
    enough to live in the prose, so they are printed to the console instead.
    """
    lines = []
    A = lines.append
    A("% Auto-generated by validate_brca_biology.py "
      f"--catalog {CATALOG} -- biological validation on the")
    A("% 560 ICGC-BRCA (Nik-Zainal 2016) breast whole genomes.  The manuscript")
    A("% wants a single .tex, so paste these in rather than \\input-ing this file.")
    A("")

    t = det_df[(det_df.dataset == "wgs") & (det_df.panel == "breast")]
    piv = t.pivot_table(index="signature", columns="method", values="detection_rate")
    have_both = not piv.empty and {"sigconfide", "spa"} <= set(piv.columns)

    if have_both and not EXPECTED_PREVALENCE and conc_df is not None \
            and not conc_df.empty:
        # ── same dictionary as the reference: paired, per-tumour table ────────
        sub = conc_df[(conc_df.dataset == "wgs") & (conc_df.panel == "breast")]
        A("\\begin{table}[!t]")
        A("\\centering")
        A("\\caption{Agreement with the Nik-Zainal Supplementary Table~21 "
          "assignment of the same 560 breast whole genomes, fitted in the "
          "dictionary that assignment was built in (COSMIC~v2, "
          f"{len(BREAST_SIGS)}-signature tissue panel). \\emph{{Called}} is the "
          "fraction of tumours in which the method reports the signature "
          "(Wilson 95\\% interval), $\\Delta$ the difference from the fraction "
          "Table~21 reports, in percentage points, and $p$ McNemar's exact test "
          "on the discordant tumours---which here is also the exact test of "
          "that difference, because reference and fit share a dictionary and a "
          "cohort. MCC scores \\emph{which} tumours rather than how many, and "
          "$\\rho$ is the Spearman correlation of relative exposure across all "
          "tumours. Intervals for $\\Delta$ and MCC come from a paired "
          "bootstrap over tumours (2000 resamples), and for $\\rho$ from a "
          "Fisher $z$ transform with the Bonett--Wright standard error. "
          "$\\dagger$ marks the signatures mandatory in \\textsc{sigconfide}, "
          "which are called in almost every tumour, so their MCC carries no "
          "information.}")
        A("\\label{tab:brca_concordance}")
        A("\\small")
        A("\\setlength{\\tabcolsep}{4pt}")
        A("\\begin{tabular}{l l l l l l c}")
        A("\\toprule")
        A("Signature & Method & Called & $\\Delta$ (pp) & MCC & $\\rho$ & "
          "$p$ \\\\")
        A("\\midrule")
        det_w = det_df[(det_df.dataset == "wgs") & (det_df.panel == "breast")]
        for i, v2 in enumerate(REF_MAP):
            r = sub[sub.table21_signature == v2]
            if r.empty:
                continue
            if i:
                A("\\addlinespace[1.5pt]")
            forced = bool(r.iloc[0]["mandatory_forced"])
            label = pretty_ref(v2) + ("$^{\\dagger}$" if forced else "")
            for j, meth in enumerate(("sigconfide", "spa")):
                rm = r[r.method == meth]
                if rm.empty:
                    continue
                row = rm.iloc[0]
                # the Wilson interval belongs to the fitted rate, so read it off
                # the detection table rather than recomputing it here
                members = REF_MAP[v2]
                dw = det_w[(det_w.method == meth) & (det_w.signature.isin(members))]
                rate = row["n_pred_pos"] / row["n"]
                if len(dw) == 1:
                    called = _pct_ci(rate, dw.iloc[0]["rate_lo"],
                                     dw.iloc[0]["rate_hi"])
                else:  # a union of split members: interval from the union rate
                    lo, hi = _wilson(int(row["n_pred_pos"]), int(row["n"]))
                    called = _pct_ci(rate, lo, hi)
                mcc = ("--" if not np.isfinite(row["mcc"])
                       else _ci(row["mcc"], row["mcc_lo"], row["mcc_hi"]))
                cells = [
                    label if j == 0 else "",
                    "\\textsc{sigconfide}" if meth == "sigconfide" else "SPA",
                    called,
                    _ci(row["prevalence_diff"] * 100,
                        row["prevalence_diff_lo"] * 100,
                        row["prevalence_diff_hi"] * 100, nd=1, signed=True),
                    mcc,
                    _ci(row["exposure_rho"], row["exposure_rho_lo"],
                        row["exposure_rho_hi"]),
                    _p_str(row["mcnemar_p"]),
                ]
                A(" & ".join(cells) + " \\\\")
        A("\\bottomrule")
        A("\\end{tabular}")
        A("\\end{table}")
        A("")

    elif have_both:
        # ── cross-dictionary arm: prevalence bands ───────────────────────────
        A("\\begin{table}[h]")
        A("\\centering")
        A("\\caption{Signature prevalence across the 560 breast whole genomes "
          f"(tissue-informed {len(BREAST_SIGS)}-signature panel) against the "
          "prevalence expected from Nik-Zainal Supplementary Table~21; "
          "\\checkmark marks a rate inside the expected band. Table~21 predates "
          "the v2$\\rightarrow$v3 splits of Signature~5 into SBS5/SBS40 and "
          "Signature~17 into SBS17a/SBS17b, so it pins only the union of each "
          "pair and never the individual members; those four rows therefore "
          "carry no band of their own (--) and the two unions are checked "
          "separately below.}")
        A("\\label{tab:brca_detection}")
        A("\\begin{tabular}{l c cc cc}")
        A("\\toprule")
        A("Signature & Expected & \\multicolumn{2}{c}{\\textsc{sigconfide}} & "
          "\\multicolumn{2}{c}{SPA} \\\\")
        A("\\midrule")
        for sig in [s for s in BREAST_SIGS if s in piv.index]:
            lo, hi, _ = EXPECTED_PREVALENCE.get(sig, (np.nan, np.nan, ""))
            band = f"{lo*100:.0f}--{hi*100:.0f}\\%" if np.isfinite(lo) else "--"
            cells = []
            for meth in ("sigconfide", "spa"):
                r = piv.loc[sig, meth]
                ok = "\\checkmark" if np.isfinite(lo) and lo <= r <= hi else ""
                cells.append(f"{r*100:.1f}\\% & {ok}")
            A(f"{sig} & {band} & " + " & ".join(cells) + " \\\\")
        if uni_df is not None and not uni_df.empty:
            u = uni_df[(uni_df.dataset == "wgs") & (uni_df.panel == "breast")]
            if not u.empty:
                A("\\midrule")
                for pair in EXPECTED_UNION_PREVALENCE:
                    r = u[u.pair == "+".join(pair)]
                    if r.empty:
                        continue
                    lo = r.iloc[0]["expected_lo"]
                    hi = r.iloc[0]["expected_hi"]
                    cells = []
                    for meth in ("sigconfide", "spa"):
                        rm = r[r.method == meth]
                        if rm.empty:
                            cells += ["--", ""]
                            continue
                        rate = rm.iloc[0]["detection_rate"]
                        ok = "\\checkmark" if lo <= rate <= hi else ""
                        cells.append(f"{rate*100:.1f}\\% & {ok}")
                    label = "\\,$\\cup$\\,".join(pair)
                    A(f"{label} & {lo*100:.0f}--{hi*100:.0f}\\% & "
                      + " & ".join(cells) + " \\\\")

        A("\\midrule")
        n_ok = t.groupby("method")["in_expected_band"].sum()
        n_uni = {}
        n_pairs = 0
        if uni_df is not None and not uni_df.empty:
            u = uni_df[(uni_df.dataset == "wgs") & (uni_df.panel == "breast")]
            n_uni = u.groupby("method")["in_expected_band"].sum().to_dict()
            n_pairs = u["pair"].nunique()

        def _score(meth):
            s = f"{int(n_ok.get(meth, 0))}/{N_BANDED}"
            if n_pairs:
                s += f" $+$ {int(n_uni.get(meth, 0))}/{n_pairs}"
            return s

        A("Inside band (single $+$ union) & & \\multicolumn{2}{c}{"
          f"{_score('sigconfide')}" + "} & \\multicolumn{2}{c}{"
          f"{_score('spa')}" + "} \\\\")
        A("\\bottomrule")
        A("\\end{tabular}")
        A("\\end{table}")
        A("")

    # ── burden table, scored against Table 21 ────────────────────────────────
    if t21rec_df is not None and not t21rec_df.empty:
        sub = t21rec_df[t21rec_df.scope == "all"]
        sigma = t21rec_df[(t21rec_df.method == "SigMA")
                          & (t21rec_df.scope == "Sig3")]
        sigma_note = ""
        if not sigma.empty:
            r = sigma.iloc[0]
            sm_sens = _ci(r["sensitivity"], r["sensitivity_lo"],
                          r["sensitivity_hi"], nd=3)
            sm_spec = _ci(r["specificity"], r["specificity_lo"],
                          r["specificity_hi"], nd=3)
            sigma_note = (" For reference, SigMA, a detector built specifically "
                          "for Signature~3 in panel data, reaches "
                          f"{sm_sens} sensitivity at {sm_spec} specificity for "
                          "Signature~3 on the same panel simulation.")
        A("\\begin{table}[!t]")
        A("\\centering")
        A("\\caption{Recovery of the Nik-Zainal Table~21 per-tumour assignment "
          "as the same genomes are thinned to panel-sized catalogues. Truth is "
          "external to both methods: a signature is present in a tumour when "
          "Table~21 assigns it there. Pooled over the scored signatures except "
          "the two forced into every \\textsc{sigconfide} fit. Brackets give "
          "95\\% intervals from a bootstrap over tumours (2000 resamples), "
          "which resamples each tumour with all of its signature calls "
          "attached." + sigma_note + "}")
        A("\\label{tab:brca_burden}")
        A("\\small")
        A("\\setlength{\\tabcolsep}{4.5pt}")
        A("\\begin{tabular}{l l l l l}")
        A("\\toprule")
        A("Median mutations & Method & Sensitivity & Specificity & MCC \\\\")
        A("\\midrule")
        for i, burden in enumerate(sorted(sub.median_mut.unique(), reverse=True)):
            row = sub[sub.median_mut == burden]
            label = f"{burden:.0f}"
            if (row.dataset == "panel").any():
                label += " (panel)"
            elif (row.dataset == "wgs").any():
                label += " (WGS)"
            if i:
                A("\\addlinespace[1.5pt]")
            for j, meth in enumerate(("sigconfide", "spa")):
                r = row[row.method == meth]
                if r.empty:
                    continue
                r = r.iloc[0]
                A(" & ".join([
                    label if j == 0 else "",
                    "\\textsc{sigconfide}" if meth == "sigconfide" else "SPA",
                    _ci(r["sensitivity"], r["sensitivity_lo"], r["sensitivity_hi"],
                        nd=3),
                    _ci(r["specificity"], r["specificity_lo"], r["specificity_hi"],
                        nd=3),
                    _ci(r["mcc"], r["mcc_lo"], r["mcc_hi"]),
                ]) + " \\\\")
        A("\\bottomrule")
        A("\\end{tabular}")
        A("\\end{table}")
        A("")

    # ── tissue specificity on the unrestricted reference ─────────────────────
    if not imp_df.empty:
        n_full = len(load_cosmic().columns)
        A("\\begin{table}[!t]")
        A("\\centering")
        A("\\caption{Behaviour on the unrestricted "
          f"{n_full}-signature reference, where no tissue knowledge is supplied. "
          "A call is counted implausible when its aetiology cannot operate in a "
          "treatment-naive primary breast tumour (ultraviolet light, tobacco, "
          "aflatoxin, chemotherapy and the like). Brackets give Wilson 95\\% "
          "intervals on the percentage of tumours affected.}")
        A("\\label{tab:brca_implausible}")
        A("\\small")
        A("\\begin{tabular}{l cl cl}")
        A("\\toprule")
        A("Median & \\multicolumn{2}{c}{\\textsc{sigconfide}} & "
          "\\multicolumn{2}{c}{SPA} \\\\")
        A("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
        A("mutations & Sigs/sample & \\% implausible & Sigs/sample & "
          "\\% implausible \\\\")
        A("\\midrule")
        for burden in sorted(imp_df.median_mut.unique(), reverse=True):
            row = imp_df[imp_df.median_mut == burden]
            label = f"{burden:.0f}"
            if (row.dataset == "panel").any():
                label += " (panel)"
            elif (row.dataset == "wgs").any():
                label += " (WGS)"
            cells = []
            for meth in ("sigconfide", "spa"):
                r = row[row.method == meth]
                if r.empty:
                    cells += ["--", "--"]
                else:
                    cells += [_fmt(r.iloc[0]["mean_sigs_per_sample"], 1),
                              _pct_ci(r.iloc[0]["pct_samples_with_implausible"],
                                      r.iloc[0]["pct_implausible_lo"],
                                      r.iloc[0]["pct_implausible_hi"])]
            A(f"{label} & " + " & ".join(cells) + " \\\\")
        A("\\bottomrule")
        A("\\end{tabular}")
        A("\\end{table}")

    body = "\n".join(lines) + "\n"
    name = f"{FIG_PREFIX}validation_tables.tex"
    paths = [OUT_DIR / name, FIG_DIR / name]
    for path in paths:
        path.write_text(body)
    print("\nLaTeX tables → " + ", ".join(str(p) for p in paths))


def _print_report(det_df, imp_df, apo_df, rec_df, integ_df, uni_df=None,
                  conc_df=None, prof_df=None, t21rec_df=None):
    pd.set_option("display.width", 220)

    print("\n" + "=" * 78)
    print(f"CATALOGUE: {CATALOGS[CATALOG]['label']}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("1. Prevalence vs the Table 21 assignment (WGS, tissue-informed panel)")
    print("=" * 78)
    t = det_df[(det_df.dataset == "wgs") & (det_df.panel == "breast")]
    piv = t.pivot_table(index="signature", columns="method", values="detection_rate")
    piv = (piv * 100).reindex([s for s in BREAST_SIGS if s in piv.index])
    band = t.drop_duplicates("signature").set_index("signature")
    piv["table21"] = [f"{reference_rate(s)*100:.1f}%" if np.isfinite(reference_rate(s))
                      else "-" for s in piv.index]
    ci = t.drop_duplicates(["signature", "method"]).set_index(["signature", "method"])
    for meth in [m for m in ("sigconfide", "spa") if m in piv.columns]:
        piv[f"{meth} 95% CI"] = [
            (f"[{ci.loc[(sig, meth), 'rate_lo']*100:.1f}, "
             f"{ci.loc[(sig, meth), 'rate_hi']*100:.1f}]")
            if (sig, meth) in ci.index else "-" for sig in piv.index]
    if EXPECTED_PREVALENCE:
        piv["expected"] = [
            (f"{band.loc[s,'expected_lo']*100:.0f}-{band.loc[s,'expected_hi']*100:.0f}%"
             if s in band.index and np.isfinite(band.loc[s, "expected_lo"]) else "-")
            for s in piv.index
        ]
    print(piv.round(1).to_string())
    if EXPECTED_PREVALENCE:
        print("\nsignatures inside the expected band:",
              t.groupby("method")["in_expected_band"].sum().to_dict(),
              f"of {N_BANDED} individually constrained "
              f"({len(EXPECTED_UNION_PREVALENCE)} split pairs checked as unions)")

    if uni_df is not None and not uni_df.empty:
        sub = uni_df[(uni_df.dataset == "wgs") & (uni_df.panel == "breast")]
        print("\nsplit pairs, checked as unions:")
        print(sub[["pair", "method", "detection_rate", "expected_lo",
                   "expected_hi", "in_expected_band"]].round(3)
              .to_string(index=False))

    if conc_df is not None and not conc_df.empty:
        print("\n" + "=" * 78)
        print("1b. Per-sample agreement with Nik-Zainal Table 21 "
              "(same 560 tumours)")
        print("=" * 78)
        sub = conc_df[(conc_df.dataset == "wgs") & (conc_df.panel == "breast")]
        show = sub.assign(
            MCC=[("forced" if r.mandatory_forced and not np.isfinite(r.mcc)
                  else f"{r.mcc:.2f} [{r.mcc_lo:.2f},{r.mcc_hi:.2f}]")
                 for r in sub.itertuples()],
            Jaccard=[f"{r.jaccard:.2f} [{r.jaccard_lo:.2f},{r.jaccard_hi:.2f}]"
                     for r in sub.itertuples()],
            dPrev=[f"{r.prevalence_diff:+.3f} "
                   f"[{r.prevalence_diff_lo:+.3f},{r.prevalence_diff_hi:+.3f}]"
                   for r in sub.itertuples()],
            rho=[f"{r.exposure_rho:.2f} "
                 f"[{r.exposure_rho_lo:.2f},{r.exposure_rho_hi:.2f}]"
                 for r in sub.itertuples()],
        )
        print(show[["table21_signature", "fitted_signatures", "method",
                    "n_truth_pos", "n_pred_pos", "sensitivity", "specificity",
                    "MCC", "Jaccard", "dPrev", "mcnemar_p", "rho"]]
              .round(3).to_string(index=False))
        print("\n[95% CIs: paired bootstrap for MCC/Jaccard/dPrev, "
              "Fisher-z (Bonett-Wright) for rho; mcnemar_p tests whether the "
              "discordant calls are one-sided]")
        if sub.mandatory_forced.any():
            print(f"[MCC 'forced': {'/'.join(MANDATORY)} are mandatory in the "
                  f"fit, so they are called in every tumour and MCC is "
                  f"undefined; read exposure rho instead]")

    if prof_df is not None and not prof_df.empty:
        print("\nper-sample exposure-profile cosine vs Table 21:")
        print(prof_df[prof_df.panel == "breast"]
              [["dataset", "method", "n", "cosine_median", "cosine_q25",
                "cosine_q75", "cosine_frac_above_0_9", "mean_sigs_called",
                "mean_sigs_table21"]].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("2. APOBEC coherence (the two APOBEC signatures share an aetiology)")
    print("=" * 78)
    sub = apo_df[(apo_df.panel == "breast")].sort_values(["method", "dataset"])
    print(sub.round(3).to_string(index=False))

    if t21rec_df is not None and not t21rec_df.empty:
        print("\n" + "=" * 78)
        print("3. Recovery of the Table 21 assignment at falling burden "
              "(external truth)")
        print("=" * 78)
        for scope in ("all", "Sig3", "Sig2", "Sig13", "Sig8"):
            sub = t21rec_df[t21rec_df.scope == scope].sort_values(
                ["method", "median_mut"])
            if sub.empty:
                continue
            print(f"\n-- {scope} --")
            show = sub.assign(
                Sens=[_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi, 3)
                      for r in sub.itertuples()],
                Spec=[_ci(r.specificity, r.specificity_lo, r.specificity_hi, 3)
                      for r in sub.itertuples()],
                MCC=[_ci(r.mcc, r.mcc_lo, r.mcc_hi) for r in sub.itertuples()],
            )
            print(show[["dataset", "method", "median_mut", "n_positive",
                        "Sens", "Spec", "MCC", "ppv", "f1"]]
                  .round(3).to_string(index=False))
        print("\n[95% CIs: bootstrap over tumours, each resampled with all of "
              "its signature calls]")

    print("\n" + "=" * 78)
    print("4. Recovery of whole-genome calls at panel burden (consensus reference)")
    print("=" * 78)
    for scope in ("all", SIG3, APOBEC_PAIR[0]):
        sub = rec_df[rec_df.scope == scope].sort_values(["method", "median_mut"])
        if sub.empty:
            continue
        print(f"\n-- {scope} --")
        print(sub[["dataset", "method", "median_mut", "n_positive", "sensitivity",
                   "specificity", "ppv", "f1"]].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("5. Tissue-implausible calls (unrestricted reference)")
    print("=" * 78)
    print(imp_df.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("6. HRDetect label integrity check")
    print("=" * 78)
    print(integ_df.round(3).to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", choices=list(CATALOGS), default="v2",
                   help="reference dictionary (default: v2, native to Table 21)")
    p.add_argument("--stage", choices=["fit", "report", "all"], default="all")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="wgs panel ds03 ds06 ... (default: all)")
    p.add_argument("--panels", nargs="+", default=["breast", "full"],
                   choices=["breast", "full"])
    p.add_argument("--methods", nargs="+", default=["sigconfide", "spa"],
                   choices=["sigconfide", "spa"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--append", action="store_true",
                   help="merge with existing assignments.csv instead of overwriting")
    p.add_argument("--no-figures", action="store_true",
                   help="skip figure generation in the report stage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure(args.catalog)
    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("report", "all"):
        stage_report(args)
