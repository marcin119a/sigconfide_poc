"""Batch interface: TSV catalogues in, activities TSV out."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.selection import exposure_intervals, hybrid_stepwise_selection

OUTPUT_NAME = "Assignment_Solution_Activities.txt"
INTERVALS_NAME = "Assignment_Solution_Intervals.txt"
INTERVAL_COLUMNS = [
    "Samples",
    "Signature",
    "Exposure",
    "Lower",
    "Upper",
    "SD",
    "sigma_b",
]


def _fit_sample(args):
    m, P, kwargs, intervals, R, overdispersion, level = args
    n = m.sum()
    if n <= 0:
        return np.zeros(P.shape[1]), []
    sel_idx, exposures, _ = hybrid_stepwise_selection(m, P, **kwargs)
    dense = np.zeros(P.shape[1])
    dense[sel_idx] = exposures
    rows = []
    if intervals:
        ci = exposure_intervals(
            m,
            P,
            sel_idx,
            B=R,
            overdispersion=overdispersion,
            level=level,
            decomposition_method=kwargs.get("decomposition_method", decomposeQP),
        )
        rows = [
            (
                int(j),
                float(e * n),
                float(lo * n),
                float(hi * n),
                float(sd * n),
                float(ci["sigma_b"]),
            )
            for j, e, lo, hi, sd in zip(
                sel_idx, exposures, ci["lower"], ci["upper"], ci["sd"]
            )
        ]
    return dense * n, rows


def fit_spa_style(
    samples_tsv,
    signature_db_tsv,
    output_dir,
    R=100,
    n_jobs=None,
    intervals=False,
    overdispersion="auto",
    level=0.90,
    **selection_kwargs,
) -> pd.DataFrame:
    """Fit every sample and write ``<output_dir>/Assignment_Solution_Activities.txt``.

    Both TSVs have mutation contexts as rows; samples are columns of
    ``samples_tsv`` and signatures are columns of ``signature_db_tsv``. The
    output has samples as rows, signatures as columns and absolute mutation
    counts as values (0 for unassigned signatures). Keyword arguments go to
    ``hybrid_stepwise_selection``.

    With ``intervals=True`` a second file, ``Assignment_Solution_Intervals.txt``,
    holds one row per assigned signature with the bootstrap interval of its
    exposure (``exposure_intervals`` with ``R`` replicates, the over-dispersion
    ``overdispersion`` and the central ``level`` share), in mutation counts.
    """
    samples = pd.read_csv(samples_tsv, sep="\t", index_col=0)
    panel = pd.read_csv(signature_db_tsv, sep="\t", index_col=0)

    missing = panel.index.difference(samples.index)
    if len(missing):
        raise ValueError(f"{len(missing)} signature contexts missing from samples")
    samples = samples.loc[panel.index]

    P = panel.values.astype(float)
    tasks = [
        (
            samples[s].values.astype(float),
            P,
            selection_kwargs,
            intervals,
            R,
            overdispersion,
            level,
        )
        for s in samples.columns
    ]
    if n_jobs == 1:
        results = [_fit_sample(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_fit_sample, tasks))

    activities = pd.DataFrame(
        np.vstack([r[0] for r in results]), index=samples.columns, columns=panel.columns
    )
    activities.index.name = "Samples"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    activities.to_csv(out / OUTPUT_NAME, sep="\t")
    if intervals:
        rows = [
            (s, panel.columns[j], e, lo, hi, sd, sigma)
            for s, (_, ci) in zip(samples.columns, results)
            for j, e, lo, hi, sd, sigma in ci
        ]
        pd.DataFrame(rows, columns=INTERVAL_COLUMNS).to_csv(
            out / INTERVALS_NAME, sep="\t", index=False
        )
    return activities


def read_activities(path) -> pd.DataFrame:
    """Read an ``Assignment_Solution_Activities.txt`` ."""
    return pd.read_csv(path, sep="\t", index_col=0, dtype={0: str})


def read_intervals(path) -> pd.DataFrame:
    """Read an ``Assignment_Solution_Intervals.txt``."""
    return pd.read_csv(path, sep="\t", dtype={0: str})
