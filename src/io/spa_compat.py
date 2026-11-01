"""Batch interface: TSV catalogues in, activities TSV out."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sigconfide.estimates.selection import hybrid_stepwise_selection

OUTPUT_NAME = "Assignment_Solution_Activities.txt"


def _fit_sample(args):
    m, P, R, kwargs = args
    if m.sum() <= 0:
        return np.zeros(P.shape[1])
    sel_idx, exposures, _ = hybrid_stepwise_selection(m, P, R=R, **kwargs)
    dense = np.zeros(P.shape[1])
    dense[sel_idx] = exposures
    return dense * m.sum()


def fit_spa_style(
    samples_tsv,
    signature_db_tsv,
    output_dir,
    R=100,
    n_jobs=None,
    **selection_kwargs,
) -> pd.DataFrame:
    """Fit every sample and write ``<output_dir>/Assignment_Solution_Activities.txt``.

    Both TSVs have mutation contexts as rows; samples are columns of
    ``samples_tsv`` and signatures are columns of ``signature_db_tsv``. The
    output has samples as rows, signatures as columns and absolute mutation
    counts as values (0 for unassigned signatures).
    """
    samples = pd.read_csv(samples_tsv, sep="\t", index_col=0)
    panel = pd.read_csv(signature_db_tsv, sep="\t", index_col=0)

    missing = panel.index.difference(samples.index)
    if len(missing):
        raise ValueError(f"{len(missing)} signature contexts missing from samples")
    samples = samples.loc[panel.index]

    P = panel.values.astype(float)
    tasks = [
        (samples[s].values.astype(float), P, R, selection_kwargs)
        for s in samples.columns
    ]
    if n_jobs == 1:
        rows = [_fit_sample(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_fit_sample, tasks))

    activities = pd.DataFrame(
        np.vstack(rows), index=samples.columns, columns=panel.columns
    )
    activities.index.name = "Samples"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    activities.to_csv(out / OUTPUT_NAME, sep="\t")
    return activities


def read_activities(path) -> pd.DataFrame:
    """Read an ``Assignment_Solution_Activities.txt`` ."""
    return pd.read_csv(path, sep="\t", index_col=0, dtype={0: str})
