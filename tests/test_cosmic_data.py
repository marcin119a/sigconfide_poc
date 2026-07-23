"""End-to-end tests on the real COSMIC v3.6 SBS signature matrix.

The file ``tests/data/COSMIC_v3.6_SBS_GRCh37.txt`` is a 96 x 101 panel: 96
mutation contexts (rows) and 101 signatures (columns, each summing to 1).
Because there are more signatures than contexts the matrix is rank-deficient,
which used to break ``decomposeQP`` (``quadprog`` needs a positive-definite
Gram matrix).  These tests exercise the public methods directly on that panel
and assert that a known signature mixture is recovered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.selection import hybrid_stepwise_selection
from sigconfide.estimates.standard import findSigExposures

DATA_FILE = Path(__file__).parent / "data" / "COSMIC_v3.6_SBS_GRCh37.txt"


@pytest.fixture(scope="module")
def cosmic():
    """The full COSMIC v3.6 signature matrix as a DataFrame (96 x 101)."""
    return pd.read_csv(DATA_FILE, sep="\t", index_col=0)


@pytest.fixture(scope="module")
def P(cosmic):
    """The signature matrix as a plain float array (96 contexts x 101 sigs)."""
    return cosmic.to_numpy()


def _mixture_counts(cosmic, weights, total=10000):
    """A count profile that is a known convex mix of named signatures."""
    probs = np.zeros(cosmic.shape[0])
    for sig, w in weights.items():
        probs += w * cosmic[sig].to_numpy()
    return np.round(probs * total)


class TestFindSigExposuresOnCosmic:
    def test_full_panel_does_not_crash(self, cosmic, P):
        # Regression: the full 96x101 panel is rank-deficient; decomposeQP must
        # still solve it (fixed by ridge-regularising the Gram matrix).
        m = _mixture_counts(cosmic, {"SBS1": 0.3, "SBS5": 0.3, "SBS4": 0.4})
        exposures, errors = findSigExposures(m.reshape(-1, 1), P)
        assert exposures.shape == (P.shape[1], 1)
        assert exposures.sum() == pytest.approx(1.0)
        assert errors[0] < 1e-2

    def test_recovers_known_mixture(self, cosmic, P):
        sigs = list(cosmic.columns)
        weights = {"SBS1": 0.3, "SBS5": 0.3, "SBS4": 0.4}
        m = _mixture_counts(cosmic, weights)
        exposures, _ = findSigExposures(m.reshape(-1, 1), P)
        exp = exposures[:, 0]
        for sig, w in weights.items():
            assert exp[sigs.index(sig)] == pytest.approx(w, abs=0.02)


class TestDecomposeQPRankDeficient:
    def test_ridge_handles_more_sigs_than_contexts(self, cosmic, P):
        # Directly exercise the solver on the rank-deficient panel.
        m = _mixture_counts(cosmic, {"SBS2": 0.5, "SBS13": 0.5})
        m = m / m.sum()
        exposures = decomposeQP(m, P)
        assert exposures.shape == (P.shape[1],)
        assert np.all(exposures >= 0)
        assert exposures.sum() == pytest.approx(1.0)


class TestHybridSelectionOnCosmic:
    def test_prefilter_recovers_true_signatures(self, cosmic, P):
        sigs = list(cosmic.columns)
        m = _mixture_counts(cosmic, {"SBS1": 0.3, "SBS5": 0.3, "SBS4": 0.4})
        np.random.seed(0)
        sel_idx, exposures, errors = hybrid_stepwise_selection(
            m, P, R=20, pre_filter_threshold=0.001
        )
        selected = {sigs[i] for i in sel_idx}
        assert {"SBS1", "SBS4", "SBS5"} <= selected
        assert exposures.sum() == pytest.approx(1.0)
        assert exposures.shape[0] == len(sel_idx)
