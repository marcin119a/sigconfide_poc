import numpy as np
import pytest

from sigconfide.decompose.qp import decomposeQP, decomposeQP_batch
from sigconfide.estimates.standard import findSigExposures


def _panel(rng, contexts, n_sigs):
    P = rng.random((contexts, n_sigs)) ** 3
    return P / P.sum(axis=0)


def _profiles(rng, P, n_profiles):
    weights = rng.dirichlet(np.ones(P.shape[1]) * 0.3, size=n_profiles).T
    counts = rng.poisson(2000 * (P @ weights))
    return counts / counts.sum(axis=0)


@pytest.mark.parametrize("n_sigs", [5, 30, 120])  # 120 > 96: rank-deficient panel
def test_batch_matches_column_by_column(n_sigs):
    rng = np.random.default_rng(0)
    P = _panel(rng, 96, n_sigs)
    M = _profiles(rng, P, 20)

    expected = np.column_stack([decomposeQP(M[:, j], P) for j in range(M.shape[1])])
    got = decomposeQP_batch(M, P)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_batch_single_column():
    rng = np.random.default_rng(1)
    P = _panel(rng, 96, 10)
    M = _profiles(rng, P, 1)
    np.testing.assert_allclose(
        decomposeQP_batch(M, P)[:, 0], decomposeQP(M[:, 0], P), atol=1e-10
    )


def test_batch_does_not_modify_inputs():
    rng = np.random.default_rng(2)
    P = _panel(rng, 96, 10)
    M = _profiles(rng, P, 5)
    P0, M0 = P.copy(), M.copy()
    decomposeQP_batch(M, P)
    np.testing.assert_array_equal(P, P0)
    np.testing.assert_array_equal(M, M0)


def test_find_sig_exposures_default_matches_custom_solver():
    """The default solver takes the batch path; a wrapped one takes the old path."""
    rng = np.random.default_rng(3)
    P = _panel(rng, 96, 12)
    M = _profiles(rng, P, 8)

    def wrapped(m, P):
        return decomposeQP(m, P)

    fast, fast_err = findSigExposures(M, P)
    slow, slow_err = findSigExposures(M, P, decomposition_method=wrapped)

    np.testing.assert_allclose(fast, slow, atol=1e-10)
    np.testing.assert_allclose(fast_err, slow_err, atol=1e-10)
