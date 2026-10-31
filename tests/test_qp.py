import numpy as np
import pytest
from sigconfide.decompose.qp import decomposeQP


class TestDecomposeQP:
    def test_output_shape(self, simple_P, m_from_P):
        exposures = decomposeQP(m_from_P, simple_P)
        assert exposures.shape == (simple_P.shape[1],)

    def test_exposures_sum_to_one(self, simple_P, m_from_P):
        exposures = decomposeQP(m_from_P, simple_P)
        assert exposures.sum() == pytest.approx(1.0)

    def test_exposures_non_negative(self, simple_P, m_from_P):
        exposures = decomposeQP(m_from_P, simple_P)
        assert np.all(exposures >= 0.0)

    def test_recovers_known_weights(self, simple_P, known_weights, m_from_P):
        # m is an exact convex combination of the columns of P, so the QP
        # solution should recover those weights.
        exposures = decomposeQP(m_from_P, simple_P)
        assert exposures == pytest.approx(known_weights, abs=1e-6)

    def test_pure_signature_is_identified(self, simple_P):
        # If m equals a single column of P, the solver should put ~all weight
        # on that signature.
        for j in range(simple_P.shape[1]):
            exposures = decomposeQP(simple_P[:, j], simple_P)
            expected = np.zeros(simple_P.shape[1])
            expected[j] = 1.0
            assert exposures == pytest.approx(expected, abs=1e-6)

    def test_two_signature_mixture(self, simple_P):
        m = 0.3 * simple_P[:, 0] + 0.7 * simple_P[:, 1]
        exposures = decomposeQP(m, simple_P)
        assert exposures == pytest.approx([0.3, 0.7, 0.0], abs=1e-6)


def test_unused_signatures_are_exact_zeros():
    """quadprog leaves inactive variables as 1e-24..1e-12 noise; the wrapper
    must return exact zeros so that a positive exposure means 'used'."""
    import numpy as np
    from sigconfide.decompose.qp import ZERO_TOL, decomposeQP, decomposeQP_batch

    rng = np.random.default_rng(0)
    K, N = 96, 30
    P = rng.random((K, N)); P /= P.sum(axis=0)
    truth = np.zeros(N); truth[[0, 5, 17]] = [0.6, 0.3, 0.1]
    m = P @ truth
    x = decomposeQP(m, P)
    assert np.all((x == 0) | (x >= ZERO_TOL))
    assert (x == 0).sum() >= N - 6
    M = rng.poisson(5000 * m[:, None] * np.ones((1, 20))).astype(float)
    M /= M.sum(axis=0)
    E = decomposeQP_batch(M, P)
    assert np.all((E == 0) | (E >= ZERO_TOL))
    assert np.allclose(E.sum(axis=0), 1.0)
