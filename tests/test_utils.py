import numpy as np
import pytest

from sigconfide.utils.utils import FrobeniusNorm, is_wholenumber


class TestIsWholeNumber:
    @pytest.mark.parametrize("value", [0, 1, 3, 3.0, -5.0, 1000000])
    def test_whole_numbers(self, value):
        assert is_wholenumber(value)

    @pytest.mark.parametrize("value", [3.4, 0.5, -2.1, 1e-3])
    def test_non_whole_numbers(self, value):
        assert not is_wholenumber(value)

    def test_within_tolerance(self):
        assert is_wholenumber(3.0 + 1e-10)

    def test_outside_tolerance(self):
        assert not is_wholenumber(3.0 + 1e-8)

    def test_custom_tolerance(self):
        assert is_wholenumber(3.0 + 1e-3, tol=1e-2)
        assert not is_wholenumber(3.0 + 1e-3, tol=1e-4)


class TestFrobeniusNorm:
    def test_perfect_reconstruction_is_zero(self, simple_P, known_weights, m_from_P):
        # m_from_P == simple_P @ known_weights, and it already sums to 1,
        # so the reconstruction error must be (near) zero.
        err = FrobeniusNorm(m_from_P, simple_P, known_weights)
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_normalises_m_by_its_sum(self, simple_P, known_weights):
        # Scaling m must not change the error: the function divides by m.sum().
        m = simple_P @ known_weights
        err_small = FrobeniusNorm(m, simple_P, known_weights)
        err_big = FrobeniusNorm(m * 1000.0, simple_P, known_weights)
        assert err_small == pytest.approx(err_big)

    def test_known_value(self):
        # Hand-computed 2x2 example.
        # m normalised = [0.5, 0.5]; reconstruction = P @ exposures.
        m = np.array([1.0, 1.0])
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        exposures = np.array([1.0, 0.0])  # reconstruction = [1, 0]
        # residual = [0.5, 0.5] - [1, 0] = [-0.5, 0.5]; norm = sqrt(0.5)
        err = FrobeniusNorm(m, P, exposures)
        assert err == pytest.approx(np.sqrt(0.5))

    def test_positive_for_wrong_exposures(self, simple_P, known_weights, m_from_P):
        wrong = np.array([0.0, 0.0, 1.0])
        assert FrobeniusNorm(m_from_P, simple_P, wrong) > 0
