import numpy as np
import pytest
from sigconfide.estimates.standard import findSigExposures


class TestFindSigExposuresValidation:
    def test_row_mismatch_raises(self, simple_P):
        M = np.ones((simple_P.shape[0] + 1, 2))
        with pytest.raises(ValueError, match="same number of rows"):
            findSigExposures(M, simple_P)

    def test_single_column_P_raises(self, simple_P):
        M = np.ones((simple_P.shape[0], 2))
        with pytest.raises(ValueError, match="at least 2 columns"):
            findSigExposures(M, simple_P[:, :1])

    def test_non_callable_method_raises(self, simple_P):
        M = np.ones((simple_P.shape[0], 2))
        with pytest.raises(ValueError, match="must be a function"):
            findSigExposures(M, simple_P, decomposition_method="not_a_function")


class TestFindSigExposuresBehaviour:
    def test_output_shapes(self, simple_P, m_from_P):
        M = np.column_stack([m_from_P, m_from_P, m_from_P])  # 3 samples
        exposures, errors = findSigExposures(M, simple_P)
        assert exposures.shape == (simple_P.shape[1], 3)
        assert errors.shape == (3,)

    def test_columns_sum_to_one(self, simple_P, m_from_P):
        M = np.column_stack([m_from_P, m_from_P])
        exposures, _ = findSigExposures(M, simple_P)
        assert exposures.sum(axis=0) == pytest.approx(np.ones(2))

    def test_recovers_known_weights(self, simple_P, known_weights, m_from_P):
        M = m_from_P.reshape(-1, 1)
        exposures, errors = findSigExposures(M, simple_P)
        assert exposures[:, 0] == pytest.approx(known_weights, abs=1e-6)
        assert errors[0] == pytest.approx(0.0, abs=1e-6)

    def test_column_normalisation(self, simple_P, m_from_P):
        # An unnormalised column must give the same result as a normalised one.
        M_norm = m_from_P.reshape(-1, 1)
        M_scaled = (m_from_P * 1234.0).reshape(-1, 1)
        exp_norm, _ = findSigExposures(M_norm, simple_P)
        exp_scaled, _ = findSigExposures(M_scaled, simple_P)
        assert exp_norm == pytest.approx(exp_scaled)

    def test_custom_decomposition_method_is_used(self, simple_P):
        # Inject a stub method to verify findSigExposures wires M/P through it
        # without depending on quadprog.
        n_sigs = simple_P.shape[1]
        constant = np.full(n_sigs, 1.0 / n_sigs)

        def stub(column, P):
            assert column.shape == (P.shape[0],)
            return constant

        M = np.ones((simple_P.shape[0], 2))
        exposures, _ = findSigExposures(M, simple_P, decomposition_method=stub)
        assert exposures == pytest.approx(np.column_stack([constant, constant]))
