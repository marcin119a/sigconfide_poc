import numpy as np
import pytest
from sigconfide.estimates.bootstrap import bootstrapSigExposures


class TestBootstrapValidation:
    def test_length_mismatch_raises(self, simple_P):
        m = np.ones(simple_P.shape[0] + 1)
        with pytest.raises(ValueError, match="must be the same"):
            bootstrapSigExposures(m, simple_P, R=5, mutation_count=100)

    def test_single_column_P_raises(self, simple_P):
        m = np.ones(simple_P.shape[0])
        with pytest.raises(ValueError, match="at least 2 columns"):
            bootstrapSigExposures(m, simple_P[:, :1], R=5, mutation_count=100)

    def test_fractional_m_without_mutation_count_raises(self, m_from_P, simple_P):
        # m_from_P holds probabilities (not whole numbers) and no mutation_count.
        with pytest.raises(ValueError, match="mutation_count"):
            bootstrapSigExposures(m_from_P, simple_P, R=5)


class TestBootstrapBehaviour:
    def test_output_shapes(self, counts_profile, simple_P):
        np.random.seed(0)
        R = 8
        exposures, errors = bootstrapSigExposures(counts_profile, simple_P, R=R)
        assert exposures.shape == (simple_P.shape[1], R)
        assert errors.shape == (R,)

    def test_columns_sum_to_one(self, counts_profile, simple_P):
        np.random.seed(0)
        exposures, _ = bootstrapSigExposures(counts_profile, simple_P, R=6)
        assert exposures.sum(axis=0) == pytest.approx(np.ones(6))

    def test_mutation_count_inferred_from_counts(self, counts_profile, simple_P):
        # Integer profile → mutation_count need not be passed explicitly.
        np.random.seed(0)
        exposures, _ = bootstrapSigExposures(counts_profile, simple_P, R=4)
        assert exposures.shape[1] == 4

    def test_reproducible_with_seed(self, counts_profile, simple_P):
        np.random.seed(42)
        exp1, err1 = bootstrapSigExposures(counts_profile, simple_P, R=5)
        np.random.seed(42)
        exp2, err2 = bootstrapSigExposures(counts_profile, simple_P, R=5)
        assert np.array_equal(exp1, exp2)
        assert np.array_equal(err1, err2)

    def test_probabilities_with_explicit_mutation_count(self, m_from_P, simple_P):
        np.random.seed(1)
        exposures, _ = bootstrapSigExposures(
            m_from_P, simple_P, R=5, mutation_count=3000
        )
        assert exposures.shape == (simple_P.shape[1], 5)
        assert exposures.sum(axis=0) == pytest.approx(np.ones(5))

    def test_recovers_signals_on_average(self, counts_profile, simple_P, known_weights):
        # With enough replicates the mean bootstrap exposure should be close to
        # the true mixing weights (loose tolerance — this is a statistical test).
        np.random.seed(7)
        exposures, _ = bootstrapSigExposures(counts_profile, simple_P, R=200)
        assert exposures.mean(axis=1) == pytest.approx(known_weights, abs=0.1)
