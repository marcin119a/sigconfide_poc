import numpy as np
import pytest

from sigconfide.estimates.selection import (
    _bootstrap_matrix,
    _p_values,
    hybrid_stepwise_selection,
)


class TestPValues:
    def test_known_matrix(self):
        # rows = signatures, cols = bootstrap replicates.
        exposures = np.array(
            [
                [0.5, 0.5, 0.0],  # 2/3 replicates above threshold -> p = 1/3
                [0.0, 0.0, 0.0],  # 0/3 replicates above threshold -> p = 1.0
                [0.2, 0.2, 0.2],  # 3/3 replicates above threshold -> p = 0.0
            ]
        )
        pv = _p_values(exposures, threshold=0.1)
        assert pv == pytest.approx([1 / 3, 1.0, 0.0])

    def test_bounds(self):
        exposures = np.random.default_rng(0).random((4, 10))
        pv = _p_values(exposures, threshold=0.5)
        assert np.all((pv >= 0.0) & (pv <= 1.0))


class TestBootstrapMatrix:
    def test_shape(self, counts_profile):
        np.random.seed(0)
        K = len(counts_profile)
        M = _bootstrap_matrix(counts_profile, mutation_count=None, R=7)
        assert M.shape == (K, 7)

    def test_columns_sum_to_one(self, counts_profile):
        np.random.seed(0)
        M = _bootstrap_matrix(counts_profile, mutation_count=None, R=5)
        assert M.sum(axis=0) == pytest.approx(np.ones(5))

    def test_fractional_without_count_raises(self, m_from_P):
        with pytest.raises(ValueError, match="mutation_count"):
            _bootstrap_matrix(m_from_P, mutation_count=None, R=5)

    def test_fractional_with_count_ok(self, m_from_P):
        np.random.seed(0)
        M = _bootstrap_matrix(m_from_P, mutation_count=2000, R=4)
        assert M.shape == (len(m_from_P), 4)


@pytest.fixture
def selection_panel():
    """A 6-context x 5-signature panel with well-separated columns."""
    P = np.array(
        [
            [0.60, 0.05, 0.10, 0.05, 0.10],
            [0.10, 0.60, 0.10, 0.05, 0.10],
            [0.10, 0.10, 0.55, 0.05, 0.10],
            [0.10, 0.10, 0.10, 0.60, 0.10],
            [0.05, 0.10, 0.10, 0.20, 0.30],
            [0.05, 0.05, 0.05, 0.05, 0.30],
        ],
        dtype=float,
    )
    return P / P.sum(axis=0)


class TestHybridStepwiseSelection:
    def _counts(self, P, weights, total=5000):
        probs = P @ weights
        return np.round(probs * total)

    def test_selects_true_signatures(self, selection_panel):
        # Profile is a clean mix of signatures 0 and 2.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, errors = hybrid_stepwise_selection(
            m, selection_panel, R=50
        )
        assert 0 in sel_idx
        assert 2 in sel_idx

    def test_return_shapes_consistent(self, selection_panel):
        weights = np.array([0.5, 0.0, 0.5, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, errors = hybrid_stepwise_selection(
            m, selection_panel, R=40
        )
        assert exposures.shape[0] == len(sel_idx)
        assert exposures.sum() == pytest.approx(1.0)
        assert errors.shape == (1,)

    def test_mandatory_indices_always_present(self, selection_panel):
        # Signature 4 contributes nothing, but is marked mandatory -> must stay.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, mandatory_indices=[4]
        )
        assert 4 in sel_idx

    def test_pre_filter_keeps_mandatory(self, selection_panel):
        # Aggressive pre-filter would drop the absent signature 4, but mandatory
        # protection must keep it in the final result.
        weights = np.array([0.7, 0.0, 0.3, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m,
            selection_panel,
            R=40,
            pre_filter_threshold=0.05,
            mandatory_indices=[4],
        )
        assert 4 in sel_idx

    def test_indices_map_to_original_columns(self, selection_panel):
        # With pre-filter on, returned indices must reference the ORIGINAL P
        # columns (0..N-1), not positions in the filtered matrix.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, pre_filter_threshold=0.01
        )
        assert np.all(sel_idx >= 0)
        assert np.all(sel_idx < selection_panel.shape[1])
        # No duplicates and sorted ascending (as constructed in the function).
        assert len(set(sel_idx.tolist())) == len(sel_idx)
