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

    def test_overdispersion_none_is_the_plain_bootstrap(self, counts_profile):
        np.random.seed(0)
        plain = _bootstrap_matrix(counts_profile, mutation_count=None, R=5)
        np.random.seed(0)
        explicit = _bootstrap_matrix(
            counts_profile, mutation_count=None, R=5, overdispersion=None
        )
        assert np.array_equal(plain, explicit)

    def test_overdispersion_widens_the_replicates(self):
        """A deep profile: multinomial SD is sqrt(c), the gamma term adds sigma*c."""
        rng = np.random.default_rng(0)
        m = rng.poisson(2000, size=96).astype(float)
        c = m / m.sum()
        N = m.sum()
        np.random.seed(0)
        plain = _bootstrap_matrix(m, mutation_count=None, R=200)
        np.random.seed(0)
        wide = _bootstrap_matrix(m, mutation_count=None, R=200, overdispersion=0.1)
        assert wide.sum(axis=0) == pytest.approx(np.ones(200))
        sd_plain = plain.std(axis=1) / np.sqrt(c * (1 - c) / N)
        sd_wide = wide.std(axis=1) / np.sqrt(c * (1 - c) / N + (0.1 * c) ** 2)
        assert sd_plain.mean() == pytest.approx(1.0, abs=0.1)
        assert sd_wide.mean() == pytest.approx(1.0, abs=0.1)
        assert (wide.std(axis=1) > 2 * plain.std(axis=1)).all()

    def test_negative_overdispersion_raises(self, counts_profile):
        with pytest.raises(ValueError, match="overdispersion"):
            _bootstrap_matrix(
                counts_profile, mutation_count=None, R=2, overdispersion=-0.1
            )


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
        sel_idx, exposures, errors = hybrid_stepwise_selection(m, selection_panel, R=50)
        assert 0 in sel_idx
        assert 2 in sel_idx

    def test_return_shapes_consistent(self, selection_panel):
        weights = np.array([0.5, 0.0, 0.5, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, errors = hybrid_stepwise_selection(m, selection_panel, R=40)
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


class TestFitGainPruning:
    """`min_fit_improvement` must evict signatures the reconstruction ignores.

    The bootstrap criterion keeps any exposure that is stably above `threshold`,
    which on deep profiles admits signatures that contribute nothing to the fit.
    """

    def _counts(self, P, weights, total=20000):
        probs = P @ weights
        return np.round(probs * total)

    def _contaminated_counts(self, P, eps=0.06, total=40000):
        """A 2-signature mix plus a component no column of P can represent.

        This is the situation that makes the bootstrap over-select on real data:
        the residual has to go somewhere, the flattest available signature
        absorbs it at a stable few percent, and with enough mutations that
        exposure clears `threshold` in every replicate.
        """
        contamination = np.array([0.10, 0.15, 0.15, 0.20, 0.20, 0.20])
        contamination /= contamination.sum()
        profile = (1 - eps) * (P @ np.array([0.6, 0.0, 0.4, 0.0, 0.0]))
        profile += eps * contamination
        return np.round(profile / profile.sum() * total)

    def test_drops_signature_that_only_absorbs_residual(self, selection_panel):
        m = self._contaminated_counts(selection_panel)
        np.random.seed(0)
        loose, _, _ = hybrid_stepwise_selection(m, selection_panel, R=40)
        np.random.seed(0)
        pruned, exposures, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, min_fit_improvement=0.002
        )
        # The bootstrap keeps the residual sponge; the fit-gain gate evicts it.
        assert 4 in loose
        assert set(pruned.tolist()) == {0, 2}
        assert exposures.shape[0] == len(pruned)
        assert exposures.sum() == pytest.approx(1.0)

    def test_keeps_signatures_the_fit_needs(self, selection_panel):
        # Every one of the three contributes a distinct, large component.
        weights = np.array([0.4, 0.35, 0.25, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        pruned, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, min_fit_improvement=0.002
        )
        assert {0, 1, 2} <= set(pruned.tolist())

    def test_mandatory_survives_pruning(self, selection_panel):
        # Signature 4 contributes nothing to the fit but is protected.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        pruned, _, _ = hybrid_stepwise_selection(
            m,
            selection_panel,
            R=40,
            mandatory_indices=[4],
            min_fit_improvement=0.5,  # aggressive enough to strip everything else
        )
        assert 4 in pruned

    def test_never_falls_below_two_signatures(self, selection_panel):
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        pruned, exposures, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, min_fit_improvement=1.0
        )
        assert len(pruned) == 2
        assert exposures.sum() == pytest.approx(1.0)

    def test_disabled_by_default(self, selection_panel):
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        baseline, _, _ = hybrid_stepwise_selection(m, selection_panel, R=40)
        np.random.seed(0)
        explicit_off, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, min_fit_improvement=None
        )
        assert baseline.tolist() == explicit_off.tolist()

    def test_indices_stay_global_with_pre_filter(self, selection_panel):
        # Pruning happens in the filtered index space; the result must still be
        # expressed in original P columns.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        pruned, _, _ = hybrid_stepwise_selection(
            m,
            selection_panel,
            R=40,
            pre_filter_threshold=0.001,
            min_fit_improvement=0.002,
        )
        assert np.all((pruned >= 0) & (pruned < selection_panel.shape[1]))
        assert {0, 2} <= set(pruned.tolist())


class TestCycleGuard:
    """The greedy add/remove walk must terminate even when moves undo each other.

    A pair of signatures whose apparent significance depends on the presence of
    the other makes the search oscillate:  {0,1,2,3} -> {0,1,3} -> {0,1} ->
    {0,1,2} -> {0,1,2,3} -> ...  Before the visited-set guard this looped
    forever, which is exactly what happened on real profiles carrying only a
    handful of mutations (panel-sized breast catalogues).
    """

    @staticmethod
    def _oscillating_decomposer(counter):
        """QP stand-in whose exposures depend on which signatures are present.

        Signature identity is read off the columns of the submatrix handed to
        the decomposer (each column has a unique maximum row).
        """

        def decompose(m_col, P_sub):
            counter["calls"] += 1
            present = [int(np.argmax(P_sub[:, j])) for j in range(P_sub.shape[1])]
            exposures = np.zeros(P_sub.shape[1])
            for j, sig in enumerate(present):
                if sig in (0, 1):  # always clearly significant
                    exposures[j] = 0.4
                elif sig == 2:  # significant only while 3 is absent
                    exposures[j] = 0.0 if 3 in present else 0.2
                elif sig == 3:  # significant only while 2 is present
                    exposures[j] = 0.2 if 2 in present else 0.0
            total = exposures.sum()
            return exposures / total if total > 0 else exposures

        return decompose

    @pytest.fixture
    def identity_panel(self):
        P = np.eye(4) * 0.7 + 0.1
        return P / P.sum(axis=0)

    def test_terminates_on_oscillating_moves(self, identity_panel):
        counter = {"calls": 0}
        m = np.array([25.0, 25.0, 25.0, 25.0])
        np.random.seed(0)
        sel_idx, exposures, _ = hybrid_stepwise_selection(
            m,
            identity_panel,
            R=5,
            decomposition_method=self._oscillating_decomposer(counter),
            max_iterations=50,
        )
        # Terminated by cycle detection, not by exhausting max_iterations:
        # a cycling search would keep evaluating moves for all 50 iterations.
        assert counter["calls"] < 200
        assert set(sel_idx.tolist()) == {0, 1, 2}
        assert exposures.sum() == pytest.approx(1.0)

    def test_max_iterations_caps_the_search(self, identity_panel):
        counter = {"calls": 0}
        m = np.array([25.0, 25.0, 25.0, 25.0])
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m,
            identity_panel,
            R=5,
            decomposition_method=self._oscillating_decomposer(counter),
            max_iterations=1,
        )
        # One move only: signature 2 dropped from the initial full set.
        assert set(sel_idx.tolist()) == {0, 1, 3}

    def test_low_count_profile_terminates(self, selection_panel):
        """A 3-mutation profile against the whole panel must still return."""
        m = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        np.random.seed(7)
        sel_idx, exposures, _ = hybrid_stepwise_selection(
            m, selection_panel, R=20, pre_filter_threshold=0.001
        )
        assert len(sel_idx) >= 2
        assert exposures.sum() == pytest.approx(1.0)
