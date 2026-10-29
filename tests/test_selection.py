import numpy as np
import pytest
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.selection import (
    _bootstrap_matrix,
    _support,
    estimate_overdispersion,
    hybrid_stepwise_selection,
)


class TestSupport:
    def test_known_matrix(self):
        # rows = signatures, cols = bootstrap replicates.
        exposures = np.array(
            [
                [0.5, 0.5, 0.0],  # 2/3 replicates above threshold -> support = 2/3
                [0.0, 0.0, 0.0],  # 0/3 replicates above threshold -> support = 0.0
                [0.2, 0.2, 0.2],  # 3/3 replicates above threshold -> support = 1.0
            ]
        )
        sup = _support(exposures, threshold=0.1)
        assert sup == pytest.approx([2 / 3, 0.0, 1.0])

    def test_bounds(self):
        exposures = np.random.default_rng(0).random((4, 10))
        sup = _support(exposures, threshold=0.5)
        assert np.all((sup >= 0.0) & (sup <= 1.0))


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

    def test_replicates_are_whole_counts_over_the_mutation_count(self, counts_profile):
        n = int(counts_profile.sum())
        M = _bootstrap_matrix(counts_profile, mutation_count=None, R=6, rng=0)
        counts = M * n
        assert counts == pytest.approx(np.round(counts))
        assert (counts >= 0).all()

    def test_replicates_follow_the_multinomial_moments(self):
        """Per channel: mean m, variance m(1-m)/N."""
        rng = np.random.default_rng(0)
        m = rng.poisson(300, size=20).astype(float)
        c = m / m.sum()
        N = int(m.sum())
        M = _bootstrap_matrix(m, mutation_count=None, R=4000, rng=1)
        assert M.mean(axis=1) == pytest.approx(c, rel=0.02)
        assert M.var(axis=1) == pytest.approx(c * (1 - c) / N, rel=0.15)

    def test_zero_channels_stay_zero(self):
        m = np.array([5.0, 0.0, 3.0, 0.0, 2.0])
        M = _bootstrap_matrix(m, mutation_count=None, R=50, rng=0)
        assert (M[[1, 3]] == 0).all()


class TestBootstrapRng:
    """`rng` picks the random source; None keeps following `np.random.seed`."""

    @pytest.mark.parametrize("overdispersion", [None, 0.1])
    def test_same_seed_same_replicates(self, counts_profile, overdispersion):
        a = _bootstrap_matrix(counts_profile, None, 5, overdispersion, rng=42)
        b = _bootstrap_matrix(counts_profile, None, 5, overdispersion, rng=42)
        assert np.array_equal(a, b)

    def test_int_seed_matches_the_generator_built_from_it(self, counts_profile):
        by_int = _bootstrap_matrix(counts_profile, None, 5, rng=7)
        gen = np.random.default_rng(7)
        by_gen = _bootstrap_matrix(counts_profile, None, 5, rng=gen)
        assert np.array_equal(by_int, by_gen)

    def test_different_seeds_differ(self, counts_profile):
        a = _bootstrap_matrix(counts_profile, None, 5, rng=1)
        b = _bootstrap_matrix(counts_profile, None, 5, rng=2)
        assert not np.array_equal(a, b)

    def test_a_generator_advances_between_calls(self, counts_profile):
        gen = np.random.default_rng(3)
        a = _bootstrap_matrix(counts_profile, None, 5, rng=gen)
        b = _bootstrap_matrix(counts_profile, None, 5, rng=gen)
        assert not np.array_equal(a, b)

    def test_spawned_children_are_independent_and_reproducible(self, counts_profile):
        def draw():
            kids = np.random.SeedSequence(11).spawn(2)
            return [_bootstrap_matrix(counts_profile, None, 5, rng=k) for k in kids]

        first, again = draw(), draw()
        assert not np.array_equal(first[0], first[1])
        assert all(np.array_equal(x, y) for x, y in zip(first, again))

    def test_none_follows_the_global_seed(self, counts_profile):
        np.random.seed(5)
        a = _bootstrap_matrix(counts_profile, None, 5)
        np.random.seed(5)
        b = _bootstrap_matrix(counts_profile, None, 5, rng=None)
        assert np.array_equal(a, b)

    def test_an_explicit_rng_leaves_the_global_state_alone(self, counts_profile):
        np.random.seed(9)
        expected = np.random.random()
        np.random.seed(9)
        _bootstrap_matrix(counts_profile, None, 5, rng=0)
        _bootstrap_matrix(counts_profile, None, 5, 0.1, rng=0)
        assert np.random.random() == expected

    def test_invalid_rng_raises(self, counts_profile):
        with pytest.raises(TypeError):
            _bootstrap_matrix(counts_profile, None, 5, rng="seed")


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

    def test_anchor_indices_always_present(self, selection_panel):
        # Signature 4 contributes nothing, but is an anchor -> must stay.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, anchor_indices=[4], anchor_floor=None
        )
        assert 4 in sel_idx

    def test_pre_filter_keeps_anchors(self, selection_panel):
        # Aggressive pre-filter would drop the absent signature 4, but the
        # anchor protection must keep it in the final result.
        weights = np.array([0.7, 0.0, 0.3, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m,
            selection_panel,
            R=40,
            pre_filter_threshold=0.05,
            anchor_indices=[4],
            anchor_floor=None,
        )
        assert 4 in sel_idx

    def test_anchor_floor_default_is_0002(self):
        import inspect

        params = inspect.signature(hybrid_stepwise_selection).parameters
        assert params["anchor_floor"].default == 0.002

    def test_anchor_floor_default_drops_unused_anchor(self, selection_panel):
        # Same profile as test_anchor_indices_always_present, package default.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, _, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, anchor_indices=[4]
        )
        assert 4 not in sel_idx

    def test_anchor_floor_drops_unused_anchor(self, selection_panel):
        # Signature 4 is absent and anchored; with a floor the final fit may let
        # it go, because its exposure is (near) zero.
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, errors = hybrid_stepwise_selection(
            m, selection_panel, R=40, anchor_indices=[4], anchor_floor=0.01
        )
        assert 4 not in sel_idx
        assert set(sel_idx) == {0, 2}
        assert exposures.shape[0] == len(sel_idx)
        assert exposures.sum() == pytest.approx(1.0)
        assert errors.shape == (1,)

    def test_anchor_floor_keeps_used_anchor(self, selection_panel):
        # Signature 4 is anchored AND really present well above the floor.
        weights = np.array([0.5, 0.0, 0.3, 0.0, 0.2])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, anchor_indices=[4], anchor_floor=0.01
        )
        assert 4 in sel_idx
        assert exposures[list(sel_idx).index(4)] > 0.01

    def test_anchor_floor_keeps_at_least_two(self, selection_panel):
        # Only the two anchor signatures survive the search and one is unused:
        # the floor never empties the set below two signatures.
        weights = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        sel_idx, exposures, _ = hybrid_stepwise_selection(
            m, selection_panel, R=40, anchor_indices=[3, 4],
            pre_filter_threshold=0.5, anchor_floor=0.01,
        )
        assert len(sel_idx) >= 2
        assert 3 in sel_idx
        assert exposures.sum() == pytest.approx(1.0)

    def test_anchor_floor_without_anchor_set_is_a_no_op(self, selection_panel):
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(0)
        a = hybrid_stepwise_selection(m, selection_panel, R=40)
        np.random.seed(0)
        b = hybrid_stepwise_selection(m, selection_panel, R=40, anchor_floor=None)
        assert list(a[0]) == list(b[0])
        np.testing.assert_allclose(a[1], b[1])

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

    def test_same_rng_seed_gives_the_same_fit(self, selection_panel):
        weights = np.array([0.5, 0.0, 0.3, 0.2, 0.0])
        m = self._counts(selection_panel, weights)
        first = hybrid_stepwise_selection(m, selection_panel, R=40, rng=3)
        again = hybrid_stepwise_selection(m, selection_panel, R=40, rng=3)
        for a, b in zip(first, again):
            assert np.array_equal(a, b)

    def test_rng_leaves_the_global_seed_alone(self, selection_panel):
        weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])
        m = self._counts(selection_panel, weights)
        np.random.seed(4)
        expected = np.random.random()
        np.random.seed(4)
        hybrid_stepwise_selection(m, selection_panel, R=40, rng=0)
        assert np.random.random() == expected

    def test_per_sample_children_make_the_run_order_independent(self, selection_panel):
        """The reason to pass `rng`: with one spawned child per sample the result
        does not depend on which worker, or in what order, a sample is fitted."""
        mixes = [
            np.array([0.6, 0.0, 0.4, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0, 0.5, 0.0]),
            np.array([0.4, 0.0, 0.2, 0.0, 0.4]),
        ]
        profiles = [self._counts(selection_panel, w) for w in mixes]
        children = np.random.SeedSequence(0).spawn(len(profiles))

        def fit(i):
            return hybrid_stepwise_selection(
                profiles[i], selection_panel, R=40, rng=children[i]
            )

        forward = [fit(i) for i in range(3)]
        backward = [fit(i) for i in reversed(range(3))][::-1]
        for a, b in zip(forward, backward):
            for x, y in zip(a, b):
                assert np.array_equal(x, y)


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
            pre_filter_threshold=None,
            min_deviance_gain=None,  # these tests exercise the greedy walk alone  # the walk must start from the full panel
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
            pre_filter_threshold=None,
            min_deviance_gain=None,  # these tests exercise the greedy walk alone  # the walk must start from the full panel
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


class TestDevianceSteps:
    """`min_deviance_gain`: deviance-scored swap and backward elimination."""

    @staticmethod
    def _counts(P, weights, total=20000):
        return np.round(total * (P @ weights))

    def test_poisson_deviance_is_zero_at_the_data_and_positive_elsewhere(self):
        from sigconfide.estimates.selection import _poisson_deviance

        x = np.array([10.0, 0.0, 25.0, 5.0])
        assert _poisson_deviance(x, x) == pytest.approx(0.0, abs=1e-2)
        assert _poisson_deviance(x, x * 1.3) > 0
        assert _poisson_deviance(x, np.array([10.0, 3.0, 25.0, 5.0])) > 0

    def test_swap_brings_back_the_generating_signature(self, simple_P):
        from sigconfide.estimates.selection import _swap_by_deviance

        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        got = _swap_by_deviance(counts, simple_P, [1, 2], 5.0, set(), decomposeQP)
        assert got == [0, 2]

    def test_swap_never_touches_a_protected_signature(self, simple_P):
        from sigconfide.estimates.selection import _swap_by_deviance

        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        got = _swap_by_deviance(counts, simple_P, [1, 2], 5.0, {1}, decomposeQP)
        assert 1 in got  # the spurious column is protected, so it stays
        assert got == [0, 1]  # and the swap replaces the other one instead

    def test_swap_keeps_a_set_that_already_fits(self, simple_P, known_weights):
        from sigconfide.estimates.selection import _swap_by_deviance

        counts = self._counts(simple_P, known_weights)
        got = _swap_by_deviance(counts, simple_P, [0, 1, 2], 5.0, set(), decomposeQP)
        assert got == [0, 1, 2]

    def test_prune_drops_an_unneeded_signature_and_keeps_the_needed(self, simple_P):
        from sigconfide.estimates.selection import _prune_by_deviance

        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        got = _prune_by_deviance(counts, simple_P, [0, 1, 2], 5.0, set(), decomposeQP)
        assert got == [0, 2]

    def test_prune_respects_protection_and_the_two_signature_floor(self, simple_P):
        from sigconfide.estimates.selection import _prune_by_deviance

        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        assert _prune_by_deviance(counts, simple_P, [0, 1, 2], 5.0, {1}, decomposeQP) == [0, 1, 2]
        assert _prune_by_deviance(counts, simple_P, [1, 2], 1e9, set(), decomposeQP) == [1, 2]

    def test_selection_with_the_option_recovers_the_mixture(self, simple_P, known_weights, counts_profile):
        np.random.seed(0)
        idx, expo, _ = hybrid_stepwise_selection(
            counts_profile, simple_P, R=30, min_deviance_gain=5.0
        )
        assert sorted(idx.tolist()) == [0, 1, 2]
        assert expo == pytest.approx(known_weights, abs=0.02)

    def test_selection_with_the_option_drops_a_spurious_column(self, simple_P):
        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        np.random.seed(0)
        idx, _, _ = hybrid_stepwise_selection(counts, simple_P, R=30, min_deviance_gain=5.0)
        assert sorted(idx.tolist()) == [0, 2]

    def test_fractional_profile_uses_the_mutation_count(self, simple_P):
        probs = simple_P @ np.array([0.6, 0.0, 0.4])
        np.random.seed(0)
        idx, _, _ = hybrid_stepwise_selection(
            probs, simple_P, R=30, mutation_count=20000, min_deviance_gain=5.0
        )
        assert sorted(idx.tolist()) == [0, 2]

    def test_negative_margin_raises(self, counts_profile, simple_P):
        with pytest.raises(ValueError, match="min_deviance_gain"):
            hybrid_stepwise_selection(counts_profile, simple_P, R=2, min_deviance_gain=-1.0)

    def test_dispersion_scaling_is_a_no_op_on_poisson_or_quieter_data(self, simple_P):
        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        np.random.seed(0)
        a = hybrid_stepwise_selection(counts, simple_P, R=30, min_deviance_gain=5.0)
        np.random.seed(0)
        b = hybrid_stepwise_selection(
            counts, simple_P, R=30, min_deviance_gain=5.0, dispersion_scaled=True
        )
        assert sorted(a[0].tolist()) == sorted(b[0].tolist()) == [0, 2]

    def test_dispersion_scaling_raises_the_margin_on_noisy_data(self, simple_P):
        from sigconfide.estimates.selection import _deviance_of

        rng = np.random.default_rng(3)
        clean = 20000 * (simple_P @ np.array([0.6, 0.0, 0.4]))
        noisy = np.round(clean * rng.gamma(4.0, 0.25, size=len(clean)))  # CV 0.5
        phi = _deviance_of(noisy, simple_P, [0, 1, 2], decomposeQP) / 1
        assert phi > 1  # over-dispersed by construction, so the margin scales up

    def test_the_default_is_the_scaled_deviance_step_at_margin_five(self, simple_P):
        counts = self._counts(simple_P, np.array([0.6, 0.0, 0.4]))
        np.random.seed(0)
        default = hybrid_stepwise_selection(counts, simple_P, R=30)
        np.random.seed(0)
        explicit = hybrid_stepwise_selection(
            counts, simple_P, R=30, min_deviance_gain=5.0, dispersion_scaled=True
        )
        np.random.seed(0)
        off = hybrid_stepwise_selection(counts, simple_P, R=30, min_deviance_gain=None)
        assert sorted(default[0].tolist()) == sorted(explicit[0].tolist()) == [0, 2]
        assert len(off[0]) >= 2  # the plain search still runs with the step disabled


class TestOverdispersionEstimate:
    """`estimate_overdispersion`: sigma_b by the method of moments on the residual."""

    @staticmethod
    def _panel(K=96, N=6, seed=0):
        rng = np.random.default_rng(seed)
        P = rng.gamma(0.5, size=(K, N))  # peaked columns, like real signatures
        return P / P.sum(axis=0)

    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])

    def test_exact_mixture_gives_zero(self):
        P = self._panel()
        counts = np.round(50000 * (P @ self.weights))
        assert estimate_overdispersion(counts, P) == 0.0

    def test_poisson_noise_stays_within_counting_noise(self):
        """Residual inside the Poisson noise: the estimate is (close to) zero."""
        P = self._panel()
        rng = np.random.default_rng(1)
        mu = 50000 * (P @ self.weights)
        est = [
            estimate_overdispersion(rng.poisson(mu).astype(float), P)
            for _ in range(20)
        ]
        assert min(est) >= 0.0
        assert max(est) < 0.03
        assert np.mean(est) < 0.01

    @pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20])
    def test_recovers_the_gamma_poisson_dispersion(self, sigma):
        """Deep gamma-Poisson profile with CV `sigma` per channel: Var = c + (sigma c)^2."""
        P = self._panel()
        rng = np.random.default_rng(2)
        mu = 200000 * (P @ self.weights)
        est = []
        for _ in range(20):
            g = rng.gamma(sigma**-2, sigma**2, size=P.shape[0])
            est.append(estimate_overdispersion(rng.poisson(mu * g).astype(float), P))
        assert np.mean(est) == pytest.approx(sigma, abs=0.2 * sigma)

    def test_sharpens_with_depth(self):
        """Noisy at small n, precise at large n (where sigma_b matters)."""
        P = self._panel()
        rng = np.random.default_rng(3)
        spread = {}
        for n in (500, 200000):
            mu = n * (P @ self.weights)
            est = []
            for _ in range(20):
                g = rng.gamma(100.0, 0.01, size=P.shape[0])  # CV 0.10
                est.append(estimate_overdispersion(rng.poisson(mu * g).astype(float), P))
            spread[n] = np.std(est)
        assert spread[200000] < spread[500] / 3

    def test_fractional_profile_needs_the_mutation_count(self, m_from_P, simple_P):
        with pytest.raises(ValueError, match="mutation_count"):
            estimate_overdispersion(m_from_P, simple_P)

    def test_fractional_with_count_matches_the_counts(self):
        P = self._panel()
        rng = np.random.default_rng(4)
        g = rng.gamma(100.0, 0.01, size=P.shape[0])
        counts = rng.poisson(100000 * (P @ self.weights) * g).astype(float)
        n = int(counts.sum())
        assert estimate_overdispersion(counts / n, P, mutation_count=n) == pytest.approx(
            estimate_overdispersion(counts, P)
        )


class TestAutoOverdispersion:
    """`overdispersion="auto"` plugs the estimate into the bootstrap."""

    @staticmethod
    def _noisy_counts(P, weights, n, sigma, seed):
        rng = np.random.default_rng(seed)
        g = rng.gamma(sigma**-2, sigma**2, size=P.shape[0])
        return rng.poisson(n * (P @ weights) * g).astype(float)

    def test_auto_matches_the_explicit_estimate(self):
        P = TestOverdispersionEstimate._panel(N=8)
        w = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])
        counts = self._noisy_counts(P, w, 100000, 0.10, seed=5)
        sigma = estimate_overdispersion(counts, P)
        assert sigma > 0.05
        auto = hybrid_stepwise_selection(
            counts, P, R=30, pre_filter_threshold=None, overdispersion="auto", rng=7
        )
        explicit = hybrid_stepwise_selection(
            counts, P, R=30, pre_filter_threshold=None, overdispersion=sigma, rng=7
        )
        assert auto[0].tolist() == explicit[0].tolist()
        assert auto[1] == pytest.approx(explicit[1])

    def test_auto_on_a_clean_profile_is_the_multinomial_bootstrap(self, simple_P, counts_profile):
        auto = hybrid_stepwise_selection(
            counts_profile, simple_P, R=30, overdispersion="auto", rng=1
        )
        plain = hybrid_stepwise_selection(
            counts_profile, simple_P, R=30, overdispersion=None, rng=1
        )
        assert auto[0].tolist() == plain[0].tolist()
        assert auto[1] == pytest.approx(plain[1])

    def test_auto_uses_the_pre_filtered_panel(self):
        """With the pre-filter on, the estimate is taken against the kept columns."""
        from sigconfide.decompose.qp import decomposeQP

        P = TestOverdispersionEstimate._panel(N=8)
        w = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])
        counts = self._noisy_counts(P, w, 100000, 0.10, seed=6)
        keep = decomposeQP(counts / counts.sum(), P) > 0.001
        sigma = estimate_overdispersion(counts, P[:, keep])
        auto = hybrid_stepwise_selection(counts, P, R=30, overdispersion="auto", rng=2)
        explicit = hybrid_stepwise_selection(counts, P, R=30, overdispersion=sigma, rng=2)
        assert auto[0].tolist() == explicit[0].tolist()

    def test_auto_recovers_the_mixture_on_a_deep_noisy_profile(self):
        P = TestOverdispersionEstimate._panel(N=8)
        w = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])
        counts = self._noisy_counts(P, w, 100000, 0.10, seed=8)
        idx, expo, _ = hybrid_stepwise_selection(counts, P, R=50, overdispersion="auto", rng=3)
        assert {0, 1, 2, 3} <= set(idx.tolist())
        assert len(idx) <= 5

    def test_fractional_profile_with_count_works(self, simple_P):
        probs = simple_P @ np.array([0.6, 0.0, 0.4])
        idx, _, _ = hybrid_stepwise_selection(
            probs, simple_P, R=30, mutation_count=20000, overdispersion="auto", rng=4
        )
        assert sorted(idx.tolist()) == [0, 2]

    def test_unknown_string_raises(self, simple_P, counts_profile):
        with pytest.raises(ValueError, match="overdispersion"):
            hybrid_stepwise_selection(counts_profile, simple_P, R=2, overdispersion="guess")
