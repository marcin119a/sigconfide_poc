import numpy as np
import pytest
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.selection import (
    _bootstrap_matrix,
    _forward_backward,
    _poisson_deviance,
    bonferroni_margin,
    exposure_intervals,
    hybrid_stepwise_selection,
    selection_stability,
    support_dispersion,
)


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


def _counts(P, weights, total=20000):
    return np.round(total * (P @ weights))


class TestDeviance:
    def test_poisson_deviance_is_zero_at_the_data_and_positive_elsewhere(self):
        x = np.array([10.0, 0.0, 25.0, 5.0])
        assert _poisson_deviance(x, x) == pytest.approx(0.0, abs=1e-2)
        assert _poisson_deviance(x, x * 1.3) > 0
        assert _poisson_deviance(x, np.array([10.0, 3.0, 25.0, 5.0])) > 0

    def test_support_dispersion_is_small_on_exact_and_about_one_on_poisson(
        self, selection_panel
    ):
        clean = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]), 50000)
        assert support_dispersion(clean, selection_panel) < 0.5
        rng = np.random.default_rng(1)
        P = rng.dirichlet(np.ones(48) * 0.5, size=6).T
        mu = 50000 * (P @ np.array([0.5, 0.0, 0.3, 0.0, 0.2, 0.0]))
        phis = [
            support_dispersion(rng.poisson(mu).astype(float), P) for _ in range(20)
        ]
        assert 0.5 < np.median(phis) < 2.0

    def test_support_dispersion_grows_with_multiplicative_noise(self):
        rng = np.random.default_rng(2)
        P = rng.dirichlet(np.ones(48) * 0.5, size=6).T
        mu = 200000 * (P @ np.array([0.5, 0.0, 0.3, 0.0, 0.2, 0.0]))
        noisy = np.round(mu * rng.gamma(25.0, 0.04, size=len(mu)))  # CV 0.2
        assert support_dispersion(noisy, P) > 3.0


class TestForwardBackward:
    def test_adds_the_generating_signatures_and_stops(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        got = _forward_backward(
            counts, selection_panel, [0, 2], 10.0, set(), decomposeQP, 50
        )
        assert got == [0, 2, 4]

    def test_removes_a_start_column_that_is_not_needed(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        got = _forward_backward(
            counts, selection_panel, [0, 1, 2], 10.0, set(), decomposeQP, 50
        )
        assert got == [0, 2, 4]

    def test_never_removes_a_protected_column(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        got = _forward_backward(
            counts, selection_panel, [0, 1, 2], 10.0, {1}, decomposeQP, 50
        )
        assert 1 in got and {0, 2, 4} <= set(got)

    def test_a_huge_margin_adds_nothing(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        assert _forward_backward(
            counts, selection_panel, [0, 2], 1e9, set(), decomposeQP, 50
        ) == [0, 2]

    def test_max_iterations_caps_the_moves(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        assert _forward_backward(
            counts, selection_panel, [0, 1], 10.0, set(), decomposeQP, 1
        ) != [0, 2, 4]


class TestHybridStepwiseSelection:
    def test_selects_true_signatures(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx, expo, err = hybrid_stepwise_selection(counts, selection_panel)
        assert sorted(idx.tolist()) == [0, 2, 4]
        assert expo == pytest.approx([0.5, 0.3, 0.2], abs=0.02)
        assert expo.sum() == pytest.approx(1.0)
        assert err.shape == (1,)

    def test_recovers_the_mixture(self, simple_P, counts_profile, known_weights):
        idx, expo, _ = hybrid_stepwise_selection(counts_profile, simple_P)
        assert sorted(idx.tolist()) == [0, 1, 2]
        assert expo == pytest.approx(known_weights, abs=0.02)

    def test_drops_a_spurious_column(self, simple_P):
        counts = _counts(simple_P, np.array([0.6, 0.0, 0.4]))
        idx, _, _ = hybrid_stepwise_selection(counts, simple_P)
        assert sorted(idx.tolist()) == [0, 2]

    def test_is_deterministic(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        np.random.seed(1)
        a = hybrid_stepwise_selection(counts, selection_panel)
        np.random.seed(2)
        b = hybrid_stepwise_selection(counts, selection_panel)
        assert a[0].tolist() == b[0].tolist()
        assert a[1] == pytest.approx(b[1])

    def test_fractional_profile_uses_the_mutation_count(self, simple_P):
        probs = simple_P @ np.array([0.6, 0.0, 0.4])
        idx, _, _ = hybrid_stepwise_selection(probs, simple_P, mutation_count=20000)
        assert sorted(idx.tolist()) == [0, 2]

    def test_fractional_profile_without_count_raises(self, simple_P):
        with pytest.raises(ValueError, match="mutation_count"):
            hybrid_stepwise_selection(simple_P @ np.array([0.6, 0.0, 0.4]), simple_P)

    def test_anchor_indices_always_present(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[1], anchor_floor=None
        )
        assert 1 in idx.tolist()
        assert {0, 2, 4} <= set(idx.tolist())

    def test_anchor_floor_drops_unused_anchor(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx, expo, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[1], anchor_floor=0.002
        )
        assert 1 not in idx.tolist()
        assert sorted(idx.tolist()) == [0, 2, 4]
        assert expo.sum() == pytest.approx(1.0)

    def test_anchor_floor_keeps_used_anchor(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.1, 0.3, 0.0, 0.1]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[1], anchor_floor=0.002
        )
        assert 1 in idx.tolist()

    def test_anchor_floor_keeps_at_least_two(self, simple_P):
        counts = _counts(simple_P, np.array([0.0, 0.0, 1.0]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, simple_P, anchor_indices=[0, 1], anchor_floor=0.5
        )
        assert len(idx) >= 2

    def test_one_anchor_starts_from_the_strongest_column_too(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[3], anchor_floor=None
        )
        assert {0, 2, 3, 4} == set(idx.tolist())

    def test_dispersion_scaling_raises_the_margin_on_noisy_data(self):
        rng = np.random.default_rng(3)
        P = rng.dirichlet(np.ones(48) * 0.5, size=6).T
        mu = 200000 * (P @ np.array([0.5, 0.0, 0.3, 0.0, 0.2, 0.0]))
        noisy = np.round(mu * rng.gamma(25.0, 0.04, size=len(mu)))  # CV 0.2
        scaled = hybrid_stepwise_selection(noisy, P)
        unscaled = hybrid_stepwise_selection(noisy, P, dispersion_scaled=False)
        assert len(scaled[0]) <= len(unscaled[0])
        assert {0, 2, 4} <= set(scaled[0].tolist())

    def test_margin_must_be_positive(self, counts_profile, simple_P):
        for bad in (None, 0.0, -1.0):
            with pytest.raises(ValueError, match="min_deviance_gain"):
                hybrid_stepwise_selection(
                    counts_profile, simple_P, min_deviance_gain=bad
                )

    def test_defaults(self):
        import inspect

        sig = inspect.signature(hybrid_stepwise_selection)
        assert sig.parameters["min_deviance_gain"].default == 10.0
        assert sig.parameters["dispersion_scaled"].default is True
        assert sig.parameters["anchor_floor"].default == 0.002
        assert sig.parameters["anchor_min_cost"].default is None
        assert "R" not in sig.parameters and "min_support" not in sig.parameters

    def test_anchor_min_cost_drops_an_anchor_the_fit_does_not_rely_on(
        self, selection_panel
    ):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        # no floor: the unused anchor survives on exposure alone ...
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[1], anchor_floor=None
        )
        assert 1 in idx.tolist()
        # ... and goes once removing it must raise the deviance
        idx, expo, _ = hybrid_stepwise_selection(
            counts,
            selection_panel,
            anchor_indices=[1],
            anchor_floor=None,
            anchor_min_cost=1.0,
        )
        assert sorted(idx.tolist()) == [0, 2, 4]
        assert expo.sum() == pytest.approx(1.0)

    def test_anchor_min_cost_keeps_an_anchor_the_fit_relies_on(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.1, 0.3, 0.0, 0.1]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, anchor_indices=[1], anchor_min_cost=10.0
        )
        assert 1 in idx.tolist()

    def test_anchor_min_cost_zero_drops_only_noise_absorbers(self):
        rng = np.random.default_rng(5)
        P = rng.dirichlet(np.ones(96) * 0.3, size=6).T
        mu = 50000 * (P @ np.array([0.45, 0.0, 0.35, 0.0, 0.2, 0.0]))
        noisy = np.floor(mu * (1 + 0.1 * rng.standard_normal(len(mu))))
        with_floor = hybrid_stepwise_selection(noisy, P, anchor_indices=[1, 5])
        with_cost = hybrid_stepwise_selection(
            noisy, P, anchor_indices=[1, 5], anchor_min_cost=0.0
        )
        assert set(with_cost[0].tolist()) <= set(with_floor[0].tolist())
        assert {0, 2, 4} <= set(with_cost[0].tolist())

    def test_anchor_min_cost_keeps_at_least_two(self, simple_P):
        counts = _counts(simple_P, np.array([0.0, 0.0, 1.0]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, simple_P, anchor_indices=[0, 1], anchor_floor=None,
            anchor_min_cost=1e6,
        )
        assert len(idx) >= 2

    def test_negative_anchor_min_cost_raises(self, counts_profile, simple_P):
        with pytest.raises(ValueError, match="anchor_min_cost"):
            hybrid_stepwise_selection(
                counts_profile, simple_P, anchor_indices=[0], anchor_min_cost=-1.0
            )

    def test_bonferroni_margin_grows_with_the_panel(self):
        assert bonferroni_margin(12) == pytest.approx(6.96, abs=0.01)
        assert bonferroni_margin(101) == pytest.approx(10.85, abs=0.01)
        assert bonferroni_margin(1) < bonferroni_margin(10) < bonferroni_margin(100)
        with pytest.raises(ValueError, match="n_candidates"):
            bonferroni_margin(0)

    def test_bonferroni_margin_selects_the_true_set(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx, _, _ = hybrid_stepwise_selection(
            counts, selection_panel, min_deviance_gain="bonferroni"
        )
        assert sorted(idx.tolist()) == [0, 2, 4]

    def test_unknown_margin_rule_raises(self, counts_profile, simple_P):
        with pytest.raises(ValueError, match="bonferroni"):
            hybrid_stepwise_selection(counts_profile, simple_P, min_deviance_gain="bic")

    def test_indices_refer_to_the_full_panel(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.0, 0.0, 0.6, 0.0, 0.4]))
        idx, expo, _ = hybrid_stepwise_selection(counts, selection_panel)
        assert sorted(idx.tolist()) == [2, 4]
        assert expo == pytest.approx([0.6, 0.4], abs=0.02)


class TestExposureIntervals:
    def test_brackets_the_estimate_and_narrows_with_depth(self, selection_panel):
        for total, width_cap in ((2000, 0.2), (200000, 0.05)):
            counts = _counts(
                selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]), total
            )
            idx, expo, _ = hybrid_stepwise_selection(counts, selection_panel)
            ci = exposure_intervals(counts, selection_panel, idx, B=60, rng=0)
            assert np.all(ci["lower"] <= expo + 1e-9) and np.all(
                expo - 1e-9 <= ci["upper"]
            )
            assert np.all(ci["upper"] - ci["lower"] < width_cap)
            assert ci["replicates"].shape == (len(idx), 60)

    def test_auto_sigma_is_zero_on_exact_data_and_positive_on_noisy(self):
        rng = np.random.default_rng(4)
        P = rng.dirichlet(np.ones(48) * 0.5, size=6).T
        mu = 200000 * (P @ np.array([0.5, 0.0, 0.3, 0.0, 0.2, 0.0]))
        idx = np.array([0, 2, 4])
        assert exposure_intervals(np.round(mu), P, idx, B=10, rng=0)["sigma_b"] == 0.0
        noisy = np.round(mu * rng.gamma(25.0, 0.04, size=len(mu)))
        assert exposure_intervals(noisy, P, idx, B=10, rng=0)["sigma_b"] > 0.05

    def test_fixed_sigma_widens_the_interval(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]), 50000)
        idx = np.array([0, 2, 4])
        narrow = exposure_intervals(
            counts, selection_panel, idx, B=60, overdispersion=None, rng=0
        )
        wide = exposure_intervals(
            counts, selection_panel, idx, B=60, overdispersion=0.2, rng=0
        )
        assert np.all(wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"])

    def test_same_rng_same_interval_and_invalid_sigma_raises(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        idx = np.array([0, 2, 4])
        a = exposure_intervals(counts, selection_panel, idx, B=20, rng=5)
        b = exposure_intervals(counts, selection_panel, idx, B=20, rng=5)
        assert np.array_equal(a["replicates"], b["replicates"])
        for bad in (-0.1, "lots"):
            with pytest.raises(ValueError, match="overdispersion"):
                exposure_intervals(
                    counts, selection_panel, idx, B=5, overdispersion=bad
                )

    def test_single_signature_has_a_degenerate_interval(self, selection_panel):
        counts = _counts(selection_panel, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        ci = exposure_intervals(counts, selection_panel, np.array([0]), B=5, rng=0)
        assert ci["lower"] == pytest.approx([1.0]) and ci["upper"] == pytest.approx(
            [1.0]
        )


class TestSelectionStability:
    def test_true_signatures_are_stable_and_absent_ones_never_picked(
        self, selection_panel
    ):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        freq = selection_stability(counts, selection_panel, B=20, rng=0)
        assert freq.shape == (5,)
        assert np.all(freq[[0, 2, 4]] > 0.9)
        assert np.all(freq[[1, 3]] < 0.2)

    def test_anchors_are_always_selected(self, selection_panel):
        counts = _counts(selection_panel, np.array([0.5, 0.0, 0.3, 0.0, 0.2]))
        freq = selection_stability(
            counts, selection_panel, B=10, rng=0, anchor_indices=[1], anchor_floor=None
        )
        assert freq[1] == 1.0
