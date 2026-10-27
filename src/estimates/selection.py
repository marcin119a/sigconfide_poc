import numpy as np
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.standard import findSigExposures
from sigconfide.utils.utils import is_wholenumber, resolve_rng


def _bootstrap_matrix(m, mutation_count, R, overdispersion=None, rng=None):
    """R bootstrap replicates of the profile, as columns summing to 1.

    Plain multinomial resampling when `overdispersion` is None.  Otherwise each
    replicate first scales every channel by an independent gamma factor with
    mean 1 and coefficient of variation `overdispersion`, then resamples
    multinomially from the scaled profile, so a channel holding c counts
    varies with SD sqrt(c + (overdispersion * c)^2) instead of sqrt(c): the
    multinomial term at low counts, the multiplicative one at high counts.
    """
    rng = resolve_rng(rng)
    K = len(m)
    if mutation_count is None:
        if all(is_wholenumber(v) for v in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError(
                "Specify 'mutation_count' or provide integer mutation counts in 'm'."
            )
    m = m / m.sum()
    if overdispersion is not None and overdispersion < 0:
        raise ValueError(
            "'overdispersion' must be a non-negative coefficient of variation."
        )
    if not overdispersion:
        counts = rng.multinomial(mutation_count, m, size=R)
    else:
        shape = 1.0 / overdispersion**2
        counts = np.empty((R, K))
        for r in range(R):
            p = m * rng.gamma(shape, 1.0 / shape, size=K)
            counts[r] = rng.multinomial(mutation_count, p / p.sum())
    return counts.T / mutation_count


def _support(exposures, threshold):
    """Bootstrap support: fraction of replicates where exposure exceeds threshold."""
    return (exposures > threshold).sum(axis=1) / exposures.shape[1]


def _evaluate(M, P, cols, threshold, decomposition_method):
    exposures, _ = findSigExposures(
        M, P[:, cols], decomposition_method=decomposition_method
    )
    return _support(exposures, threshold)


def _reconstruction_cosine(m_norm, P, cols, decomposition_method):
    exposures = decomposition_method(m_norm, P[:, cols])
    reconstruction = P[:, cols] @ exposures
    denom = np.linalg.norm(m_norm) * np.linalg.norm(reconstruction)
    return float(m_norm @ reconstruction / denom) if denom > 0 else 0.0


def _prune_by_fit_gain(m_norm, P, cols, min_gain, protected, decomposition_method):
    """Drop signatures that the reconstruction does not actually need.

    The bootstrap support measures how *stable* an exposure is, not whether the
    signature earns its place in the fit.  On deep profiles the two come apart:
    bootstrap variance shrinks with the mutation count, so a signature parked at
    a few percent is stably above `threshold` in every replicate and is kept
    even when removing it costs nothing.

    Greedy backward elimination: repeatedly drop the signature whose removal
    costs the least reconstruction cosine, while that cost stays below
    `min_gain`.  Protected (mandatory) signatures are never dropped and at least
    two signatures always survive.
    """
    cols = list(cols)
    while len(cols) > 2:
        base = _reconstruction_cosine(m_norm, P, cols, decomposition_method)
        worst, worst_cost = None, None
        for s in cols:
            if s in protected:
                continue
            kept = [c for c in cols if c != s]
            if len(kept) < 2:
                continue
            cost = base - _reconstruction_cosine(m_norm, P, kept, decomposition_method)
            if worst_cost is None or cost < worst_cost:
                worst, worst_cost = s, cost
        if worst is None or worst_cost >= min_gain:
            break
        cols.remove(worst)
    return cols


def _as_counts(m, mutation_count):
    """The profile as mutation counts, for the deviance-based steps."""
    m = np.asarray(m, dtype=float)
    if mutation_count is None:
        return m  # _bootstrap_matrix has already checked these are whole counts
    return m / m.sum() * mutation_count


def _poisson_deviance(counts, mu):
    """Poisson deviance of observed counts against fitted means ``mu``."""
    mu = np.maximum(mu, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(counts / mu), 0.0)
    return float(2.0 * np.sum(term - (counts - mu)))


def _deviance_of(counts, P, cols, decomposition_method):
    n = counts.sum()
    exposures = decomposition_method(counts / n, P[:, cols])
    return _poisson_deviance(counts, n * (P[:, cols] @ exposures))


def _swap_by_deviance(
    counts, P, cols, min_gain, protected, decomposition_method, max_swaps=50
):
    """Replace an active signature by an inactive one when that lowers the
    Poisson deviance of the fit by more than `min_gain`.

    The bootstrap search measures each signature against the detection
    threshold on its own and never compares two signatures head to head, so a
    signature that is only stable because a better-fitting competitor was
    removed earlier (a flat C>T signature standing in for a trace of APOBEC,
    for instance) is never revisited.  Greedy: one swap per round, the largest
    gain first, until no swap clears the margin.  Protected (mandatory)
    signatures are never swapped out.
    """
    cols = sorted(cols)
    N = P.shape[1]
    seen = {frozenset(cols)}
    for _ in range(max_swaps):
        base = _deviance_of(counts, P, cols, decomposition_method)
        best_gain, best = min_gain, None
        for out in cols:
            if out in protected:
                continue
            rest = [c for c in cols if c != out]
            for into in range(N):
                if into in cols:
                    continue
                cand = sorted(rest + [into])
                if frozenset(cand) in seen:
                    continue
                gain = base - _deviance_of(counts, P, cand, decomposition_method)
                if gain > best_gain:
                    best_gain, best = gain, cand
        if best is None:
            break
        cols = best
        seen.add(frozenset(cols))
    return cols


def _prune_by_deviance(counts, P, cols, min_gain, protected, decomposition_method):
    """Backward elimination on Poisson deviance.

    Same loop as `_prune_by_fit_gain`, but the cost of dropping a signature is
    the rise in Poisson deviance rather than the loss of reconstruction
    cosine.  The cosine is dominated by the high-count channels, so a peaked
    signature carrying a few percent of the profile costs as little as a flat
    one parked on the residual; the deviance weights every channel by its
    expected count and tells the two apart.  A negative cost (dropping the
    signature *lowers* the deviance, which the least-squares QP solution
    allows) always prunes.
    """
    cols = list(cols)
    while len(cols) > 2:
        base = _deviance_of(counts, P, cols, decomposition_method)
        worst, worst_cost = None, None
        for s in cols:
            if s in protected:
                continue
            kept = [c for c in cols if c != s]
            cost = _deviance_of(counts, P, kept, decomposition_method) - base
            if worst_cost is None or cost < worst_cost:
                worst, worst_cost = s, cost
        if worst is None or worst_cost >= min_gain:
            break
        cols.remove(worst)
    return cols


def hybrid_stepwise_selection(
    m,
    P,
    R,
    mutation_count=None,
    threshold=0.01,
    min_support=0.95,
    decomposition_method=decomposeQP,
    pre_filter_threshold=0.001,
    mandatory_indices=None,
    max_iterations=1000,
    min_fit_improvement=None,
    overdispersion=None,
    min_deviance_gain=5.0,
    dispersion_scaled=True,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the active signatures of one sample with a bootstrap stepwise search.

    The bootstrap support of a signature is the fraction of bootstrap
    replicates in which its exposure exceeds ``threshold``. It is a stability
    measure, not a significance test. The search starts from the full
    (pre-filtered) panel and, in each iteration, applies the single best move:
    removing an active signature whose support is below ``min_support``,
    or adding an inactive one whose support (once added) is above it. It stops
    when no move helps, when a move would revisit an already-seen active set
    (oscillation guard) or after ``max_iterations``. The bootstrap replicates
    are drawn once and reused for the whole search.

    Args:
        m (numpy.ndarray): Profile of one sample, shape ``(K,)``; counts or
            probabilities.
        P (numpy.ndarray): Signature panel, shape ``(K, N)``.
        R (int): Number of bootstrap replicates.
        mutation_count (int, optional): Total number of mutations. Required
            when ``m`` does not hold integer counts.
        threshold (float): Exposure above which a signature counts as present
            in a replicate. Default 0.01.
        min_support (float): Bootstrap support cutoff for adding or removing a
            signature. Default 0.95.
        decomposition_method (callable): Solver ``f(m, P) -> exposures``.
            Default ``decomposeQP``.
        pre_filter_threshold (float | None): Run one QP on the original profile
            first and discard signatures with exposure below this value before
            the bootstrap loop. The default 0.001 showed zero recall loss on
            typical COSMIC data while reducing N about 4x. None disables it.
        mandatory_indices (list[int] | None): Column indices of ``P`` that are
            always active: they survive the pre-filter, are present from the
            first iteration and are never removed. Useful for ubiquitous
            signatures such as SBS1 and SBS5.
        max_iterations (int): Hard cap on add/remove moves, a last-resort guard
            against oscillation on degenerate low-count profiles. Default 1000.
        min_fit_improvement (float | None): If set, follow the search with a
            backward elimination that drops any signature whose removal costs
            less than this much reconstruction cosine. The bootstrap asks
            whether an exposure is stable; this asks whether it is needed.
            0.002 was validated on ICGC-BRCA (mean MCC 0.51 -> 0.63).
            Mandatory signatures are exempt. None disables it.
        overdispersion (float | None): Coefficient of variation of a per-channel
            gamma multiplier applied before every bootstrap draw (see
            ``_bootstrap_matrix``), so a channel with ``c`` counts varies with
            SD ``sqrt(c + (overdispersion * c)^2)``. Prevents flat signatures
            from surviving on deep profiles. None means plain multinomial.
        min_deviance_gain (float | None): If set, follow the search with two
            steps scored by the Poisson deviance of the fit, both using this
            margin: a swap step that replaces an active signature by an
            inactive one whenever that lowers the deviance by more than the
            margin, then a backward elimination that drops any signature
            whose removal raises the deviance by less than it. The bootstrap
            asks whether an exposure is stable; this asks whether the
            signature is needed, and unlike ``min_fit_improvement`` it
            weights every channel by its expected count, so a peaked
            signature at a few percent of the profile is kept while a flat
            one parked on the residual is not. 5 (the chi-square scale, one
            degree of freedom) was validated on the in-silico breast
            benchmark and is the default. Mandatory signatures are exempt.
            Runs before ``min_fit_improvement`` when both are set. None
            disables it and recovers the plain bootstrap search.
        dispersion_scaled (bool): Put ``min_deviance_gain`` on the
            quasi-likelihood scale: the margin is multiplied by the dispersion
            ``max(1, deviance / (K - N))`` of the fit against the whole
            pre-filtered panel (the most saturated model, so the estimate
            carries as little misfit as possible). Real catalogues are
            over-dispersed relative to Poisson (deviance per degree of freedom
            of 2--11 on breast whole genomes against 0--0.4 on the synthetic
            benchmarks), and an unscaled margin that filters the latter is a
            no-op on the former. The floor at 1 leaves Poisson-or-quieter
            profiles exactly as with ``dispersion_scaled=False``. Default
            True.
        rng (None | int | SeedSequence | Generator): Source of the bootstrap
            draws. None uses the global ``np.random``. Anything else goes
            through ``np.random.default_rng`` and leaves the global state alone;
            give each sample its own child of
            ``np.random.SeedSequence(seed).spawn(n_samples)`` for reproducible
            parallel runs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``(indices,
        exposures, errors)``: indices of the selected columns in the original
        ``P``, their relative exposures (summing to 1) and the reconstruction
        error (Frobenius norm).
    """
    N = P.shape[1]
    _mandatory = list(mandatory_indices) if mandatory_indices is not None else []
    if min_deviance_gain is not None and min_deviance_gain < 0:
        raise ValueError("'min_deviance_gain' must be a non-negative deviance margin.")

    # --- optional pre-filter ------------------------------------------------
    if pre_filter_threshold is not None:
        m_norm = m / m.sum()
        init_exp = decomposition_method(m_norm, P)
        keep_mask = init_exp > pre_filter_threshold
        for idx in _mandatory:  # mandatory sigs always survive pre-filter
            keep_mask[idx] = True
        keep = np.where(keep_mask)[0]
        if len(keep) < 2:  # safety: need at least 2 sigs
            keep = np.argsort(init_exp)[-2:]
        P = P[:, keep]
        N = P.shape[1]
        # remap mandatory global indices → local indices in filtered P
        keep_list = keep.tolist()
        mandatory_local = set(keep_list.index(i) for i in _mandatory if i in keep_list)
    else:
        keep = None
        mandatory_local = set(_mandatory)
    # ------------------------------------------------------------------------

    M = _bootstrap_matrix(m, mutation_count, R, overdispersion, rng)
    # Mandatory sigs are in `selected` from the start (same as all others since
    # we begin with the full set, but the backward step will never evict them).
    selected = set(range(N))

    # Active sets already visited by the greedy search.  The search moves one
    # signature at a time and can undo an earlier move, so without this the
    # loop can cycle indefinitely (observed on profiles with a handful of
    # mutations, where bootstrap supports flip around the cutoff).
    visited = {frozenset(selected)}

    for _ in range(max_iterations):
        best_benefit = 0.0
        best_move = None
        current_cols = sorted(selected)

        # Backward: try removing one selected signature.
        # Mandatory signatures are protected — skip them.
        if len(selected) > 2:
            sup = _evaluate(
                M, P, np.array(current_cols), threshold, decomposition_method
            )
            sup_map = {col: sup[i] for i, col in enumerate(current_cols)}
            for s in selected:
                if s in mandatory_local:  # ← SPA-style: never evict
                    continue
                benefit = min_support - sup_map[s]
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_move = ("remove", s)

        # Forward: add one discarded signature
        for s in set(range(N)) - selected:
            test_cols = np.array(sorted(selected | {s}))
            sup = _evaluate(M, P, test_cols, threshold, decomposition_method)
            s_pos = list(test_cols).index(s)
            benefit = sup[s_pos] - min_support
            if benefit > best_benefit:
                best_benefit = benefit
                best_move = ("add", s)

        if best_move is None:
            break

        action, sig = best_move
        candidate = selected - {sig} if action == "remove" else selected | {sig}
        if frozenset(candidate) in visited:
            # The best move would return the search to an active set it has
            # already evaluated: the greedy walk is oscillating, so stop here
            # and keep the current set rather than looping forever.
            break

        selected = candidate
        visited.add(frozenset(selected))

    if min_deviance_gain is not None:
        counts = _as_counts(m, mutation_count)
        margin = min_deviance_gain
        if dispersion_scaled:
            df = max(1, P.shape[0] - N)
            phi = _deviance_of(counts, P, list(range(N)), decomposition_method) / df
            margin = min_deviance_gain * max(1.0, phi)
        selected = set(
            _swap_by_deviance(
                counts, P, sorted(selected), margin, mandatory_local,
                decomposition_method,
            )
        )
        selected = set(
            _prune_by_deviance(
                counts, P, sorted(selected), margin, mandatory_local,
                decomposition_method,
            )
        )

    if min_fit_improvement is not None:
        selected = set(
            _prune_by_fit_gain(
                m / m.sum(),
                P,
                sorted(selected),
                min_fit_improvement,
                mandatory_local,
                decomposition_method,
            )
        )

    local_indices = np.array(sorted(selected))

    # map back to original P column indices (identity when pre_filter disabled)
    global_indices = (
        keep[local_indices] if pre_filter_threshold is not None else local_indices
    )
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, local_indices], decomposition_method=decomposition_method
    )
    return global_indices, exposures.flatten(), errors
