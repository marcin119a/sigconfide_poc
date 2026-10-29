import numpy as np
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.standard import findSigExposures
from sigconfide.utils.utils import is_wholenumber, resolve_rng


def _mutation_count(m, mutation_count):
    """The total mutation count: as given, or the sum of whole-number counts."""
    if mutation_count is not None:
        return mutation_count
    if all(is_wholenumber(v) for v in m):
        return int(m.sum())
    raise ValueError(
        "Specify 'mutation_count' or provide integer mutation counts in 'm'."
    )


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
    mutation_count = _mutation_count(m, mutation_count)
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


def _as_counts(m, mutation_count):
    """The profile as mutation counts, for the residual-based steps."""
    m = np.asarray(m, dtype=float)
    return m / m.sum() * _mutation_count(m, mutation_count)


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


def estimate_overdispersion(
    m, P, mutation_count=None, decomposition_method=decomposeQP
) -> float:
    """Estimate the bootstrap over-dispersion ``sigma_b`` from the fit residual.

    The over-dispersed bootstrap varies a channel with expected count ``c``
    by ``Var = c + (sigma_b c)^2``, which is linear in ``sigma_b^2``, so the
    residual of a fit gives ``sigma_b`` by the method of moments (the classic
    negative-binomial dispersion estimator).  With ``c_k = n (P e)_k`` the
    fitted counts of the QP solution against all ``p`` columns of ``P`` and
    ``d_k = X_k - c_k`` the residuals,

        sigma_b^2 = max{ 0, ( K/(K-p) * sum d_k^2 - sum c_k ) / sum c_k^2 }.

    The part of the residual that counting noise explains (``sum c_k``) is
    taken out and the rest is normalised by ``sum c_k^2``; ``K/(K-p)`` is
    the degrees-of-freedom correction for the fitted parameters.  A residual
    within Poisson noise gives 0, which recovers the plain multinomial
    bootstrap.

    The estimate is noisy at small ``n``, but there the ``sqrt(c)`` term
    dominates the replicates and ``sigma_b`` hardly matters.  At large ``n``,
    where ``sigma_b`` decides whether flat signatures survive, ``sum c_k^2``
    is large and the estimate is precise.

    Args:
        m (numpy.ndarray): Profile of one sample, shape ``(K,)``; counts or
            probabilities.
        P (numpy.ndarray): The panel to fit, shape ``(K, p)``: the active set
            the residual is taken against. ``hybrid_stepwise_selection``
            passes the pre-filtered panel (the most saturated model, so the
            estimate carries as little misfit as possible).
        mutation_count (int, optional): Total number of mutations. Required
            when ``m`` does not hold integer counts.
        decomposition_method (callable): Solver ``f(m, P) -> exposures``.
            Default ``decomposeQP``.

    Returns:
        float: ``sigma_b``, a non-negative coefficient of variation.
    """
    counts = _as_counts(m, mutation_count)
    n = counts.sum()
    K, p = P.shape
    exposures = decomposition_method(counts / n, P)
    c = n * (P @ exposures)
    d = counts - c
    scale = np.sum(c**2)
    if scale <= 0:
        return 0.0
    df = max(1, K - p)
    sigma2 = (K / df * np.sum(d**2) - np.sum(c)) / scale
    return float(np.sqrt(max(0.0, sigma2)))


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
    gain first, until no swap clears the margin.  Protected (anchor)
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

    Greedy loop: repeatedly drop the signature whose removal raises the
    Poisson deviance least, while that rise stays below `min_gain`.
    Protected (anchor) signatures are never dropped and at least two
    signatures always survive.  A reconstruction error such as the cosine
    similarity is dominated by the high-count channels, so a peaked signature
    carrying a few percent of the profile costs as little as a flat one
    parked on the residual; the deviance weights every channel by its
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


def _apply_anchor_floor(
    m, P, cols, exposures, errors, protected, floor, decomposition_method
):
    """Drop anchor signatures whose final relative exposure is below `floor`.

    An anchor signature is carried through the search unconditionally so
    that a trace of it is fitted at all and a look-alike cannot stand in for
    it; this last step asks whether the fit actually used it.  One column at
    a time, the smallest exposure first, refitting after each removal because
    the remaining exposures shift; at least two signatures always survive.
    """
    cols = np.asarray(cols)
    while len(cols) > 2:
        below = [
            (exposures[i], i)
            for i, c in enumerate(cols)
            if c in protected and exposures[i] < floor
        ]
        if not below:
            break
        _, i_drop = min(below)
        cols = np.delete(cols, i_drop)
        exposures, errors = findSigExposures(
            m.reshape(-1, 1), P[:, cols], decomposition_method=decomposition_method
        )
        exposures = exposures.flatten()
    return cols, exposures, errors


def hybrid_stepwise_selection(
    m,
    P,
    R,
    mutation_count=None,
    threshold=0.01,
    min_support=0.95,
    decomposition_method=decomposeQP,
    pre_filter_threshold=0.001,
    anchor_indices=None,
    max_iterations=1000,
    overdispersion=None,
    min_deviance_gain=5.0,
    dispersion_scaled=True,
    anchor_floor=0.002,
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
        anchor_indices (list[int] | None): Column indices of ``P`` anchored
            in the model: they survive the pre-filter, are present from the
            first iteration and are never removed by the search. They face
            one check at the end, see ``anchor_floor``. Useful for ubiquitous
            signatures such as SBS1 and SBS5.
        max_iterations (int): Hard cap on add/remove moves, a last-resort guard
            against oscillation on degenerate low-count profiles. Default 1000.
        overdispersion (float | "auto" | None): Coefficient of variation of a
            per-channel gamma multiplier applied before every bootstrap draw
            (see ``_bootstrap_matrix``), so a channel with ``c`` counts varies
            with SD ``sqrt(c + (overdispersion * c)^2)``. Prevents flat
            signatures from surviving on deep profiles. ``"auto"`` estimates
            it from the residual of the fit against the pre-filtered panel
            by the method of moments (see ``estimate_overdispersion``): 0,
            and hence the plain multinomial bootstrap, when the residual is
            within counting noise, the excess dispersion of the catalogue
            otherwise. None means plain multinomial.
        min_deviance_gain (float | None): If set, follow the search with two
            steps scored by the Poisson deviance of the fit, both using this
            margin: a swap step that replaces an active signature by an
            inactive one whenever that lowers the deviance by more than the
            margin, then a backward elimination that drops any signature
            whose removal raises the deviance by less than it. The bootstrap
            asks whether an exposure is stable; this asks whether the
            signature is needed, and it weights every channel by its
            expected count, so a peaked signature at a few percent of the
            profile is kept while a flat one parked on the residual is not. 5.
            None disables it and recovers the plain bootstrap search.
        dispersion_scaled (bool): Put ``min_deviance_gain`` on the
            quasi-likelihood scale: the margin is multiplied by the dispersion
            ``max(1, deviance / (K - N))`` of the fit against the whole
            pre-filtered panel (the most saturated model, so the estimate
            carries as little misfit as possible). Real catalogues are
            over-dispersed relative to Poisson and an unscaled margin that
            filters the latter is a no-op on the former. The floor at 1
            leaves Poisson-or-quieter profiles exactly as with
            ``dispersion_scaled=False``. Default True.
        anchor_floor (float | None): Relative exposure below which an
            anchor signature is dropped from the final answer. The anchor
            set is carried through the search unconditionally, so that a
            trace of SBS1 is fitted at all and a look-alike cannot stand in
            for SBS5; this last step then asks whether the final fit
            actually used the anchored column. Columns below the floor are
            removed one at a time, the smallest first, with a refit after
            each, and at least two signatures always survive. Default 0.002.
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
    _anchors = list(anchor_indices) if anchor_indices is not None else []
    if min_deviance_gain is not None and min_deviance_gain < 0:
        raise ValueError("'min_deviance_gain' must be a non-negative deviance margin.")

    # --- optional pre-filter ------------------------------------------------
    if pre_filter_threshold is not None:
        m_norm = m / m.sum()
        init_exp = decomposition_method(m_norm, P)
        keep_mask = init_exp > pre_filter_threshold
        for idx in _anchors:  # anchor sigs always survive pre-filter
            keep_mask[idx] = True
        keep = np.where(keep_mask)[0]
        if len(keep) < 2:  # safety: need at least 2 sigs
            keep = np.argsort(init_exp)[-2:]
        P = P[:, keep]
        N = P.shape[1]
        # remap anchor global indices → local indices in filtered P
        keep_list = keep.tolist()
        anchor_local = set(keep_list.index(i) for i in _anchors if i in keep_list)
    else:
        keep = None
        anchor_local = set(_anchors)
    # ------------------------------------------------------------------------

    if isinstance(overdispersion, str):
        if overdispersion != "auto":
            raise ValueError(
                "'overdispersion' must be a non-negative number, 'auto' or None."
            )
        overdispersion = estimate_overdispersion(
            m, P, mutation_count, decomposition_method
        )
    M = _bootstrap_matrix(m, mutation_count, R, overdispersion, rng)
    # Anchor sigs are in `selected` from the start (same as all others since
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
        # Anchor signatures are protected — skip them.
        if len(selected) > 2:
            sup = _evaluate(
                M, P, np.array(current_cols), threshold, decomposition_method
            )
            sup_map = {col: sup[i] for i, col in enumerate(current_cols)}
            for s in selected:
                if s in anchor_local:  # anchors are never evicted
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
                counts,
                P,
                sorted(selected),
                margin,
                anchor_local,
                decomposition_method,
            )
        )
        selected = set(
            _prune_by_deviance(
                counts,
                P,
                sorted(selected),
                margin,
                anchor_local,
                decomposition_method,
            )
        )

    local_indices = np.array(sorted(selected))
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, local_indices], decomposition_method=decomposition_method
    )
    exposures = exposures.flatten()

    if anchor_floor is not None and anchor_local:
        local_indices, exposures, errors = _apply_anchor_floor(
            m,
            P,
            local_indices,
            exposures,
            errors,
            anchor_local,
            anchor_floor,
            decomposition_method,
        )

    # map back to original P column indices (identity when pre_filter disabled)
    global_indices = (
        keep[local_indices] if pre_filter_threshold is not None else local_indices
    )
    return global_indices, exposures, errors
