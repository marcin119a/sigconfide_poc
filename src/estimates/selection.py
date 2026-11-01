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
