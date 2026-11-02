"""Selection of a sample's active signatures and the uncertainty of their exposures.

The active set is found by a forward-backward search on the Poisson deviance of
the fit: starting from the anchor signatures, a signature is added when it
lowers the deviance by at least a margin and removed when it accounts for less
than that margin, the margin being ``min_deviance_gain`` (a number, or the
Bonferroni margin for the size of the panel) scaled by the quasi-Poisson
dispersion of the saturated fit.  The search is deterministic.
The bootstrap enters only afterwards: ``exposure_intervals`` reports percentile
intervals of the selected exposures and ``selection_stability`` the fraction of
replicates in which each signature is selected again, both from an
over-dispersed resampling of the profile whose dispersion is estimated on the
selected set.
"""

import numpy as np
from scipy.stats import chi2
from sigconfide.decompose.qp import decomposeQP
from sigconfide.estimates.standard import findSigExposures
from sigconfide.utils.utils import is_wholenumber, resolve_rng


# ---------------------------------------------------------------- profiles ---
def _mutation_count(m, mutation_count):
    """The total mutation count: as given, or the sum of whole-number counts."""
    if mutation_count is not None:
        return mutation_count
    if all(is_wholenumber(v) for v in m):
        return int(m.sum())
    raise ValueError(
        "Specify 'mutation_count' or provide integer mutation counts in 'm'."
    )


def _as_counts(m, mutation_count):
    """The profile as mutation counts."""
    m = np.asarray(m, dtype=float)
    return m / m.sum() * _mutation_count(m, mutation_count)


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


# ---------------------------------------------------------------- deviance ---
def _poisson_deviance(counts, mu):
    """Poisson deviance of observed counts against fitted means ``mu``."""
    mu = np.maximum(mu, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(counts / mu), 0.0)
    return float(2.0 * np.sum(term - (counts - mu)))


def _fit_deviance(counts, P, cols, decomposition_method):
    """Exposures of the fit against columns ``cols`` and its Poisson deviance."""
    n = counts.sum()
    exposures = decomposition_method(counts / n, P[:, cols])
    return exposures, _poisson_deviance(counts, n * (P[:, cols] @ exposures))


def _deviance_of(counts, P, cols, decomposition_method):
    return _fit_deviance(counts, P, cols, decomposition_method)[1]


def _saturated_fit(counts, P, anchors, decomposition_method):
    """The saturated fit: exposures against the whole panel, its active columns
    (positive exposure, plus the anchors; at least two) and its quasi-Poisson
    dispersion ``phi = deviance / (K - p)`` taken on those columns.

    A panel can hold more signatures than contexts, so the fit against all of
    them is not identifiable; the columns the QP actually uses are the largest
    identifiable model and carry the dispersion estimate.
    """
    n = counts.sum()
    exposures = decomposition_method(counts / n, P)
    active = set(np.where(exposures > 0)[0].tolist()) | set(anchors)
    if len(active) < 2:
        active |= set(int(j) for j in np.argsort(exposures)[-2:])
    cols = sorted(active)
    df = max(1, P.shape[0] - len(cols))
    phi = _deviance_of(counts, P, cols, decomposition_method) / df
    return exposures, np.array(cols), phi


def saturated_dispersion(
    m, P, mutation_count=None, anchor_indices=None, decomposition_method=decomposeQP
) -> float:
    """Quasi-Poisson dispersion of the saturated fit, ``deviance / (K - p)``.

    ``p`` counts the columns the fit against the whole panel actually uses
    (positive exposure) plus the anchors. It is 1 for Poisson counts, below 1
    for a catalogue quieter than Poisson and above 1 for an over-dispersed
    one; ``hybrid_stepwise_selection`` multiplies its margin by ``max(1, phi)``.
    """
    counts = _as_counts(m, mutation_count)
    anchors = [int(i) for i in (anchor_indices or [])]
    return float(_saturated_fit(counts, P, anchors, decomposition_method)[2])


def estimate_overdispersion(
    m, P, mutation_count=None, decomposition_method=decomposeQP, leverage=False
) -> float:
    """Estimate the bootstrap over-dispersion ``sigma_b`` from the fit residual.

    The over-dispersed bootstrap varies a channel with expected count ``c``
    by ``Var = c + (sigma_b c)^2``, which is linear in ``sigma_b^2``, so the
    residual of a fit gives ``sigma_b`` by the method of moments (the classic
    negative-binomial dispersion estimator).  With ``c_k = n (P e)_k`` the
    fitted counts of the QP solution against the ``p`` columns of ``P`` and
    ``d_k = X_k - c_k`` the residuals,

        sigma_b^2 = max{ 0, ( K/(K-p) * sum d_k^2 - sum c_k ) / sum c_k^2 }.

    The part of the residual that counting noise explains (``sum c_k``) is
    taken out and the rest is normalised by ``sum c_k^2``; ``K/(K-p)`` is
    the degrees-of-freedom correction for the fitted parameters.  A residual
    within Poisson noise gives 0, which recovers the plain multinomial
    bootstrap.

    The fit absorbs part of the noise, most of it in the high-count channels
    the active signatures peak on, which the uniform ``K/(K-p)`` factor
    understates.  With ``leverage=True`` each squared residual is instead
    divided by ``1 - h_k``, ``h_k`` being the leverage of channel ``k`` in the
    least-squares projection onto the columns the fit actually used, and the
    ``K/(K-p)`` factor is dropped.  Taken on the selected set, this is the
    estimate behind ``exposure_intervals`` and ``selection_stability``.

    Args:
        m (numpy.ndarray): Profile of one sample, shape ``(K,)``; counts or
            probabilities.
        P (numpy.ndarray): The panel to fit, shape ``(K, p)``: the set the
            residual is taken against.
        mutation_count (int, optional): Total number of mutations. Required
            when ``m`` does not hold integer counts.
        decomposition_method (callable): Solver ``f(m, P) -> exposures``.
            Default ``decomposeQP``.
        leverage (bool): Correct each residual for the leverage of its channel
            instead of applying the uniform ``K/(K-p)`` factor. Default False.

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
    if leverage:
        active = P[:, exposures > 0]
        hat = active @ np.linalg.pinv(active.T @ active) @ active.T
        h = np.clip(np.diag(hat), 0.0, 0.99)
        resid = np.sum(d**2 / (1.0 - h))
    else:
        resid = K / max(1, K - p) * np.sum(d**2)
    sigma2 = (resid - np.sum(c)) / scale
    return float(np.sqrt(max(0.0, sigma2)))


def bonferroni_margin(n_candidates, alpha=0.05) -> float:
    """The deviance margin that bounds the family-wise false-positive rate of
    the search at ``alpha`` under Poisson noise, for a panel of
    ``n_candidates`` signatures.
    """
    if n_candidates < 1:
        raise ValueError("'n_candidates' must be at least 1.")
    return float(chi2.isf(min(1.0, 2.0 * alpha / n_candidates), 1))


# ---------------------------------------------------------------- search -----
def _forward_backward(
    counts, P, start, margin, protected, decomposition_method, max_iterations
):
    """Best-first forward-backward search on the Poisson deviance.

    From ``start``, each iteration scores every single move: adding an
    inactive signature, whose benefit is its deviance gain over the margin,
    ``(gain - margin) / margin``, and removing an active unprotected one,
    whose benefit is the margin over its cost, ``(margin - cost) / margin``;
    the move with the largest positive benefit is applied.  The search stops
    when no move has a positive benefit, when the best move would revisit a
    set already seen (oscillation guard) or after ``max_iterations``.  At
    least two signatures stay active.
    """
    cols = sorted(start)
    active = set(cols)
    N = P.shape[1]
    visited = {frozenset(cols)}
    for _ in range(max_iterations):
        base = _deviance_of(counts, P, cols, decomposition_method)
        best_benefit, best_move = 0.0, None
        if len(cols) > 2:
            for s in cols:
                if s in protected:
                    continue
                kept = [c for c in cols if c != s]
                cost = _deviance_of(counts, P, kept, decomposition_method) - base
                benefit = (margin - cost) / margin
                if benefit > best_benefit:
                    best_benefit, best_move = benefit, ("remove", s)
        for s in range(N):
            if s in active:
                continue
            gain = base - _deviance_of(
                counts, P, sorted(cols + [s]), decomposition_method
            )
            benefit = (gain - margin) / margin
            if benefit > best_benefit:
                best_benefit, best_move = benefit, ("add", s)
        if best_move is None:
            break
        action, s = best_move
        candidate = active - {s} if action == "remove" else active | {s}
        if frozenset(candidate) in visited:
            break
        active = candidate
        cols = sorted(active)
        visited.add(frozenset(cols))
    return cols


def _anchor_check(
    m, counts, P, cols, protected, floor, min_cost, decomposition_method
):
    """Drop anchor signatures the final fit does not use.

    An anchor signature is carried through the search unconditionally so
    that a trace of it is fitted at all and a look-alike cannot stand in for
    it; this last step asks whether the fit actually used it.  An anchor
    fails when its relative exposure is below ``floor`` or, with ``min_cost``
    given, when removing it from the final set raises the Poisson deviance by
    less than ``min_cost`` (a negative cost means the fit without it is the
    better one: the anchor only absorbed noise).  
    """
    cols = list(cols)
    while True:
        exposures, errors = findSigExposures(
            m.reshape(-1, 1), P[:, cols], decomposition_method=decomposition_method
        )
        exposures = exposures.flatten()
        if len(cols) <= 2:
            break
        base = None
        failing = []
        for i, c in enumerate(cols):
            if c not in protected:
                continue
            bad = floor is not None and exposures[i] < floor
            if not bad and min_cost is not None:
                if base is None:
                    base = _deviance_of(counts, P, cols, decomposition_method)
                kept = [x for x in cols if x != c]
                cost = _deviance_of(counts, P, kept, decomposition_method) - base
                bad = cost < min_cost
            if bad:
                failing.append((exposures[i], i))
        if not failing:
            break
        cols.pop(min(failing)[1])
    return np.array(cols), exposures, errors


def hybrid_stepwise_selection(
    m,
    P,
    mutation_count=None,
    anchor_indices=None,
    min_deviance_gain=10.0,
    dispersion_scaled=True,
    anchor_floor=0.002,
    anchor_min_cost=None,
    decomposition_method=decomposeQP,
    max_iterations=200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the active signatures of one sample by a forward-backward search
    on the Poisson deviance of the fit.

    The search starts from the anchor signatures (or, with fewer than two
    anchors, from the strongest columns of the saturated fit) and considers
    every single move over the whole panel: a signature enters when it lowers
    the Poisson deviance of the fit by at least the margin and leaves when it
    accounts for less than the margin, the best move first, until none
    qualifies (see ``_forward_backward``).  The margin is
    ``min_deviance_gain * max(1, phi)`` with ``phi`` the quasi-Poisson
    dispersion of the saturated fit (``saturated_dispersion``), so a
    catalogue that is noisier than Poisson needs a proportionally larger gain.
    The procedure is deterministic: no resampling, no seed.

    Args:
        m (numpy.ndarray): Profile of one sample, shape ``(K,)``; counts or
            probabilities.
        P (numpy.ndarray): Signature panel, shape ``(K, N)``.
        mutation_count (int, optional): Total number of mutations. Required
            when ``m`` does not hold integer counts.
        anchor_indices (list[int] | None): Column indices of ``P`` anchored
            in the model: the search starts from them and never removes them.
            They face one check at the end, see ``anchor_floor``. Useful for
            ubiquitous signatures such as SBS1 and SBS5.
        min_deviance_gain (float | str): The deviance margin ``gamma_0`` a
            signature must clear, before the dispersion scaling. Default 10:
            under Poisson noise the gain of an absent signature is at most a
            ``chi2_1`` draw, which exceeds 10 with probability 0.0016, about
            the 95th percentile of the largest gain among a hundred absent
            candidates, and about ``log n`` (the BIC penalty) for a few
            thousand mutations. 
        dispersion_scaled (bool): Multiply the margin by ``max(1, phi)``, the
            quasi-Poisson dispersion of the saturated fit, so that the margin
            keeps its meaning on over-dispersed catalogues; Poisson-or-quieter
            profiles are left at ``min_deviance_gain``. Default True.
        anchor_floor (float | None): Relative exposure below which an anchor
            signature is dropped from the final answer: the anchor is carried
            through the search unconditionally, so that a trace of SBS1 is
            fitted at all and a look-alike cannot stand in for SBS5; this last
            step asks whether the final fit actually used it. Columns below
            the floor go one at a time, the smallest first, with a refit after
            each; at least two signatures survive. Default 0.002; None keeps
            every anchor.
        anchor_min_cost (float | None): Second condition of that check, off
            by default: an anchor is also dropped when removing it from the
            final set raises the Poisson deviance by less than this margin
            (scaled by ``max(1, phi)`` like the search margin when
            ``dispersion_scaled``). 
        decomposition_method (callable): Solver ``f(m, P) -> exposures``.
            Default ``decomposeQP``.
        max_iterations (int): Cap on the number of moves. Default 200.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``(indices,
        exposures, errors)``: indices of the selected columns of ``P``, their
        relative exposures (summing to 1) and the reconstruction error
        (Frobenius norm).
    """
    m = np.asarray(m, dtype=float)
    if isinstance(min_deviance_gain, str):
        if min_deviance_gain != "bonferroni":
            raise ValueError(
                "'min_deviance_gain' must be a positive deviance margin or 'bonferroni'."
            )
        min_deviance_gain = bonferroni_margin(P.shape[1])
    if min_deviance_gain is None or min_deviance_gain <= 0:
        raise ValueError("'min_deviance_gain' must be a positive deviance margin.")
    if anchor_min_cost is not None and anchor_min_cost < 0:
        raise ValueError("'anchor_min_cost' must be a non-negative deviance margin.")
    anchors = sorted(set(int(i) for i in (anchor_indices or [])))
    counts = _as_counts(m, mutation_count)
    saturated, _, phi = _saturated_fit(counts, P, anchors, decomposition_method)
    scale = max(1.0, phi) if dispersion_scaled else 1.0
    margin = min_deviance_gain * scale

    start = list(anchors)
    for j in np.argsort(saturated)[::-1]:
        if len(start) >= 2:
            break
        if int(j) not in start:
            start.append(int(j))

    cols = _forward_backward(
        counts, P, start, margin, set(anchors), decomposition_method, max_iterations
    )
    indices = np.array(sorted(cols))
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, indices], decomposition_method=decomposition_method
    )
    exposures = exposures.flatten()
    if anchors and (anchor_floor is not None or anchor_min_cost is not None):
        indices, exposures, errors = _anchor_check(
            m,
            counts,
            P,
            indices,
            set(anchors),
            anchor_floor,
            None if anchor_min_cost is None else anchor_min_cost * scale,
            decomposition_method,
        )
    return indices, exposures, errors


# ---------------------------------------------------------------- bootstrap --
def _resolve_sigma(counts, Pf, overdispersion, decomposition_method):
    if isinstance(overdispersion, str):
        if overdispersion != "auto":
            raise ValueError(
                "'overdispersion' must be a non-negative number, 'auto' or None."
            )
        if Pf.shape[1] < 2:
            return 0.0
        return estimate_overdispersion(
            counts, Pf, int(round(counts.sum())), decomposition_method, leverage=True
        )
    if overdispersion is None:
        return 0.0
    if overdispersion < 0:
        raise ValueError(
            "'overdispersion' must be a non-negative number, 'auto' or None."
        )
    return float(overdispersion)


def exposure_intervals(
    m,
    P,
    indices,
    B=100,
    overdispersion="auto",
    level=0.90,
    mutation_count=None,
    decomposition_method=decomposeQP,
    rng=None,
) -> dict:
    """Bootstrap percentile intervals of the exposures of a selected set.

    ``B`` replicates of the profile are drawn (``_bootstrap_matrix``) and the
    selected signatures are refitted on each; the interval of a signature is
    the central ``level`` share of its replicate exposures.  With
    ``overdispersion="auto"`` the replicates are over-dispersed by the
    ``sigma_b`` estimated on the selected set with the leverage-corrected
    method of moments (``estimate_overdispersion``), which is 0, the plain
    multinomial bootstrap, when the residual is within counting noise; a
    number fixes ``sigma_b`` and None is the multinomial bootstrap.

    Returns:
        dict: ``lower`` and ``upper`` bounds and ``sd`` of the relative
        exposures (arrays aligned with ``indices``), ``sigma_b`` used, and the
        ``replicates`` matrix, shape ``(len(indices), B)``.
    """
    m = np.asarray(m, dtype=float)
    indices = np.asarray(indices, dtype=int)
    counts = _as_counts(m, mutation_count)
    n = int(round(counts.sum()))
    Pf = P[:, indices]
    sigma = _resolve_sigma(counts, Pf, overdispersion, decomposition_method)
    if len(indices) == 1:
        rep = np.ones((1, B))
    else:
        M = _bootstrap_matrix(counts, n, B, sigma if sigma > 0 else None, rng)
        rep, _ = findSigExposures(M, Pf, decomposition_method=decomposition_method)
    alpha = (1.0 - level) / 2.0
    return {
        "lower": np.quantile(rep, alpha, axis=1),
        "upper": np.quantile(rep, 1.0 - alpha, axis=1),
        "sd": rep.std(axis=1, ddof=1) if B > 1 else np.zeros(len(indices)),
        "sigma_b": sigma,
        "replicates": rep,
    }


def selection_stability(
    m,
    P,
    B=100,
    overdispersion="auto",
    mutation_count=None,
    rng=None,
    **selection_kwargs,
) -> np.ndarray:
    """Fraction of bootstrap replicates in which each signature is selected.

    The selection is run once on the profile and then again on each of ``B``
    replicates (over-dispersed by the ``sigma_b`` of the selected set, as in
    ``exposure_intervals``); the result, one value per column of ``P``, is
    how often the search picks that signature under resampling.  Keyword
    arguments go to ``hybrid_stepwise_selection``.
    """
    m = np.asarray(m, dtype=float)
    counts = _as_counts(m, mutation_count)
    n = int(round(counts.sum()))
    method = selection_kwargs.get("decomposition_method", decomposeQP)
    indices, _, _ = hybrid_stepwise_selection(
        counts, P, mutation_count=n, **selection_kwargs
    )
    sigma = _resolve_sigma(counts, P[:, indices], overdispersion, method)
    M = _bootstrap_matrix(counts, n, B, sigma if sigma > 0 else None, rng)
    freq = np.zeros(P.shape[1])
    for b in range(B):
        idx, _, _ = hybrid_stepwise_selection(
            M[:, b] * n, P, mutation_count=n, **selection_kwargs
        )
        freq[idx] += 1.0
    return freq / B
