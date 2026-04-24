import numpy as np
from scipy.optimize import nnls as _scipy_nnls
from sigconfide.estimates.standard import findSigExposures
from sigconfide.decompose.qp import decomposeQP
from sigconfide.utils.utils import is_wholenumber


def _bootstrap_matrix(m, mutation_count, R, bootstrap_method='multinomial'):
    K = len(m)
    if mutation_count is None:
        if all(is_wholenumber(v) for v in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError(
                "Specify 'mutation_count' or provide integer mutation counts in 'm'."
            )
    m_counts = m * (mutation_count / m.sum())
    m = m / m.sum()

    if bootstrap_method == 'poisson':
        def _sample():
            sampled = np.random.poisson(m_counts)
            total = sampled.sum()
            if total == 0:
                return m  # fallback: degenerate sample
            return sampled / total
        cols = [_sample() for _ in range(R)]
    else:
        cols = [
            np.bincount(np.random.choice(K, size=mutation_count, p=m), minlength=K)
            / mutation_count
            for _ in range(R)
        ]
    return np.column_stack(cols)


def _p_values(exposures, threshold):
    return 1.0 - (exposures > threshold).sum(axis=1) / exposures.shape[1]


def _evaluate(M, P, cols, threshold, decomposition_method):
    exposures, _ = findSigExposures(M, P[:, cols], decomposition_method=decomposition_method)
    return _p_values(exposures, threshold)


def _cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _spa_error_prune(m_norm, P, selected, mandatory_local, threshold, decomposition_method):
    """
    SPA-style iterative removal (mirrors SPA's removeSignatures).

    Each iteration: for every non-mandatory signature, compute the relative
    reconstruction error ε = ||v - Sa||² / ||v||² after removing it.
    Remove the signature whose removal causes the smallest increase in ε,
    provided that increase is ≤ threshold (SPA default: 0.01).
    Repeat until no removal qualifies or only one signature remains.
    """
    selected = set(selected)

    def _eps(cols):
        if len(cols) == 1:
            exp = np.array([np.dot(m_norm, P[:, cols[0]]) / np.dot(P[:, cols[0]], P[:, cols[0]])])
            residual = m_norm - P[:, cols[0]] * exp[0]
        else:
            exp = decomposition_method(m_norm, P[:, cols])
            residual = m_norm - P[:, cols] @ exp
        v_sq = np.dot(m_norm, m_norm)
        return np.dot(residual, residual) / v_sq if v_sq > 1e-12 else 0.0

    while True:
        cols = sorted(selected)
        if len(cols) <= 1:
            break

        eps_base = _eps(cols)

        best_sig, min_eps = None, float('inf')
        for s in cols:
            if s in mandatory_local:
                continue
            trial = [x for x in cols if x != s]
            if not trial:
                continue
            eps_t = _eps(trial)
            if eps_t < min_eps:
                min_eps = eps_t
                best_sig = s

        if best_sig is None:
            break

        if min_eps - eps_base <= threshold:
            selected.discard(best_sig)
        else:
            break

    return selected


def _cosine_prune(m_norm, P, selected, mandatory_local, threshold, decomposition_method):
    """Iteratively remove signatures whose removal costs less than threshold in cosine similarity."""
    selected = set(selected)
    while True:
        cols = sorted(selected)
        if len(cols) <= 1:
            break
        exp, _ = findSigExposures(
            m_norm.reshape(-1, 1), P[:, cols], decomposition_method=decomposition_method
        )
        base_cos = _cosine_sim(m_norm, P[:, cols] @ exp.flatten())

        best_sig, min_loss = None, float('inf')
        for s in cols:
            if s in mandatory_local:
                continue
            trial = [x for x in cols if x != s]
            if not trial:
                continue
            exp_t, _ = findSigExposures(
                m_norm.reshape(-1, 1), P[:, trial], decomposition_method=decomposition_method
            )
            loss = base_cos - _cosine_sim(m_norm, P[:, trial] @ exp_t.flatten())
            if loss < threshold and loss < min_loss:
                min_loss = loss
                best_sig = s

        if best_sig is None:
            break
        selected.discard(best_sig)

    return selected


def _dominant_cleaning(m_norm, P, dominant_threshold, mandatory_local, decomposition_method):
    """
    SigProfilerCleaner-inspired preprocessing.

    Identifies dominant signatures (exposure > dominant_threshold), assigns them
    the maximum plausible attribution via NNLS (largest alpha such that the residual
    stays non-negative), subtracts that attribution, and returns the cleaned catalog
    together with the non-dominant sub-matrix of P.

    Mandatory signatures are never treated as dominant and survive unchanged.

    Returns
    -------
    cleaned_norm        : ndarray (96,)  — re-normalised residual catalog
    sub_P               : ndarray (96, K) — P restricted to non-dominant columns
    dominant_local      : ndarray of int  — column indices in P that were cleaned away
    non_dominant_local  : ndarray of int  — column indices in P that remain in sub_P
    effective_fraction  : float           — fraction of mutations left after subtraction
    """
    N = P.shape[1]
    init_exp = decomposition_method(m_norm, P)

    dominant_mask = init_exp > dominant_threshold
    for idx in mandatory_local:
        dominant_mask[idx] = False  # mandatory sigs are never cleaned

    dominant_local = np.where(dominant_mask)[0]
    non_dominant_local = np.where(~dominant_mask)[0]

    if len(dominant_local) == 0 or len(non_dominant_local) < 2:
        return m_norm, P, np.array([], dtype=int), np.arange(N), 1.0

    # Maximum plausible attribution: NNLS fit of m_norm using only dominant columns
    alpha_dom, _ = _scipy_nnls(P[:, dominant_local], m_norm)
    cleaned_unnorm = np.maximum(m_norm - P[:, dominant_local] @ alpha_dom, 0.0)
    effective_fraction = float(cleaned_unnorm.sum())

    if effective_fraction < 1e-6:
        return m_norm, P, np.array([], dtype=int), np.arange(N), 1.0

    return (
        cleaned_unnorm / effective_fraction,
        P[:, non_dominant_local],
        dominant_local,
        non_dominant_local,
        effective_fraction,
    )


def hybrid_stepwise_selection(
    m,
    P,
    R,
    mutation_count=None,
    threshold=0.01,
    significance_level=0.01,
    decomposition_method=decomposeQP,
    pre_filter_threshold=None,
    mandatory_indices=None,
    bootstrap_method='multinomial',
    start_full=True,
    cosine_prune_threshold=None,
    correlation_penalty=False,
    dominant_cleaning_threshold=None,
    spa_error_prune_threshold=None,
    forward_significance=None,
    backward_significance=None,
):
    """
    pre_filter_threshold : float or None
        If set, run a single cheap QP solve on the original profile first and
        discard signatures whose exposure is below this value before entering
        the bootstrap loop.  Recommended value: 0.001 (zero recall loss on
        typical COSMIC data while reducing N ~4x).  Default: None (disabled).

    mandatory_indices : list of int or None
        Column indices in the original P that are treated as permanently
        active — analogous to SPA's permanent_sigs / background_sigs.
        These signatures survive pre_filter, seed the active set, and are
        never evicted by backward removal or cosine pruning.
        Default: None (disabled).

    cosine_prune_threshold : float or None
        If set, after the bootstrap stepwise loop run a post-hoc greedy pass
        that removes signatures whose removal reduces cosine similarity by
        less than this value.  Default: None (disabled).

    correlation_penalty : bool
        If True, scale the effective significance_level by (1 - max_cosine_sim)
        where max_cosine_sim is the highest cosine similarity between the
        candidate/evaluated signature and the current active set.
        Effect: correlated signatures require stronger bootstrap evidence to
        be added (forward) and are removed with weaker evidence (backward),
        directly targeting false positives caused by signature correlation.
        Default: False.

    dominant_cleaning_threshold : float or None
        If set, applies a SigProfilerCleaner-inspired preprocessing step after
        the optional pre-filter.  Signatures whose initial exposure exceeds this
        value are considered dominant; their maximum plausible attribution is
        subtracted from the catalog via NNLS before the bootstrap loop runs.
        The bootstrap selection then operates on the cleaned residual catalog
        using only non-dominant signatures.  At the end the dominant signatures
        are unconditionally reunited with the bootstrap-selected set and the
        final exposure is refitted on the original catalog.
        Recommended starting value: 0.3 (signatures carrying > 30 % of mutations).
        Mandatory signatures are never cleaned.  Default: None (disabled).

    spa_error_prune_threshold : float or None
        If set, applies a SPA-style post-hoc pruning pass after the bootstrap
        loop (and after cosine pruning, if enabled).  Mirrors SPA's
        removeSignatures: iteratively removes the signature whose removal
        increases relative reconstruction error ε = ||v-Sa||²/||v||² the least,
        as long as that increase does not exceed this threshold.
        Recommended value: 0.01 (matches SPA's internal default).
        Mandatory signatures are never removed.  Default: None (disabled).

    forward_significance : float or None
        Significance level used in the forward (add) step.  If None, falls
        back to significance_level.  Set lower than backward_significance to
        require stronger bootstrap evidence before adding a signature, which
        directly reduces false positives.  Recommended: 0.005.

    backward_significance : float or None
        Significance level used in the backward (remove) step.  If None, falls
        back to significance_level.  Set higher than forward_significance to
        make removal easier.  Recommended: 0.02.
    """
    N = P.shape[1]
    _mandatory = list(mandatory_indices) if mandatory_indices is not None else []

    # --- optional pre-filter ------------------------------------------------
    if pre_filter_threshold is not None:
        m_norm = m / m.sum()
        init_exp = decomposition_method(m_norm, P)
        keep_mask = init_exp > pre_filter_threshold
        for idx in _mandatory:          # mandatory sigs always survive pre-filter
            keep_mask[idx] = True
        keep = np.where(keep_mask)[0]
        if len(keep) < 2:               # safety: need at least 2 sigs
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

    # P at this point is post-pre-filter (or original if pre-filter disabled).
    # Save it so the final exposure refit can address the full selected set.
    P_prefilter = P

    # --- dominant-signature cleaning (SigProfilerCleaner-inspired) ----------
    # dominant_in_prefilter : column indices in P_prefilter that were cleaned
    # cleaning_remap        : maps bootstrap-local indices → P_prefilter indices
    dominant_in_prefilter = np.array([], dtype=int)
    cleaning_remap = None
    m_bootstrap = m          # profile fed to _bootstrap_matrix
    count_bootstrap = mutation_count  # mutation count for bootstrapping
    P_bootstrap = P          # signature matrix for the bootstrap loop

    if dominant_cleaning_threshold is not None:
        m_norm = m / m.sum()
        _cn, _sub_P, _dom, _non_dom, _eff_frac = _dominant_cleaning(
            m_norm, P, dominant_cleaning_threshold, mandatory_local, decomposition_method
        )
        if len(_dom) > 0 and _sub_P.shape[1] >= 2:
            dominant_in_prefilter = _dom
            cleaning_remap = _non_dom
            P_bootstrap = _sub_P
            N = _sub_P.shape[1]
            # Bootstrap uses the cleaned, normalised catalog with an adjusted
            # mutation count so that resampling variance stays realistic.
            m_bootstrap = _cn
            if mutation_count is not None:
                count_bootstrap = max(1, int(round(mutation_count * _eff_frac)))
            elif all(is_wholenumber(v) for v in m):
                count_bootstrap = max(1, int(round(m.sum() * _eff_frac)))
            # else: leave None — _bootstrap_matrix will raise if m is non-integer
            # remap mandatory_local into the bootstrap (non-dominant) index space
            non_dom_list = _non_dom.tolist()
            mandatory_local = {
                non_dom_list.index(i) for i in mandatory_local if i in non_dom_list
            }
    # ------------------------------------------------------------------------

    M = _bootstrap_matrix(m_bootstrap, count_bootstrap, R, bootstrap_method=bootstrap_method)
    # start_full=True  → backward+forward (current default)
    # start_full=False → forward-only from mandatory seeds (higher precision)
    selected = set(range(N)) if start_full else set(mandatory_local)

    _fwd_sig = forward_significance  if forward_significance  is not None else significance_level
    _bwd_sig = backward_significance if backward_significance is not None else significance_level

    # Precompute pairwise cosine similarities between signature columns (used
    # only when correlation_penalty=True — zero cost otherwise).
    if correlation_penalty:
        norms = np.linalg.norm(P_bootstrap, axis=0)
        norms[norms == 0] = 1.0
        P_unit = P_bootstrap / norms
        cos_matrix = P_unit.T @ P_unit   # shape (N, N)

    while True:
        best_benefit = 0.0
        best_move = None
        current_cols = sorted(selected)

        # Backward: try removing one selected signature.
        # Mandatory signatures are protected — skip them.
        if len(selected) > 2:
            pv = _evaluate(M, P_bootstrap, np.array(current_cols), threshold, decomposition_method)
            pv_map = {col: pv[i] for i, col in enumerate(current_cols)}
            for s in selected:
                if s in mandatory_local:
                    continue
                if correlation_penalty:
                    others = list(selected - {s} - mandatory_local)
                    max_sim = float(cos_matrix[s, others].max()) if others else 0.0
                    eff_sig = _bwd_sig * (1.0 - max_sim)
                else:
                    eff_sig = _bwd_sig
                benefit = pv_map[s] - eff_sig
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_move = ('remove', s)

        # Forward: add one discarded signature
        for s in set(range(N)) - selected:
            test_cols = np.array(sorted(selected | {s}))
            pv = _evaluate(M, P_bootstrap, test_cols, threshold, decomposition_method)
            s_pos = list(test_cols).index(s)
            if correlation_penalty and selected:
                max_sim = float(cos_matrix[s, list(selected)].max())
                eff_sig = _fwd_sig * (1.0 - max_sim)
            else:
                eff_sig = _fwd_sig
            benefit = eff_sig - pv[s_pos]
            if benefit > best_benefit:
                best_benefit = benefit
                best_move = ('add', s)

        if best_move is None:
            break

        action, sig = best_move
        if action == 'remove':
            selected.discard(sig)
        else:
            selected.add(sig)

    # optional post-hoc cosine pruning
    if cosine_prune_threshold is not None:
        m_norm = m / m.sum()
        selected = _cosine_prune(
            m_norm, P_bootstrap, selected, mandatory_local,
            cosine_prune_threshold, decomposition_method
        )

    # optional SPA-style ε-based pruning
    if spa_error_prune_threshold is not None:
        m_norm = m / m.sum()
        selected = _spa_error_prune(
            m_norm, P_bootstrap, selected, mandatory_local,
            spa_error_prune_threshold, decomposition_method
        )

    bootstrap_local = np.array(sorted(selected))   # indices in P_bootstrap space

    # --- Remap bootstrap-local indices back to P_prefilter, then to original P
    if cleaning_remap is not None:
        # bootstrap-local → P_prefilter-local (non-dominant columns)
        selected_prefilter = cleaning_remap[bootstrap_local]
        # dominant signatures always rejoin the selected set
        prefilter_local = np.sort(np.concatenate([dominant_in_prefilter, selected_prefilter]))
    else:
        prefilter_local = bootstrap_local

    # P_prefilter-local → original P column indices
    global_indices = keep[prefilter_local] if pre_filter_threshold is not None else prefilter_local

    # Final exposure refit on original m with the full selected signature set
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P_prefilter[:, prefilter_local],
        decomposition_method=decomposition_method
    )
    return global_indices, exposures.flatten(), errors
