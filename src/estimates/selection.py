import numpy as np
from sigconfide.estimates.standard import findSigExposures
from sigconfide.decompose.qp import decomposeQP
from sigconfide.utils.utils import is_wholenumber

# Per-mutation-type tuned defaults.
MUT_TYPE_DEFAULTS = {
    "SBS": {
        "significance_level":      0.05,
        "pre_filter_threshold":    0.001,
        "spa_error_prune":         0.005,
        "mandatory_sig_names":     ["SBS1", "SBS5"],
    },
    "DBS": {
        "significance_level":      0.05,
        "pre_filter_threshold":    0.001,
        "spa_error_prune":         0.005,
        "mandatory_sig_names":     ["DBS2"],
    },
    "ID": {
        "significance_level":      0.05,
        "pre_filter_threshold":    0.001,
        "spa_error_prune":         None,
        "mandatory_sig_names":     ["ID1", "ID2"],
    },
}


def _spa_error_prune(m_norm, P, selected, mandatory_local, threshold, decomposition_method):
    """
    Iteratively remove the signature whose removal causes the smallest increase
    in relative reconstruction error, as long as that increase ≤ threshold.
    Mandatory signatures are never removed.
    """
    selected = set(selected)

    def _eps(cols):
        if len(cols) == 1:
            a = np.dot(m_norm, P[:, cols[0]]) / np.dot(P[:, cols[0]], P[:, cols[0]])
            residual = m_norm - P[:, cols[0]] * a
        else:
            exp = decomposition_method(m_norm, P[:, cols])
            residual = m_norm - P[:, cols] @ exp
        v_sq = np.dot(m_norm, m_norm)
        return np.dot(residual, residual) / v_sq if v_sq > 1e-12 else 0.0

    while True:
        cols = sorted(selected)
        if len(cols) <= 2:
            break
        eps_base = _eps(cols)
        best_sig, min_eps = None, float("inf")
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


def _bootstrap_matrix(m, mutation_count, R):
    K = len(m)
    if mutation_count is None:
        if all(is_wholenumber(v) for v in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError(
                "Specify 'mutation_count' or provide integer mutation counts in 'm'."
            )
    m = m / m.sum()
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


def hybrid_stepwise_selection(
    m,
    P,
    R,
    mutation_count=None,
    threshold=0.01,
    significance_level=0.05,
    decomposition_method=decomposeQP,
    pre_filter_threshold=None,
    mandatory_indices=None,
    spa_error_prune_threshold=None,
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
        These signatures:
          1. survive pre_filter removal,
          2. are present in the active set from the very first bootstrap
             iteration (so QP always decomposes other signatures relative
             to them), and
          3. are skipped in the backward-removal step (cannot be evicted).
        Useful for biologically ubiquitous signatures (e.g. SBS1, SBS5).
        Default: None (disabled).

    spa_error_prune_threshold : float or None
        After bootstrap selection, iteratively remove the signature whose
        removal increases relative reconstruction error the least, as long
        as that increase ≤ threshold.  Reduces false positives caused by
        correlated signatures in noisy data.  Recommended: 0.005.
        Default: None (disabled).
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

    M = _bootstrap_matrix(m, mutation_count, R)
    # Mandatory sigs are in `selected` from the start (same as all others since
    # we begin with the full set, but the backward step will never evict them).
    selected = set(range(N))

    while True:
        best_benefit = 0.0
        best_move = None
        current_cols = sorted(selected)

        # Backward: try removing one selected signature.
        # Mandatory signatures are protected — skip them.
        if len(selected) > 2:
            pv = _evaluate(M, P, np.array(current_cols), threshold, decomposition_method)
            pv_map = {col: pv[i] for i, col in enumerate(current_cols)}
            for s in selected:
                if s in mandatory_local:        # ← SPA-style: never evict
                    continue
                benefit = pv_map[s] - significance_level
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_move = ('remove', s)

        # Forward: add one discarded signature
        for s in set(range(N)) - selected:
            test_cols = np.array(sorted(selected | {s}))
            pv = _evaluate(M, P, test_cols, threshold, decomposition_method)
            s_pos = list(test_cols).index(s)
            benefit = significance_level - pv[s_pos]
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

    local_indices = np.array(sorted(selected))

    # --- optional post-selection error pruning --------------------------------
    if spa_error_prune_threshold is not None and len(local_indices) > 1:
        m_norm = m / m.sum()
        # remap mandatory_local (indices in pre-filtered P) → local sub-matrix positions
        mandatory_in_pruning = {
            i for i, li in enumerate(local_indices.tolist()) if li in mandatory_local
        }
        pruned = _spa_error_prune(
            m_norm, P[:, local_indices],
            set(range(len(local_indices))),
            mandatory_in_pruning,
            spa_error_prune_threshold,
            decomposition_method,
        )
        local_indices = local_indices[np.array(sorted(pruned))]
    # --------------------------------------------------------------------------

    # map back to original P column indices (identity when pre_filter disabled)
    global_indices = keep[local_indices] if pre_filter_threshold is not None else local_indices
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, local_indices], decomposition_method=decomposition_method
    )
    return global_indices, exposures.flatten(), errors
