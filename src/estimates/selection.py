import numpy as np
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

    M = _bootstrap_matrix(m, mutation_count, R, bootstrap_method=bootstrap_method)
    # start_full=True  → backward+forward (current default)
    # start_full=False → forward-only from mandatory seeds (higher precision)
    selected = set(range(N)) if start_full else set(mandatory_local)

    # Precompute pairwise cosine similarities between signature columns (used
    # only when correlation_penalty=True — zero cost otherwise).
    if correlation_penalty:
        norms = np.linalg.norm(P, axis=0)
        norms[norms == 0] = 1.0
        P_unit = P / norms
        cos_matrix = P_unit.T @ P_unit   # shape (N, N)

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
                if s in mandatory_local:
                    continue
                if correlation_penalty:
                    others = list(selected - {s} - mandatory_local)
                    max_sim = float(cos_matrix[s, others].max()) if others else 0.0
                    eff_sig = significance_level * (1.0 - max_sim)
                else:
                    eff_sig = significance_level
                benefit = pv_map[s] - eff_sig
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_move = ('remove', s)

        # Forward: add one discarded signature
        for s in set(range(N)) - selected:
            test_cols = np.array(sorted(selected | {s}))
            pv = _evaluate(M, P, test_cols, threshold, decomposition_method)
            s_pos = list(test_cols).index(s)
            if correlation_penalty and selected:
                max_sim = float(cos_matrix[s, list(selected)].max())
                eff_sig = significance_level * (1.0 - max_sim)
            else:
                eff_sig = significance_level
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
            m_norm, P, selected, mandatory_local,
            cosine_prune_threshold, decomposition_method
        )

    local_indices = np.array(sorted(selected))

    # map back to original P column indices (identity when pre_filter disabled)
    global_indices = keep[local_indices] if pre_filter_threshold is not None else local_indices
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, local_indices], decomposition_method=decomposition_method
    )
    return global_indices, exposures.flatten(), errors
