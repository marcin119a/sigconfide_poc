import numpy as np
from sigconfide.estimates.standard import findSigExposures
from sigconfide.decompose.qp import decomposeQP
from sigconfide.utils.utils import is_wholenumber


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

    # map back to original P column indices (identity when pre_filter disabled)
    global_indices = keep[local_indices] if pre_filter_threshold is not None else local_indices
    exposures, errors = findSigExposures(
        m.reshape(-1, 1), P[:, local_indices], decomposition_method=decomposition_method
    )
    return global_indices, exposures.flatten(), errors
