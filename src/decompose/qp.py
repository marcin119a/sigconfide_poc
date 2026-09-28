import numpy as np
import quadprog


def _qp_constraints(P):
    """Quadratic-programming setup that depends only on the panel `P`.

    Returns (G, C, b): the Gram matrix of the objective and the constraints
    `sum(x) == 1`, `x >= 0`.  None of it depends on the profile being fitted,
    so a caller fitting many profiles against the same `P` can build it once.
    """
    # N: how many signatures are selected
    N = P.shape[1]
    # G: matrix appearing in the quadratic programming objective function
    G = np.dot(P.T, P).astype(float)
    # quadprog requires G to be strictly positive definite. When P holds more
    # signatures than mutation contexts (e.g. the full 96x101 COSMIC v3.6
    # panel) its columns are linearly dependent, so G is only positive
    # semi-definite and the Cholesky factorisation inside solve_qp fails. Only
    # in that rank-deficient case add a tiny ridge to lift the zero
    # eigenvalues; it is scaled to G so it stays negligible and independent of
    # the data scale. Well-conditioned panels are left untouched.
    if N > P.shape[0]:
        G.flat[:: N + 1] += 1e-9 * np.trace(G) / N
    # C: matrix constraints under which we want to minimize the quadratic
    # programming objective function.
    C = np.column_stack([np.ones(N), np.eye(N)]).astype(float)
    # b: vector containing the values of b_0.
    b = np.array([1] + [0] * N).astype(float)
    return G, C, b


def _solve_qp(G, C, b, d):
    # Solve quadratic programming problem
    out = quadprog.solve_qp(G, d, C, b, meq=1)

    # Some exposure values are negative, but very close to 0
    # Change these negative values to zero and renormalize
    exposures = out[0]
    exposures[exposures < 0] = 0
    exposures /= sum(exposures)

    # return the exposures
    return exposures


def decomposeQP(m, P):
    G, C, b = _qp_constraints(P)
    # d: vector appearing in the quadratic programming objective function
    d = np.dot(m.T, P).astype(float)
    return _solve_qp(G, C, b, d)


def decomposeQP_batch(M, P):
    """`decomposeQP` for every column of `M` against the same panel `P`.

    Returns the (N, G) matrix of exposures, column `j` being the fit of
    `M[:, j]`.  Same solutions as `np.apply_along_axis(decomposeQP, 0, M, P)`,
    but `P.T @ P` and the constraint matrices are built once instead of once
    per column, and the linear terms of all the columns come from one matrix
    product.  That set-up is a large share of the cost of a single small QP,
    and the bootstrap search solves the same panel R times per evaluation.
    """
    G, C, b = _qp_constraints(P)
    # Row j is the objective's linear term for column j of M.
    D = np.ascontiguousarray(M.T @ P, dtype=float)
    exposures = np.empty((P.shape[1], M.shape[1]))
    for j in range(M.shape[1]):
        # solve_qp may factorise G in place, so it gets a fresh copy each time.
        exposures[:, j] = _solve_qp(G.copy(), C, b, D[j])
    return exposures
