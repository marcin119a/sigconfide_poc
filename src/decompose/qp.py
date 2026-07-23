import numpy as np
import quadprog


def decomposeQP(m, P):
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
    # d: vector appearing in the quadratic programming objective function
    d = np.dot(m.T, P).astype(float)

    # Solve quadratic programming problem
    out = quadprog.solve_qp(G, d, C, b, meq=1)

    # Some exposure values are negative, but very close to 0
    # Change these negative values to zero and renormalize
    exposures = out[0]
    exposures[exposures < 0] = 0
    exposures /= sum(exposures)

    # return the exposures
    return exposures
