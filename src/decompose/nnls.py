from scipy.optimize import nnls
import numpy as np


def decomposeNNLS(m, P):
    # Solve: minimize ||P @ x - m||^2  subject to  x >= 0
    exposures, _ = nnls(P, m)

    # Renormalize to sum to 1 (same post-processing as decomposeQP)
    total = exposures.sum()
    if total > 0:
        exposures /= total

    return exposures
