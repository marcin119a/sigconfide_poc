import numpy as np


def is_wholenumber(x, tol=1e-9):
    return abs(x - round(x)) < tol


def FrobeniusNorm(m, P, exposures):
    reconstruction = P @ exposures
    return np.linalg.norm(m / m.sum() - reconstruction)
