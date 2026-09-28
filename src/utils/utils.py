import numpy as np


def resolve_rng(rng=None):
    """Random source for the bootstrap draws.

    `None` keeps the module-level `np.random`, so `np.random.seed` still
    controls the result.  Anything `np.random.default_rng` accepts (an int, a
    `SeedSequence`, a `Generator`, which comes back unchanged) gives an
    independent generator that never touches the global state.  Both expose
    the `multinomial` and `gamma` methods the bootstrap uses.
    """
    return np.random if rng is None else np.random.default_rng(rng)


def is_wholenumber(x, tol=1e-9):
    return abs(x - round(x)) < tol


def FrobeniusNorm(m, P, exposures):
    reconstruction = P @ exposures
    return np.linalg.norm(m / m.sum() - reconstruction)
