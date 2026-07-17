"""Shared fixtures for the sigconfide unit tests.

The core idea for deterministic testing: build a small synthetic signature
matrix ``P`` and construct tumour profiles ``m`` as *known* linear combinations
of its columns.  A correct decomposition must then recover those known weights,
which lets us assert on exact numbers instead of just shape/normalisation.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """A seeded NumPy Generator for reproducible randomised tests."""
    return np.random.default_rng(20260717)


@pytest.fixture
def simple_P():
    """A tiny, well-conditioned 4x3 signature matrix (columns already sum to 1).

    Columns are distinct probability distributions over 4 mutation contexts so
    that any convex combination has a unique non-negative decomposition.
    """
    P = np.array(
        [
            [0.7, 0.1, 0.2],
            [0.1, 0.6, 0.2],
            [0.1, 0.2, 0.3],
            [0.1, 0.1, 0.3],
        ],
        dtype=float,
    )
    # Normalise columns to sum to 1 (defensive; already close).
    return P / P.sum(axis=0)


@pytest.fixture
def known_weights():
    """A convex weight vector matching ``simple_P`` (3 signatures)."""
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def m_from_P(simple_P, known_weights):
    """A profile that is exactly ``simple_P @ known_weights`` (sums to 1)."""
    return simple_P @ known_weights


@pytest.fixture
def counts_profile(simple_P, known_weights):
    """Same mixture as ``m_from_P`` but expressed as integer mutation counts."""
    total = 5000
    probs = simple_P @ known_weights
    counts = np.round(probs * total)
    return counts
