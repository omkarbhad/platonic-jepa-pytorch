"""Finite worlds and positive-pair distributions (paper section 2.1).

A world is a finite set of states z with probabilities p(z). The positives of a JEPA are pairs of states drawn
from a symmetric joint distribution Pi (both marginals equal to p): Pi[z, z'] = P(z, z'). Everything the
theorems say is a statement about the matrix Pi.
"""

import numpy as np


def random_marginal(n: int, rng: np.random.Generator) -> np.ndarray:
    p = rng.dirichlet(np.ones(n) * 2.0)
    return np.clip(p, 1e-3, None) / np.clip(p, 1e-3, None).sum()          # no state is almost impossible


def symmetric_sinkhorn(W: np.ndarray, p: np.ndarray, iters: int = 20000, tol: float = 1e-13) -> np.ndarray:
    """Scale a symmetric non-negative matrix W to Pi = diag(c) W diag(c) whose row sums are exactly p."""
    c = np.sqrt(p / W.sum(1))
    for _ in range(iters):
        new = np.sqrt(c * p / (W @ c))
        if np.abs(new - c).max() < tol:
            c = new
            break
        c = new
    Pi = c[:, None] * W * c[None, :]
    if not np.allclose(Pi.sum(1), p, atol=1e-8):
        raise RuntimeError("Sinkhorn scaling did not converge")
    return Pi


def random_pair_distribution(p: np.ndarray, rng: np.random.Generator, clustered: bool = False,
                             n_clusters: int = 3, off_block: float = 3e-3) -> np.ndarray:
    """A random symmetric positive-pair distribution with marginal p.

    clustered=True builds a graph with a few tightly linked groups of states, which gives small spectral
    gaps (the hard case for alignment)."""
    n = len(p)
    if clustered:
        labels = rng.integers(0, n_clusters, size=n)
        same = (labels[:, None] == labels[None, :]).astype(float)
        W = (same + off_block) * rng.uniform(0.5, 1.5, size=(n, n))
    else:
        W = rng.uniform(0.0, 1.0, size=(n, n)) + 1e-3
    return symmetric_sinkhorn((W + W.T) / 2, p)


def mix(Pi_a: np.ndarray, Pi_b: np.ndarray, weight: float) -> np.ndarray:
    """(1 - weight) * Pi_a + weight * Pi_b. Still symmetric with the same marginal if both are."""
    return (1.0 - weight) * Pi_a + weight * Pi_b
