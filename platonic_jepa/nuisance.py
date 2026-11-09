"""Observations carry a nuisance, and the nuisance costs alignment (paper section 2.3, Theorem 3).

An observation is x = g(z, nu): the world state z plus a nuisance nu of its own modality (which words a caption
uses, from which angle a photo was taken), independent of z. Everything here is exact finite sums.
Encoder outputs are arrays of shape (n_states, n_nuisances, k).
"""

from typing import Tuple

import numpy as np


def center_and_whiten(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Make an encoder f(z, nu) have mean 0 and identity covariance under p(z) q(nu)."""
    w = p[:, None] * q[None, :]
    flat = f.reshape(-1, f.shape[-1])
    ww = w.reshape(-1)
    flat = flat - (ww[:, None] * flat).sum(0)
    cov = flat.T @ (ww[:, None] * flat)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T
    return (flat @ W).reshape(f.shape)


def conditional_mean(f: np.ndarray, q: np.ndarray) -> np.ndarray:
    """m(z) = E[f(z, nu) | z]: the part of the embedding that the world state explains. Shape (n, k)."""
    return (f * q[None, :, None]).sum(1)


def scene_explained_fraction(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """rho(f) = tr(C_f^{1/2}) / k with C_f = Cov_z E[f | z], for a centered, whitened f. Lies in [0, 1];
    equals 1 exactly when f depends on the world state only."""
    m = conditional_mean(f, q)
    C = m.T @ (p[:, None] * m)
    vals = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    return float(np.sqrt(vals).sum() / f.shape[-1])


def cross_modal_cost(f1: np.ndarray, f2: np.ndarray, p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """min over rotations Q of E ||Q f1(x1) - f2(x2)||^2 for two observations of ONE world state with
    independent nuisances (the left side of Theorem 3). With both encoders whitened it equals 2k - 2 ||M||_*,
    M = sum_z p(z) m1(z) m2(z)^T, because the nuisances are independent given z."""
    m1, m2 = conditional_mean(f1, q1), conditional_mean(f2, q2)
    M = m1.T @ (p[:, None] * m2)
    k = f1.shape[-1]
    return max(0.0, float(2 * k - 2 * np.linalg.svd(M, compute_uv=False).sum()))     # clamp rounding noise


def theorem3_bound(f1: np.ndarray, f2: np.ndarray, p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """2k (1 - min_m rho(f_m)): the lower bound of Theorem 3."""
    k = f1.shape[-1]
    rho = min(scene_explained_fraction(f1, p, q1), scene_explained_fraction(f2, p, q2))
    return 2 * k * (1 - rho)


def extend(Pi: np.ndarray, p: np.ndarray, q: np.ndarray, bad_view_weight: float = 0.0,
           corruption: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Positive pairs on the extended world of observations (z, nu), flattened as index z * n_nu + nu.

    good views (weight 0):  two observations of states z, z' related by Pi, with independent nuisances.
    bad views (weight 1):   two corruptions of ONE observation: they share the nuisance, and the state may move
                            only a little, according to `corruption` (a pair distribution with marginal p).
    In between: a mixture. Returns (Pi_extended, p_extended)."""
    n, m = len(p), len(q)
    good = np.kron(Pi, np.outer(q, q))
    C = Pi if corruption is None else corruption
    bad = np.zeros((n * m, n * m))
    for v in range(m):
        idx = np.arange(n) * m + v
        bad[np.ix_(idx, idx)] = q[v] * C
    return (1 - bad_view_weight) * good + bad_view_weight * bad, np.kron(p, q)
