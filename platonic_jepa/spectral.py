"""Spectral embeddings (paper Proposition 1) and the rotation bound (Theorem 2)."""

from typing import Tuple

import numpy as np


def kernel(Pi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """K_pi = D^{-1/2} Pi D^{-1/2}, D = diag(p): the normalised adjacency of the graph the positives draw."""
    s = 1.0 / np.sqrt(p)
    return Pi * s[:, None] * s[None, :]


def spectrum(Pi: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of K_pi. The first pair is the known trivial one,
    mu_0 = 1 with v_0 = sqrt(p) (paper: v_0 = D^{1/2} 1).

    The trivial pair is removed explicitly before the rest is computed. A graph whose positives split the world into
    several disconnected parts has eigenvalue 1 with multiplicity > 1, and a plain eigen-decomposition could then
    return any vector of that eigenspace, mixing the trivial one with the others."""
    u = np.sqrt(p)
    deflated = kernel(Pi, p) - np.outer(u, u)               # K u = u, so this sets the trivial eigenvalue to 0
    mu, V = np.linalg.eigh(deflated)
    order = np.argsort(-mu)
    mu, V = mu[order], V[:, order]
    # the deflated matrix has eigenvalue 0 on u; move that eigenpair out and put the true (1, u) first
    keep = [i for i in range(len(mu)) if abs(V[:, i] @ u) < 0.5]
    mu, V = mu[keep], V[:, keep]
    return np.concatenate([[1.0], mu]), np.column_stack([u, V])


def spectral_embedding(Pi: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    """Phi_pi(z) = (v_1(z), ..., v_k(z)) / sqrt(p(z)): the top k NON-trivial eigenvectors. Shape (n, k).

    It has mean 0 and identity covariance under p, and it minimises E_pi ||h(z) - h(z')||^2 among such maps."""
    _, V = spectrum(Pi, p)
    return V[:, 1 : k + 1] / np.sqrt(p)[:, None]


def spectral_loss(Pi: np.ndarray, p: np.ndarray, k: int) -> float:
    """Proposition 1: the minimum value 2 * sum_{i<=k} (1 - mu_i)."""
    mu, _ = spectrum(Pi, p)
    return float(2.0 * np.sum(1.0 - mu[1 : k + 1]))


def eigengap(Pi: np.ndarray, p: np.ndarray, k: int) -> float:
    """delta = mu_k - mu_{k+1} (the trivial eigenvalue mu_0 = 1 is not counted)."""
    mu, _ = spectrum(Pi, p)
    return float(mu[k] - mu[k + 1])


def theorem2_bound(Pi_1: np.ndarray, Pi_2: np.ndarray, p: np.ndarray, k: int) -> float:
    """Right-hand side of Theorem 2:  2^{3/2} / delta * min( sqrt(k) ||K1 - K2||_2 , ||K1 - K2||_F ),
    where delta is the eigengap of the SECOND kernel."""
    diff = kernel(Pi_1, p) - kernel(Pi_2, p)
    spectral_norm = np.linalg.norm(diff, 2)
    frob = np.linalg.norm(diff, "fro")
    return float(2 ** 1.5 / eigengap(Pi_2, p, k) * min(np.sqrt(k) * spectral_norm, frob))
