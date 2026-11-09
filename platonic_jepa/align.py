"""Alignment up to one rotation (paper Proposition 2: orthogonal Procrustes)."""

from typing import Tuple

import numpy as np


def procrustes(A: np.ndarray, B: np.ndarray, weights: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """The orthogonal Q minimising  sum_i w_i ||A_i - Q B_i||^2  for rows A_i, B_i in R^k (rows are samples).

    Returns (Q, rms) with rms = sqrt of the weighted mean squared residual. weights default to uniform."""
    n = len(A)
    w = np.full(n, 1.0 / n) if weights is None else weights / weights.sum()
    M = (A * w[:, None]).T @ B                       # sum_i w_i A_i B_i^T, shape (k, k)
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    resid = A - B @ Q.T
    return Q, float(np.sqrt((w * (resid ** 2).sum(1)).sum()))


def misalignment(Phi_1: np.ndarray, Phi_2: np.ndarray, p: np.ndarray) -> float:
    """min over rotations Q of (E_p ||Phi_1(z) - Q Phi_2(z)||^2)^(1/2): the left side of Theorem 2."""
    return procrustes(Phi_1, Phi_2, p)[1]
