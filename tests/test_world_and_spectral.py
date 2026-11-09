import numpy as np
import pytest

from platonic_jepa import (eigengap, kernel, mix, random_marginal, random_pair_distribution, spectral_embedding,
                           spectral_loss, spectrum)


def world(n=12, seed=0, clustered=False):
    rng = np.random.default_rng(seed)
    p = random_marginal(n, rng)
    return p, random_pair_distribution(p, rng, clustered=clustered), rng


@pytest.mark.parametrize("clustered", [False, True])
def test_pair_distribution_is_symmetric_with_the_right_marginal(clustered):
    for seed in range(5):
        p, Pi, _ = world(seed=seed, clustered=clustered)
        assert np.allclose(Pi, Pi.T) and (Pi >= 0).all()
        assert np.allclose(Pi.sum(1), p, atol=1e-9) and np.isclose(Pi.sum(), 1.0)


def test_mixing_keeps_the_marginal():
    p, Pi1, rng = world()
    Pi2 = random_pair_distribution(p, rng)
    assert np.allclose(mix(Pi1, Pi2, 0.3).sum(1), p)


def test_clustered_worlds_have_small_gaps_inside_their_cluster_eigenvalues():
    """Tightly linked groups put several eigenvalues of the kernel close to 1, so the gap between them is small."""
    def mean_gap(clustered, k):
        gaps = []
        for seed in range(20):
            p, Pi, _ = world(n=20, seed=seed, clustered=clustered)
            gaps.append(eigengap(Pi, p, k))
        return np.mean(gaps)
    assert mean_gap(True, 1) < 0.2 * mean_gap(False, 1)


def test_kernel_has_eigenvalue_one_with_eigenvector_sqrt_p():
    p, Pi, _ = world()
    mu, V = spectrum(Pi, p)
    assert mu[0] == pytest.approx(1.0) and (mu[1:] < 1.0).all()
    assert np.allclose(np.abs(V[:, 0]), np.sqrt(p), atol=1e-10)
    assert np.allclose(kernel(Pi, p) @ np.sqrt(p), np.sqrt(p))


def test_spectral_embedding_is_centred_and_whitened():
    p, Pi, _ = world()
    Phi = spectral_embedding(Pi, p, 4)
    assert np.allclose((p[:, None] * Phi).sum(0), 0, atol=1e-10)
    assert np.allclose(Phi.T @ (p[:, None] * Phi), np.eye(4), atol=1e-10)


def pair_loss(h, Pi):
    """E_pi ||h(z) - h(z')||^2 by direct summation over all pairs."""
    d = ((h[:, None, :] - h[None, :, :]) ** 2).sum(-1)
    return float((Pi * d).sum())


def random_whitened(p, k, rng):
    h = rng.normal(size=(len(p), k))
    h -= (p[:, None] * h).sum(0)
    C = h.T @ (p[:, None] * h)
    vals, vecs = np.linalg.eigh(C)
    return h @ vecs @ np.diag(vals ** -0.5) @ vecs.T


def test_proposition_1_minimum_value():
    """The spectral embedding attains 2 * sum (1 - mu_i), and no other centred, whitened map does better."""
    p, Pi, rng = world()
    for k in (1, 2, 4):
        best = spectral_loss(Pi, p, k)
        assert pair_loss(spectral_embedding(Pi, p, k), Pi) == pytest.approx(best, rel=1e-9)
        for _ in range(200):
            assert pair_loss(random_whitened(p, k, rng), Pi) >= best - 1e-9


def test_proposition_1_minimisers_are_the_rotations_of_phi():
    p, Pi, rng = world()
    k = 3
    Phi = spectral_embedding(Pi, p, k)
    for _ in range(5):
        Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
        assert pair_loss(Phi @ Q.T, Pi) == pytest.approx(spectral_loss(Pi, p, k), rel=1e-9)
    # a non-top eigenvector gives a strictly larger loss
    _, V = spectrum(Pi, p)
    worse = (V[:, [1, 2, 4]] / np.sqrt(p)[:, None])
    assert pair_loss(worse, Pi) > spectral_loss(Pi, p, k) + 1e-6
