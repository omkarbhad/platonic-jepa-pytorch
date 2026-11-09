import numpy as np
import pytest

from platonic_jepa import (kernel, mix, misalignment, procrustes, random_marginal, random_pair_distribution,
                           spectral_embedding, theorem2_bound)


def random_rotation(k, rng):
    Q, R = np.linalg.qr(rng.normal(size=(k, k)))
    return Q * np.sign(np.diag(R))


def test_procrustes_recovers_the_rotation_from_k_pairs():
    """Proposition 2 (ii): k pairs in general position determine the rotation."""
    rng = np.random.default_rng(0)
    k = 4
    Q = random_rotation(k, rng)
    B = rng.normal(size=(k, k))                    # k embeddings of the second modality, in general position
    A = B @ Q.T                                    # the first modality sees them rotated by Q
    Q_hat, err = procrustes(A, B)
    assert np.allclose(Q_hat, Q, atol=1e-10) and err < 1e-12


def test_procrustes_with_noise_and_weights():
    rng = np.random.default_rng(1)
    Q = random_rotation(3, rng)
    B = rng.normal(size=(200, 3))
    A = B @ Q.T + 0.01 * rng.normal(size=(200, 3))
    Q_hat, err = procrustes(A, B, rng.uniform(0.5, 1.5, 200))
    assert np.allclose(Q_hat, Q, atol=0.02) and err < 0.03


def test_rotations_do_not_change_the_first_two_moments():
    """Proposition 2 (i): a rotated whitened embedding is still centred and whitened, so unpaired first and second
    moments cannot tell which rotation relates two modalities."""
    rng = np.random.default_rng(2)
    p = random_marginal(15, rng)
    Phi = spectral_embedding(random_pair_distribution(p, rng), p, 3)
    R = Phi @ random_rotation(3, rng).T
    assert np.allclose((p[:, None] * R).sum(0), 0, atol=1e-10)
    assert np.allclose(R.T @ (p[:, None] * R), np.eye(3), atol=1e-10)
    assert misalignment(Phi, R, p) < 1e-10                      # and they are the same representation up to a rotation


def test_theorem_1_same_positives_give_the_same_representation_up_to_a_rotation():
    rng = np.random.default_rng(3)
    p = random_marginal(20, rng)
    Pi = random_pair_distribution(p, rng)
    # any orthonormal basis of the top eigenspace is a valid spectral embedding: model that by a random rotation
    Phi_1 = spectral_embedding(Pi, p, 3)
    Phi_2 = spectral_embedding(Pi, p, 3) @ random_rotation(3, rng).T
    assert misalignment(Phi_1, Phi_2, p) < 1e-10


def worlds(count, seed):
    """Random worlds like the paper's Figure 2: 8 to 39 states, k from 1 to 5, half of them clustered
    (small gaps); the second kernel mixes in an unrelated kernel with weight 1e-4 to 1."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        n = int(rng.integers(8, 40))
        k = int(rng.integers(1, 6))
        p = random_marginal(n, rng)
        Pi_1 = random_pair_distribution(p, rng, clustered=bool(i % 2))
        other = random_pair_distribution(p, rng)
        w = 10 ** rng.uniform(-4, 0)
        yield p, Pi_1, mix(Pi_1, other, w), k


def test_theorem_2_error_is_below_the_bound_in_every_world():
    ratios = []
    for p, Pi_1, Pi_2, k in worlds(120, seed=0):
        err = misalignment(spectral_embedding(Pi_1, p, k), spectral_embedding(Pi_2, p, k), p)
        bound = theorem2_bound(Pi_1, Pi_2, p, k)
        assert err <= bound + 1e-9, (err, bound)
        ratios.append(err / bound)
    assert max(ratios) <= 1.0
    assert np.mean(ratios) < 0.5                                  # the bound is not vacuous: errors are within one order of it


def test_misalignment_and_bound_grow_with_the_kernel_difference():
    rng = np.random.default_rng(4)
    p = random_marginal(25, rng)
    Pi_1, other = random_pair_distribution(p, rng), random_pair_distribution(p, rng)
    errs, bounds = [], []
    for w in (1e-4, 1e-3, 1e-2, 1e-1):
        Pi_2 = mix(Pi_1, other, w)
        errs.append(misalignment(spectral_embedding(Pi_1, p, 3), spectral_embedding(Pi_2, p, 3), p))
        bounds.append(theorem2_bound(Pi_1, Pi_2, p, 3))
    assert errs == sorted(errs) and bounds == sorted(bounds)
    assert errs[0] < 0.01 * errs[-1]


def test_identical_kernels_give_zero_error_and_zero_bound():
    rng = np.random.default_rng(5)
    p = random_marginal(10, rng)
    Pi = random_pair_distribution(p, rng)
    assert theorem2_bound(Pi, Pi, p, 2) == 0.0


def test_bound_uses_the_eigengap_of_the_second_kernel():
    """Theorem 2: delta is the gap of K_pi2, so the bound is not symmetric in its two arguments."""
    rng = np.random.default_rng(7)
    p = random_marginal(20, rng)
    Pi_1 = random_pair_distribution(p, rng, clustered=True)      # small gaps
    Pi_2 = random_pair_distribution(p, rng)                      # large gaps
    k = 1
    K = lambda P: P / np.sqrt(np.outer(p, p))
    mu2 = np.sort(np.linalg.eigvalsh(K(Pi_2)))[::-1]
    delta_2 = mu2[k] - mu2[k + 1]                                # mu_0 = 1 is the trivial eigenvalue
    D = K(Pi_1) - K(Pi_2)
    want = 2 ** 1.5 / delta_2 * min(np.sqrt(k) * np.linalg.norm(D, 2), np.linalg.norm(D, "fro"))
    assert theorem2_bound(Pi_1, Pi_2, p, k) == pytest.approx(want, rel=1e-9)
    assert theorem2_bound(Pi_1, Pi_2, p, k) != pytest.approx(theorem2_bound(Pi_2, Pi_1, p, k), rel=1e-3)
