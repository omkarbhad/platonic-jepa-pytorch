import numpy as np
import pytest

from platonic_jepa import (mix, misalignment, procrustes, random_marginal, random_pair_distribution, random_rendering,
                           spectral_embedding, spectral_loss, theorem2_bound, train_jepa)

N, K = 10, 2


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    p = random_marginal(N, rng)
    Pi = random_pair_distribution(p, rng)
    return rng, p, Pi


@pytest.fixture(scope="module")
def trained_pair(world):
    """Two modalities with different injective renderings, trained separately on the SAME positives."""
    _, p, Pi = world
    a = train_jepa(Pi, p, K, random_rendering(N, 16, seed=1), steps=2000, seed=0)
    b = train_jepa(Pi, p, K, random_rendering(N, 24, seed=2), steps=2000, seed=1)
    return a, b


def test_renderings_are_injective():
    X = random_rendering(N, 16, seed=1)
    dists = np.linalg.norm(X[:, None] - X[None], axis=-1) + np.eye(N) * 9
    assert dists.min() > 1e-3


def test_gradient_training_reaches_the_spectral_optimum(world, trained_pair):
    """Proposition 1, by gradient descent: the loss reaches 2 * sum(1 - mu_i) and the embedding is a rotation of Phi."""
    _, p, Pi = world
    Phi = spectral_embedding(Pi, p, K)
    for result in trained_pair:
        assert result["pair_loss"] == pytest.approx(spectral_loss(Pi, p, K), rel=1e-4)
        assert misalignment(result["embedding"], Phi, p) < 0.01


def test_theorem_1_separately_trained_modalities_agree_up_to_one_rotation(world, trained_pair):
    _, p, _ = world
    a, b = trained_pair
    assert misalignment(a["embedding"], b["embedding"], p) < 0.01
    # without the rotation they do NOT agree: different runs land at different rotations
    assert np.abs(a["embedding"] - b["embedding"]).max() > 0.1


def test_proposition_2_k_pairs_identify_the_rotation(world, trained_pair):
    """Find the rotation from only k paired states, then check it on all states."""
    _, p, _ = world
    a, b = trained_pair
    pairs = [0, 1]                                              # k = 2 states, in general position
    Q, _ = procrustes(a["embedding"][pairs], b["embedding"][pairs])
    resid = a["embedding"] - b["embedding"] @ Q.T
    assert np.sqrt((p * (resid ** 2).sum(1)).sum()) < 0.02


def test_different_positives_misalign_within_theorem_2(world):
    """Modality 2 draws its positives from a mix with an unrelated kernel: trained encoders differ, and the
    misalignment stays below the bound of Theorem 2."""
    rng, p, Pi = world
    Pi_2 = mix(Pi, random_pair_distribution(p, rng), 0.02)
    a = train_jepa(Pi, p, K, random_rendering(N, 16, seed=1), steps=2000)
    b = train_jepa(Pi_2, p, K, random_rendering(N, 24, seed=2), steps=2000)
    err = misalignment(a["embedding"], b["embedding"], p)
    bound = theorem2_bound(Pi, Pi_2, p, K)
    assert 0.01 < err <= bound
