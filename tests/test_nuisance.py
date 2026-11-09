import numpy as np
import pytest

from platonic_jepa import (center_and_whiten, cross_modal_cost, extend, random_marginal, random_pair_distribution,
                           scene_explained_fraction, spectral_embedding, theorem3_bound)
from platonic_jepa.spectral import spectrum


def setup(seed=0, n=8, m=5):
    rng = np.random.default_rng(seed)
    p = random_marginal(n, rng)
    q = rng.dirichlet(np.ones(m))
    return rng, p, q, random_pair_distribution(p, rng)


def random_encoder(rng, n, m, k, p, q):
    return center_and_whiten(rng.normal(size=(n, m, k)), p, q)


def test_center_and_whiten():
    rng, p, q, _ = setup()
    f = random_encoder(rng, 8, 5, 3, p, q)
    w = (p[:, None] * q[None, :]).reshape(-1)
    flat = f.reshape(-1, 3)
    assert np.allclose((w[:, None] * flat).sum(0), 0, atol=1e-10)
    assert np.allclose(flat.T @ (w[:, None] * flat), np.eye(3), atol=1e-10)


def test_scene_explained_fraction_extremes():
    rng, p, q, _ = setup()
    k = 3
    only_scene = center_and_whiten(np.repeat(rng.normal(size=(8, 1, k)), 5, axis=1), p, q)      # f depends on z only
    only_nuisance = center_and_whiten(np.repeat(rng.normal(size=(1, 5, k)), 8, axis=0), p, q)   # f depends on nu only
    assert scene_explained_fraction(only_scene, p, q) == pytest.approx(1.0, abs=1e-9)
    assert scene_explained_fraction(only_nuisance, p, q) == pytest.approx(0.0, abs=1e-9)
    mixed = random_encoder(rng, 8, 5, k, p, q)
    assert 0.0 < scene_explained_fraction(mixed, p, q) < 1.0


def test_theorem_3_lower_bound_holds_for_random_encoders():
    """min_Q E||Q f1(x1) - f2(x2)||^2 >= 2k (1 - min_m rho(f_m)) for ANY centred, whitened encoders."""
    rng, p, q, _ = setup(seed=1)
    q2 = rng.dirichlet(np.ones(4))
    slack = []
    for trial in range(300):
        k = int(rng.integers(1, 5))
        f1 = random_encoder(rng, 8, 5, k, p, q)
        f2 = random_encoder(rng, 8, 4, k, p, q2)
        cost = cross_modal_cost(f1, f2, p, q, q2)
        bound = theorem3_bound(f1, f2, p, q, q2)
        assert cost >= bound - 1e-9, (cost, bound)
        slack.append(cost - bound)
    assert min(slack) >= -1e-9


def test_cross_modal_cost_matches_a_direct_monte_carlo_over_nuisances():
    """The closed form 2k - 2||M||_* equals the explicit expectation over (z, nu1, nu2) with the best rotation."""
    rng, p, q, _ = setup(seed=2, n=5, m=3)
    q2 = rng.dirichlet(np.ones(3))
    k = 2
    f1, f2 = random_encoder(rng, 5, 3, k, p, q), random_encoder(rng, 5, 3, k, p, q2)
    # explicit second-moment matrix E[f2(x2) f1(x1)^T] over z, nu1, nu2
    M = np.zeros((k, k))
    for z in range(5):
        for a in range(3):
            for b in range(3):
                M += p[z] * q[a] * q2[b] * np.outer(f1[z, a], f2[z, b])
    cost = 2 * k - 2 * np.linalg.svd(M, compute_uv=False).sum()
    assert cross_modal_cost(f1, f2, p, q, q2) == pytest.approx(cost, abs=1e-10)


def test_encoders_of_the_scene_alone_can_align_perfectly():
    rng, p, q, _ = setup(seed=3)
    k = 3
    base = rng.normal(size=(8, 1, k))
    f1 = center_and_whiten(np.repeat(base, 5, axis=1), p, q)
    R, _ = np.linalg.qr(rng.normal(size=(k, k)))
    f2 = f1 @ R.T                                      # the same scene encoding, rotated
    assert cross_modal_cost(f1, f2, p, q, q) == pytest.approx(0.0, abs=1e-8)
    assert theorem3_bound(f1, f2, p, q, q) == pytest.approx(0.0, abs=1e-8)


def spectral_encoder(Pi_ext, p_ext, k, n, m):
    return spectral_embedding(Pi_ext, p_ext, k).reshape(n, m, k)


def test_extended_world_marginals_and_views():
    rng, p, q, Pi = setup(seed=4, n=6, m=4)
    for w in (0.0, 0.5, 1.0):
        Pi_ext, p_ext = extend(Pi, p, q, bad_view_weight=w)
        assert np.allclose(Pi_ext, Pi_ext.T)
        assert np.allclose(Pi_ext.sum(1), p_ext, atol=1e-10)


def test_good_views_encode_only_the_scene_and_bad_views_only_the_nuisance():
    """Positives that are two observations of related scenes (independent nuisances) give embeddings that depend
    on the scene alone. Positives that are two corruptions of one observation share the nuisance, so the nuisance
    is the strongest connection in the graph and takes the top directions."""
    rng, p, q, Pi = setup(seed=5, n=8, m=4)
    k = 3
    good, p_ext = extend(Pi, p, q, bad_view_weight=0.0)
    bad, _ = extend(Pi, p, q, bad_view_weight=1.0)
    f_good = spectral_encoder(good, p_ext, k, 8, 4)
    assert scene_explained_fraction(f_good, p, q) == pytest.approx(1.0, abs=1e-8)
    # for bad views, the top eigenvalues (1 with multiplicity m) are nuisance indicators: k = m - 1 directions
    f_bad = spectral_encoder(bad, p_ext, 3, 8, 4)
    assert scene_explained_fraction(f_bad, p, q) == pytest.approx(0.0, abs=1e-6)


def test_bad_views_take_over_once_the_nuisance_outranks_the_scene():
    """Mixing bad-view positives (weight w) into good ones leaves the scene eigenvalues unchanged and adds nuisance
    eigenvalues equal to w. While w is below the scene eigenvalues the embedding stays pure scene (rho = 1, perfect
    alignment); once w passes them the nuisance fills the top k directions (rho = 0, cost 2k)."""
    rng, p, q, Pi = setup(seed=6, n=8, m=4)
    k = 3
    scene_mu = spectrum(Pi, p)[0][1 : k + 1]                  # 0.155, 0.05, 0.026 in this world
    rho, cost = [], []
    ws = (0.0, 0.5 * scene_mu.min(), 2 * scene_mu.max(), 0.5, 1.0)
    for w in ws:
        Pi_ext, p_ext = extend(Pi, p, q, bad_view_weight=w)
        f = spectral_encoder(Pi_ext, p_ext, k, 8, 4)
        rho.append(scene_explained_fraction(f, p, q))
        cost.append(cross_modal_cost(f, f, p, q, q))
    assert all(r2 <= r1 + 1e-9 for r1, r2 in zip(rho, rho[1:]))              # never more scene with more bad views
    assert rho[:2] == pytest.approx([1.0, 1.0], abs=1e-8) and cost[:2] == pytest.approx([0.0, 0.0], abs=1e-8)
    assert rho[2:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6) and cost[2:] == pytest.approx([2 * k] * 3, abs=1e-6)
