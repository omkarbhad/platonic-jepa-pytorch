# platonic-jepa-pytorch

Numerical checks of the theory in
[Proving and Improving the Platonic Representation Hypothesis with JEPAs](https://arxiv.org/abs/2610.39302)
(Balestriero and LeCun, 2026): spectral embeddings, the rotation bound, and the cost of nuisance, on finite worlds.

This is an unofficial implementation written from the paper's theory sections (2.1 to 2.3 and the Appendix A
proofs). It reproduces the theorems on small synthetic worlds. It does **not** reproduce the paper's large-scale
experiments (145k captions, 940M web images, a 109M-parameter text JEPA, the captioner).

## The idea

The Platonic Representation Hypothesis says that models trained on different modalities (images, text) converge
to the same representation of the world as they scale. The paper turns this into a theorem for JEPAs.

A **world** is a finite set of states `z` with probabilities `p(z)`. Each modality sees it through an injective
rendering `x = g_m(z)`. A JEPA of modality `m` is trained on **positive pairs** `(g_m(z), g_m(z'))`, where
`(z, z')` is drawn from a symmetric distribution `Pi_m` with both marginals equal to `p`. Its loss is a
prediction term plus an anti-collapse term; here the predictor is the identity and the encoder is whitened.

- **Proposition 1.** The best encoder is the **spectral embedding** `Phi(z) = (v_1(z), ..., v_k(z)) / sqrt(p(z))`, the
  top non-trivial eigenvectors of `K = D^{-1/2} Pi D^{-1/2}`, unique up to a rotation. Its loss is `2 * sum_i (1 - mu_i)`.
- **Theorem 1.** Modalities with the same positives learn the same embedding up to **one rotation**.
- **Proposition 2.** `k` paired states are enough to find that rotation (orthogonal Procrustes).
- **Theorem 2.** If the positives differ, the misalignment is bounded:
  `min_Q (E_p ||Phi_1 - Q Phi_2||^2)^(1/2) <= 2^(3/2) / delta * min(sqrt(k) ||K_1 - K_2||_2, ||K_1 - K_2||_F)`,
  where `delta` is the eigengap of the second kernel.
- **Theorem 3.** If an observation carries a nuisance (which words a caption uses, which viewpoint a photo has),
  every direction an encoder spends on it is lost for alignment:
  `min_Q E ||Q f_1(x_1) - f_2(x_2)||^2 >= 2k (1 - min_m rho(f_m))`, with `rho` the share of the embedding the scene explains.

## Install

```bash
git clone https://github.com/omkarbhad/platonic-jepa-pytorch
cd platonic-jepa-pytorch
pip install -e ".[dev]"
pytest -q          # 29 tests, about 8 seconds
```

## Usage

```python
import numpy as np
from platonic_jepa import (random_marginal, random_pair_distribution, spectral_embedding, spectral_loss,
                           misalignment, theorem2_bound, mix, train_jepa, random_rendering)

rng = np.random.default_rng(0)
p = random_marginal(12, rng)                       # a world of 12 states
Pi = random_pair_distribution(p, rng)              # its positive pairs

Phi = spectral_embedding(Pi, p, k=3)               # Proposition 1
spectral_loss(Pi, p, k=3)                          # 2 * sum(1 - mu_i)

# train an encoder by gradient descent on a rendering of the world; it finds Phi up to a rotation
result = train_jepa(Pi, p, 3, random_rendering(12, 16, seed=1))
misalignment(result["embedding"], Phi, p)          # ~1e-8

# a second modality whose positives differ a little (Theorem 2)
Pi_2 = mix(Pi, random_pair_distribution(p, rng), 0.01)
misalignment(Phi, spectral_embedding(Pi_2, p, 3), p) <= theorem2_bound(Pi, Pi_2, p, 3)   # True
```

## Results

**Training by gradient descent reaches the spectral optimum (Proposition 1, Theorem 1, Proposition 2).**
`python examples/train_two_modalities.py`: two modalities, different random renderings, 12 states, `k = 3`, trained
separately, never seeing each other:

```
Proposition 1 minimum loss = 5.07277
modality A: loss 5.07277, distance to the spectral embedding after the best rotation 1.5e-03
modality B: loss 5.07277, distance to the spectral embedding after the best rotation 1.2e-08
A vs B, raw coordinates (max difference): 4.564
A vs B, after the best rotation:          1.5e-03
rotation from  3 pairs -> error on all 12 states: 1.6e-03
```

Both encoders reach the predicted minimum loss. Their raw coordinates differ a lot (up to 4.6), and after one
rotation they agree to 1.5e-3. Three paired states are enough to find that rotation.

**Theorem 2 on 300 random worlds** (`python examples/theorem2_bound.py`; 8 to 39 states, `k` from 1 to 5, half of
the worlds clustered so that the spectral gaps are small, the second kernel mixes in an unrelated kernel with
weight 1e-4 to 1, as in the paper's Figure 2):

```
worlds: 300   error below the bound in: 300
error / bound   max 0.108   mean 0.021   median 0.018
```

The misalignment is below the bound in all 300 worlds, at most 10.8% of it. (The paper reports at most 18% for its
own random worlds; the two sets of worlds are generated differently, so the numbers are not expected to match.)

**Good views versus bad views** (`python examples/views_demo.py`; an illustration built from the description in
section 2.3, not a reproduction of the paper's experiments). Positives are good views (two observations of related
scenes, independent nuisances) mixed with a weight `w` of bad views (two corruptions of one observation, which share
the nuisance):

```
scene eigenvalues mu_1..mu_3 of the world's own kernel: [0.155 0.05  0.026]
 bad-view weight w   scene fraction rho   cross-modal cost   (cost 0 = perfect alignment, 2k = 6)
              0.00                1.000              0.000
              0.02                1.000              0.000
              0.10                0.333              4.000
              0.20                0.000              6.000
              1.00                0.000              6.000
```

Bad views add nuisance directions whose eigenvalue equals `w`. While `w` is below the scene's eigenvalues the
embedding stays pure scene and alignment is perfect. Once `w` passes them, the nuisance takes the top directions
and the cross-modal cost reaches its maximum `2k`. This fits the paper's claim that next-token prediction, whose
positives are parts of one document, aligns poorly (though I did not check the paper's Proposition 4 here).

## What is implemented

| Paper | Here |
| --- | --- |
| Section 2.1, world, positive pairs, `K_pi` | `world.py`, `spectral.kernel` |
| Proposition 1, spectral embedding and its minimum | `spectral_embedding`, `spectral_loss` |
| Equation 1 with a whitened encoder, trained by gradient descent | `train_jepa` |
| Proposition 2, rotation from pairs (Procrustes) | `procrustes`, `misalignment` |
| Theorem 2, bound on misalignment | `theorem2_bound` |
| Theorem 3, nuisance costs alignment | `scene_explained_fraction`, `cross_modal_cost`, `theorem3_bound` |

Not implemented: Theorem 4, Propositions 3 and 4 (next-token prediction as a bad-view JEPA), the full-distribution
SIGReg constraint, and all scaling and application experiments.

## Tests

29 tests. The main ones:

- `Pi` is symmetric with marginal `p`; `K` has eigenvalue 1 with eigenvector `sqrt(p)`; `Phi` is centred and whitened.
- Proposition 1: `Phi` reaches `2 * sum(1 - mu_i)`, no random whitened map does better (200 tries), every rotation of `Phi`
  reaches it too, and a non-top eigenvector does not.
- Proposition 2: Procrustes recovers a known rotation exactly from `k` pairs, and noisy weighted pairs approximately.
- Theorem 2: error below the bound in 120 random worlds; both grow with the kernel difference; the bound uses the gap
  of the second kernel, checked against a hand computation.
- Theorem 3: the bound holds for 300 random whitened encoders; the cost formula equals an explicit sum over nuisances.
- Gradient-trained JEPAs reach the spectral loss and agree across modalities up to one rotation, and the rotation is
  found from `k` pairs.

I broke the code on purpose in eight ways (wrong kernel normalisation, trivial eigenvector in the embedding, missing
`1/sqrt(p)`, wrong Procrustes orientation, wrong eigengap, missing factor 2, `tr(C)` instead of `tr(C^{1/2})`, wrong
marginal for bad views). The tests caught all eight; one needed an extra test, because in random worlds the two gaps
are almost equal.

One correction came from the tests: with disconnected positives the top eigenvalue 1 repeats, and a plain
eigen-decomposition may mix the trivial eigenvector into the embedding. `spectrum` now removes the trivial eigenpair
first.

## Citation

```bibtex
@article{balestriero2026platonic,
    title   = {Proving and Improving the Platonic Representation Hypothesis with JEPAs},
    author  = {Randall Balestriero and Yann LeCun},
    journal = {arXiv preprint arXiv:2610.39302},
    year    = {2026}
}
```

## License

MIT
