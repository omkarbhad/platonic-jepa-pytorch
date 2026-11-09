"""Two modalities see the same world through different renderings. Each trains its own JEPA by gradient descent,
never seeing the other. Then: do they agree up to one rotation (Theorem 1), and how many pairs find it (Prop. 2)?"""

import numpy as np

from platonic_jepa import (misalignment, procrustes, random_marginal, random_pair_distribution, random_rendering,
                           spectral_embedding, spectral_loss, train_jepa)

rng = np.random.default_rng(0)
n, k = 12, 3
p = random_marginal(n, rng)
Pi = random_pair_distribution(p, rng)
Phi = spectral_embedding(Pi, p, k)
print(f"world: {n} states, embedding dimension {k}; Proposition 1 minimum loss = {spectral_loss(Pi, p, k):.5f}")

a = train_jepa(Pi, p, k, random_rendering(n, 16, seed=1), steps=3000, seed=0)
b = train_jepa(Pi, p, k, random_rendering(n, 24, seed=2), steps=3000, seed=1)
for name, r in (("modality A", a), ("modality B", b)):
    print(f"{name}: loss {r['pair_loss']:.5f}, distance to the spectral embedding after the best rotation "
          f"{misalignment(r['embedding'], Phi, p):.1e}")

print(f"A vs B, raw coordinates (max difference): {np.abs(a['embedding'] - b['embedding']).max():.3f}")
print(f"A vs B, after the best rotation:          {misalignment(a['embedding'], b['embedding'], p):.1e}")
for m in (k, k + 2, n):
    Q, _ = procrustes(a["embedding"][:m], b["embedding"][:m])                 # rotation estimated from m paired states
    resid = a["embedding"] - b["embedding"] @ Q.T
    print(f"rotation from {m:2d} pairs -> error on all {n} states: {np.sqrt((p * (resid ** 2).sum(1)).sum()):.1e}")
