"""Check Theorem 2 on random finite worlds, like the paper's Figure 2.

Each world has 8-39 states, a random marginal and embedding dimension k in 1..5. The second kernel mixes an
unrelated kernel into the first with weight 1e-4 to 1. Half of the worlds are clustered (small spectral gaps).
"""

import numpy as np

from platonic_jepa import (misalignment, mix, random_marginal, random_pair_distribution, spectral_embedding,
                           theorem2_bound)

rng = np.random.default_rng(0)
rows = []
for i in range(300):
    n, k = int(rng.integers(8, 40)), int(rng.integers(1, 6))
    p = random_marginal(n, rng)
    Pi_1 = random_pair_distribution(p, rng, clustered=bool(i % 2))
    Pi_2 = mix(Pi_1, random_pair_distribution(p, rng), 10 ** rng.uniform(-4, 0))
    err = misalignment(spectral_embedding(Pi_1, p, k), spectral_embedding(Pi_2, p, k), p)
    rows.append((err, theorem2_bound(Pi_1, Pi_2, p, k), bool(i % 2)))

err, bound, clustered = (np.array(c) for c in zip(*rows))
ratio = err / bound
print(f"worlds: {len(rows)}   error below the bound in: {(err <= bound + 1e-12).sum()}")
print(f"error / bound   max {ratio.max():.3f}   mean {ratio.mean():.3f}   median {np.median(ratio):.3f}")
print(f"  plain worlds:     max {ratio[~clustered].max():.3f}   mean {ratio[~clustered].mean():.3f}")
print(f"  clustered worlds: max {ratio[clustered].max():.3f}   mean {ratio[clustered].mean():.3f}")
