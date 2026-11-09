"""Good views versus bad views (paper section 2.3), on a finite world with a nuisance.

Each observation is (scene z, nuisance nu). Positives of a JEPA are good views (two observations of related scenes,
independent nuisances) mixed with a weight w of bad views (two corruptions of one observation, which share the
nuisance). This is an illustration built from the paper's description, not a reproduction of its experiments."""

import numpy as np

from platonic_jepa import (cross_modal_cost, extend, random_marginal, random_pair_distribution, scene_explained_fraction,
                           spectral_embedding)
from platonic_jepa.spectral import spectrum

rng = np.random.default_rng(6)
n, m, k = 8, 4, 3
p, q = random_marginal(n, rng), rng.dirichlet(np.ones(m))
Pi = random_pair_distribution(p, rng)
print("scene eigenvalues mu_1..mu_3 of the world's own kernel:", np.round(spectrum(Pi, p)[0][1 : k + 1], 3))
print(f"{'bad-view weight w':>18} {'scene fraction rho':>20} {'cross-modal cost':>18}   (cost 0 = perfect alignment, 2k = {2 * k})")
for w in (0.0, 0.02, 0.1, 0.2, 0.5, 1.0):
    Pi_ext, p_ext = extend(Pi, p, q, bad_view_weight=w)
    f = spectral_embedding(Pi_ext, p_ext, k).reshape(n, m, k)
    print(f"{w:>18.2f} {scene_explained_fraction(f, p, q):>20.3f} {cross_modal_cost(f, f, p, q, q):>18.3f}")
