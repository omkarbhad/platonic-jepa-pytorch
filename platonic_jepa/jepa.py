"""Train a JEPA on a finite world by gradient descent (paper equation 1, identity predictor).

The encoder of modality m sees a rendering x = g_m(z) of the world state (a different, injective, random
rendering per modality). The anti-collapse term is a HARD constraint (the paper allows Omega in {0, infinity}):
the encoder output is centred and whitened exactly, as in whitened-MSE JEPAs, so the loss is only the
prediction term

    L(f) = E_pi ||h(g(z)) - h(g(z'))||^2,      h = whiten(f).

The population version is used: all states at once, weighted by p and by the pair distribution Pi. By
Proposition 1 the minimisers are the spectral embedding up to a rotation.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def random_rendering(n_states: int, dim: int, seed: int) -> torch.Tensor:
    """A random injective rendering of n states into R^dim: distinct random points squashed through tanh."""
    g = torch.Generator().manual_seed(seed)
    return torch.tanh(torch.randn(n_states, dim, generator=g, dtype=torch.float64) * 1.5)


def train_jepa(Pi: np.ndarray, p: np.ndarray, k: int, X: torch.Tensor, *, steps: int = 3000, lr: float = 5e-3,
               hidden: int = 128, seed: int = 0) -> Dict:
    """Train an MLP encoder on the rendered states X (n, dim). Returns the whitened embeddings (n, k) and the loss."""
    torch.manual_seed(seed)
    Pi_t, p_t = torch.tensor(Pi, dtype=torch.float64), torch.tensor(p, dtype=torch.float64)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, k)).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def embed():
        f = net(X)
        f = f - (p_t[:, None] * f).sum(0)                                  # centre under p
        C = f.T @ (p_t[:, None] * f) + 1e-9 * torch.eye(k, dtype=torch.float64)
        vals, vecs = torch.linalg.eigh(C)
        return f @ (vecs @ torch.diag(vals ** -0.5) @ vecs.T)                # whiten: E_p h h^T = I

    def pair_loss(h):
        # E_pi ||h(z) - h(z')||^2 = 2 sum_z p_z ||h_z||^2 - 2 sum_zz' Pi_zz' h_z . h_z'
        return 2 * (p_t[:, None] * h * h).sum() - 2 * ((Pi_t @ h) * h).sum()

    for _ in range(steps):
        loss = pair_loss(embed())
        opt.zero_grad()
        loss.backward()
        opt.step()
    h = embed()
    return {"embedding": h.detach().numpy(), "pair_loss": pair_loss(h).item()}
