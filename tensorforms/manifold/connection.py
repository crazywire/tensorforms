"""The connection Γ, built from the metric (so no torsion: Γ is S-symmetric).
Deliberately NOT a Tensor: it doesn't change like one under a frame change."""
import itertools

import numpy as np
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.tensor import Tensor
from .metric import as_matrix


class Connection:
    """Γ^a_{bc}, with c the direction (step) slot."""

    def __init__(self, manifold, Gamma):
        self.manifold = manifold
        self.Gamma = np.asarray(Gamma, dtype=object)
        n = manifold.dim
        if self.Gamma.shape != (n, n, n):
            raise ValueError(f"Γ must have shape {(n, n, n)}")

    @classmethod
    def from_metric(cls, g):
        """Levi-Civita: Γ = ½ g⁻¹(dg + dg^S − dg^{ST})."""
        M = g.manifold
        n, x = M.dim, M.coords
        gm = as_matrix(g)
        gi = sp.simplify(gm.inv())
        G = np.empty((n, n, n), dtype=object)
        for a, b, c in itertools.product(range(n), repeat=3):
            G[a, b, c] = clean(sum(
                gi[a, s] * (sp.diff(gm[s, b], x[c]) + sp.diff(gm[s, c], x[b]) - sp.diff(gm[b, c], x[s]))
                for s in range(n)) / 2)
        return cls(M, G)

    def __getitem__(self, key):
        return self.Gamma[key]

    def curvature_array(self):
        """R^a_{b c e} = ∂_cΓ^a_{be} − ∂_eΓ^a_{bc} + Γ^a_{sc}Γ^s_{be} − Γ^a_{se}Γ^s_{bc}  (R = d∧Γ + Γ∧Γ).
        Computed once per connection, then reused by the curvature operators."""
        if getattr(self, "_R", None) is not None:
            return self._R
        n, x, G = self.manifold.dim, self.manifold.coords, self.Gamma
        R = np.empty((n,) * 4, dtype=object)
        for a, b, c, e in itertools.product(range(n), repeat=4):
            R[a, b, c, e] = clean(
                sp.diff(G[a, b, e], x[c]) - sp.diff(G[a, b, c], x[e])
                + sum(G[a, s, c] * G[s, b, e] - G[a, s, e] * G[s, b, c] for s in range(n)))
        self._R = R
        return R

    def curvature(self):
        M = self.manifold
        a, b, c, d = M.indices(4, names=("ρ", "σ", "μ", "ν"))
        return Tensor(M, self.curvature_array(), [a.up, b.down, c.form, d.form], label="R")

    def __repr__(self):
        return f"Connection on {self.manifold.name} (not a tensor)"
