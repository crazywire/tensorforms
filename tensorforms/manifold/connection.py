"""The connection Γ, built from the metric (so no torsion: Γ is S-symmetric).

Γ is stored as a plain tensor (ρ↑ σ↓ | μ) so you can multiply, wedge and d it,
e.g. R = d∧Γ + Γ∧Γ. It is not tensorial (it doesn't change like a tensor under a
change of coordinates), so ∇Γ has no meaning; nothing stops you writing it.
"""
import itertools

import numpy as np
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.tensor import Tensor
from .metric import as_matrix


def levi_civita(g, ginv):
    """Γ = ½ g⁻¹(dg + dg^S − dg^{ST}), i.e. Γ^a_bc = ½ g^as (∂_c g_sb + ∂_b g_sc − ∂_s g_bc)."""
    M = g.manifold
    n, x = M.dim, M.coords
    gm, gi = as_matrix(g), as_matrix(ginv)
    G = np.empty((n, n, n), dtype=object)
    for a, b, c in itertools.product(range(n), repeat=3):
        G[a, b, c] = clean(sum(
            gi[a, s] * (sp.diff(gm[s, b], x[c]) + sp.diff(gm[s, c], x[b]) - sp.diff(gm[b, c], x[s]))
            for s in range(n)) / 2)
    a, b, c = M.indices(3, names=("ρ", "σ", "μ"))
    return Tensor(M, G, [a.up, b.down, c.form], label="Γ")
