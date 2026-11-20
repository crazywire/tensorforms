"""Curvature: R from Γ, then its traces Ricci, scalar curvature and the Einstein tensor."""
import itertools

import numpy as np
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.products import contract
from ..tensor.tensor import Tensor


def riemann(Gamma):
    """R = d∧Γ + Γ∧Γ, i.e.
    R^a_bce = ∂_cΓ^a_be − ∂_eΓ^a_bc + Γ^a_sc Γ^s_be − Γ^a_se Γ^s_bc.  Slots (ρ↑ σ↓ | μ ν)."""
    M = Gamma.manifold
    n, x, G = M.dim, M.coords, Gamma.data
    R = np.empty((n,) * 4, dtype=object)
    for a, b, c, e in itertools.product(range(n), repeat=4):
        R[a, b, c, e] = clean(
            sp.diff(G[a, b, e], x[c]) - sp.diff(G[a, b, c], x[e])
            + sum(G[a, s, c] * G[s, b, e] - G[a, s, e] * G[s, b, c] for s in range(n)))
    a, b, c, d = M.indices(4, names=("ρ", "σ", "μ", "ν"))
    return Tensor(M, R, [a.up, b.down, c.form, d.form], label="R")


def ricci(R):
    """Ric = tr(R^S): pair R's up slot with its first form slot.
    The trace leaves (σ↓ | ν). Ric is returned as (σ↓ ν↓ |) so it has g's type
    and can be added to g in G = Ric − ½R_s g."""
    up = next(s.name for s in R.slots if s.is_up)
    first_form = R.form_slots[0]
    Ric = R.tr(up, first_form).as_value().simplify()
    Ric.label = "Ric"
    return Ric


def scalar_curvature(Ric, ginv):
    i, j = ginv.names
    a, b = Ric.names
    mixed = contract(ginv, Ric, j, a)          # (i↑, b↓)
    s = mixed.tr(i, b)
    return clean(s.data.item())


def einstein(Ric, Rs, g):
    """G = Ric − ½ R_s g, with Ric's slots renamed to match g."""
    Ric2 = Ric.rename(dict(zip(Ric.names, g.names)))
    G = (Ric2 - g * (sp.Rational(1, 2) * Rs)).simplify()
    G.label = "G"
    return G
