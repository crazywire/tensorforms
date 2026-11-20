"""Curvature: R from Γ, then its traces Ricci, scalar curvature and the Einstein tensor."""
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.products import contract, wedge


def riemann(Gamma):
    """R = d∧Γ + Γ∧Γ, computed with the operators themselves. Slots (ρ↑ σ↓ | μ ν)."""
    from ..operators.operators import wedge_nabla               # here, to avoid a circular import
    R = (wedge_nabla(Gamma, conn=False) + wedge(Gamma, Gamma)).simplify()
    R = R.rename({0: "ρ", 1: "σ", 2: "μ", 3: "ν"})
    R.label = "R"
    return R


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
    """G = Ric − ½ R_s g (adding is slot by slot), then named (μ↓ ν↓ |)."""
    G = (Ric - g * (sp.Rational(1, 2) * Rs)).simplify()
    G = G.rename({0: "μ", 1: "ν"})
    G.label = "G"
    return G
