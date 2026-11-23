"""Curvature: R from Γ, then its traces Ricci, scalar curvature and the Einstein tensor."""
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.products import wedge


def riemann(Gamma):
    """R = d∧Γ + Γ∧Γ, computed with the operators themselves. Slots (ρ↑ σ↓ μ ν)."""
    from ..operators.operators import wedge_d                   # here, to avoid a circular import
    R = (wedge_d(Gamma) + wedge(Gamma, Gamma)).simplify()
    R = R.rename({0: "ρ", 1: "σ", 2: "μ", 3: "ν"})
    R.label = "R"
    return R


def ricci(R):
    """Ric = tr(R^S): swap R's 2nd and 3rd slots, then pair the first two.
    The trace leaves (σ↓ ν); Ric is returned as (μ↓ ν↓) so it has g's type
    and can be added to g in G = Ric − ½R_s g."""
    Ric = R.S().tr(0, 1).as_value().simplify()
    Ric = Ric.rename({0: "μ", 1: "ν"})
    Ric.label = "Ric"
    return Ric


def ricci_scalar(Ric):
    """R_s = tr(g⁻¹ Ric): raising Ric's first slot is multiplying by g⁻¹; then take the trace."""
    return clean(Ric.raise_(0).tr(0, 1).data.item())


def einstein(Ric, Rs, g):
    """G = Ric − ½ R_s g (adding is slot by slot), then named (μ↓ ν↓)."""
    G = (Ric - g * (sp.Rational(1, 2) * Rs)).simplify()
    G = G.rename({0: "μ", 1: "ν"})
    G.label = "G"
    return G
