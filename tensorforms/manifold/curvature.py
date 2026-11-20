"""Traces of R: Ricci, scalar curvature, Einstein tensor."""
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.products import contract


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
