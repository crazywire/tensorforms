"""Traces of R: Ricci, scalar curvature, Einstein tensor."""
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.tensor import contract, tr


def ricci(R):
    """Ric = tr(R^S): pair R's up slot with its first form slot.
    Ric is symmetric (no torsion), so both remaining slots are returned as value slots."""
    up = next(s.name for s in R.slots if s.is_up)
    first_form = R.form_slots[0]
    Ric = tr(R, up, first_form).with_roles("value").simplify()
    Ric.label = "Ric"
    return Ric


def scalar_curvature(Ric, ginv):
    i, j = ginv.names
    a, b = Ric.names
    mixed = contract(ginv, Ric, j, a)          # (i↑, b↓)
    s = tr(mixed, i, b)
    return clean(s.data.item())


def einstein(Ric, Rs, g):
    """G = Ric − ½ R_s g, with Ric's slots renamed to match g."""
    Ric2 = Ric.rename(dict(zip(Ric.names, g.names)))
    G = (Ric2 - g * (sp.Rational(1, 2) * Rs)).simplify()
    G.label = "G"
    return G
