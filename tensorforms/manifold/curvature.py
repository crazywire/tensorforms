"""Traces of R: Ricci, scalar curvature, Einstein tensor."""
import sympy as sp

from ..symbolic.simplify import clean
from ..tensor.products import contract
from ..tensor.tensor import Tensor


def _forms_to_values(X):
    """The same numbers, with every form slot relabelled as a value slot."""
    return Tensor(X.manifold, X.data, [s.with_role("value") if s.is_form else s for s in X.slots])


def ricci(R):
    """Ric = tr(R^S): pair R's up slot with its first form slot.
    The trace leaves (σ↓ | ν). Ric is returned as (σ↓ ν↓ |) so it has g's type
    and can be added to g in G = Ric − ½R_s g."""
    up = next(s.name for s in R.slots if s.is_up)
    first_form = R.form_slots[0]
    Ric = _forms_to_values(R.tr(up, first_form)).simplify()
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
