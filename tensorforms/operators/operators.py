"""The operators, each written with the formula we proved (no torsion).

                 every slot              wedged (form slots antisymmetrized)
    plain        d                       wedge_d          = d∧
    once         nabla     = d + Γ       wedge_nabla      = d∧ + Γ∧  (value slots)
    twice        curvature = R           wedge_curvature  = R∧       (value slots)

Γ and R always come from the manifold (no torsion: Γ is built from the metric).
Corrections: + from the left on up slots, − from the right on down slots.
nabla and curvature correct every slot. The wedged ones correct only value slots:
the form-slot corrections cancel because Γ is S-symmetric.
New form slots always go first among the form slots.
On a manifold with no metric: nabla is d, wedge_nabla is d∧, and both curvatures are zero.
Results are labelled by the operation: dX, d∧X, ∇X, ∇∧X, ∇∧∇X, ∇∧∇∧X.
"""
import math

import numpy as np
import sympy as sp

from ..tensor.labels import wrap
from ..tensor.products import contract
from ..tensor.slot import Slot
from ..tensor.tensor import Tensor


# ---------------------------------------------------------------- the operators
def d(X):
    """Plain d: one new form slot, first among the form slots. Needs only the coordinates."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 1)
    return Tensor(X.manifold, _d(X, pos), slots, _label("d", X))


def wedge_d(X):
    """Wedge d: d∧X. Antisymmetrize X's form slots, take d, then (p+1)·Alt. No Γ."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt()
    pos, slots = _with_new_slots(Xa, 1)
    Y = _wedge_new(Tensor(X.manifold, _d(Xa, pos), slots), p, 1)
    return Y._new(label=_label("d∧", X))


def nabla(X):
    """Covariant derivative: ∇X = dX + Γ on every slot."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 1)
    data = _d(X, pos)
    G, _ = _geometry(X)
    if G is not None:
        data = data + _act(G, X, slots)
    return Tensor(X.manifold, data, slots, _label("∇", X))


def wedge_nabla(X):
    """Wedge derivative: ∇∧X = d∧X + Γ∧ on value slots."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt()                                   # wedge factors are antisymmetric first
    pos, slots = _with_new_slots(Xa, 1)
    data = _d(Xa, pos)
    G, _ = _geometry(X)
    if G is not None:
        data = data + _act(G, Xa, slots, values_only=True)
    Y = _wedge_new(Tensor(X.manifold, data, slots), p, 1)
    return Y._new(label=_label("∇∧", X))


def curvature(X):
    """Curvature operator: ∇∧∇X = R on every slot. X is never differentiated."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 2)
    _, R = _geometry(X)
    if R is None:
        data = np.zeros(tuple(s.space.dim for s in slots), dtype=object)
    else:
        data = _act(R(), X, slots)
    return Tensor(X.manifold, data, slots, _label("∇∧∇", X))


def wedge_curvature(X):
    """Wedge curvature: ∇∧∇∧X = R∧ on value slots. X is never differentiated."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt()
    pos, slots = _with_new_slots(Xa, 2)
    _, R = _geometry(X)
    if R is None:
        data = np.zeros(tuple(s.space.dim for s in slots), dtype=object)
    else:
        data = _act(R(), Xa, slots, values_only=True)
    Y = _wedge_new(Tensor(X.manifold, data, slots), p, 2)
    return Y._new(label=_label("∇∧∇∧", X))


# ---------------------------------------------------------------- helpers
def _check_tensor(X):
    if not isinstance(X, Tensor):
        raise TypeError("the operators act on tensors")


def _insert_position(X):
    """Where new form slots go: first among the form slots."""
    return next((i for i, s in enumerate(X.slots) if s.is_form), len(X.slots))


def _geometry(X):
    """The manifold's Γ and R as tensors (R computed only when asked), or (None, None) with no metric."""
    M = X.manifold
    if not M.has_metric():
        return None, None
    return M.levi_civita(), M.riemann


def _label(op, X):
    return f"{op}{wrap(X.label)}" if X.label else None


def _new_form_slots(X, count):
    """`count` fresh form slots, with names unused by X."""
    M, used = X.manifold, list(X.names)
    slots = []
    for _ in range(count):
        name = M.tangent.fresh_name(used)
        used.append(name)
        slots.append(Slot(name, M.tangent, "down", "form"))
    return slots


def _with_new_slots(X, count):
    pos = _insert_position(X)
    return pos, X.slots[:pos] + _new_form_slots(X, count) + X.slots[pos:]


def _d(X, pos):
    """Plain d: one copy per coordinate, stacked as a new axis at pos."""
    D = np.stack([np.vectorize(lambda e, xi=xi: sp.diff(e, xi), otypes=[object])(X.data)
                  for xi in X.manifold.coords], axis=-1)
    return np.moveaxis(D, -1, pos)


def _act(C, X, slots, values_only=False):
    """C (Γ or R) acts on each slot of X, as contractions by position (names are only labels):
        up slot s:    +C X   (C's down slot pairs with s; C's up slot takes s's place)
        down slot s:  −X C   (s pairs with C's up slot; C's down slot takes s's place)
    C's form slots become the new form slots, first among the form slots.
    values_only: skip X's form slots (the wedged operators).
    C must be (↑ ↓ forms…): a matrix (one up, one down value slot) with form slots, like Γ and R.
    Every term is put in the result's order: X's value slots (s replaced in place), the new form
    slots, X's form slots (s replaced in place)."""
    if (len(C.slots) < 2 or not C.slots[0].is_up or C.slots[1].is_up
            or C.slots[0].is_form or C.slots[1].is_form or not all(s.is_form for s in C.slots[2:])):
        raise TypeError(f"_act needs C shaped (↑ ↓ forms…) like Γ or R; got {C.declaration()}")
    vals = [j for j, s in enumerate(X.slots) if not s.is_form]     # positions of X's value slots
    forms = [j for j, s in enumerate(X.slots) if s.is_form]        # positions of X's form slots
    m, p, k = len(vals), len(forms), len(C.slots) - 2               # k: the new form slots
    new = list(range(m + p, m + p + k))
    total = np.zeros(tuple(s.space.dim for s in slots), dtype=object)
    for j, s in enumerate(X.slots):
        if values_only and s.is_form:
            continue
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"slot {s.name} is not in {X.manifold.tangent.name}")
        if s.is_up:
            # contract(C, X): [C's up, X's other values, C's forms, X's forms]
            i = vals.index(j)
            T = contract(C, X, 1, j)
            order = list(range(1, i + 1)) + [0] + list(range(i + 1, m + k + p))
            total = total + np.transpose(T.data, order)
        elif not s.is_form:
            # contract(X, C): [X's other values, C's down, X's forms, C's forms]
            i = vals.index(j)
            T = contract(X, C, j, 0)
            order = list(range(i)) + [m - 1] + list(range(i, m - 1)) + new + list(range(m, m + p))
            total = total - np.transpose(T.data, order)
        else:
            # contract(X, C): [X's values, C's down, X's other forms, C's forms]
            i = forms.index(j)
            T = contract(X, C, j, 0)
            order = (list(range(m)) + new
                     + list(range(m + 1, m + 1 + i)) + [m] + list(range(m + 1 + i, m + p)))
            total = total - np.transpose(T.data, order)
    return total


def _wedge_new(Y, p, new):
    """Wedge `new` new form slots with X's p form slots: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))
