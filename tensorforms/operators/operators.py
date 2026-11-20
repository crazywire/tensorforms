"""The operators, each written with the formula we proved (no torsion).

                 every slot              wedged (form slots antisymmetrized)
    plain        d                       wedge_d          = d∧
    once         nabla     = d + Γ       wedge_nabla      = d∧ + Γ∧  (value slots)
    twice        curvature = R           wedge_curvature  = R∧       (value slots)

Γ and R always come from the manifold (no torsion: Γ is built from the metric).
Corrections: + from the left on up slots, − from the right on down slots.
nabla and curvature correct every slot. The wedged ones correct only value slots:
the form-slot corrections cancel because Γ is S-symmetric.
New form slots always go first among the form slots ("first after the bar").
On a manifold with no metric: nabla is d, wedge_nabla is d∧, and both curvatures are zero.
Results are labelled by the operation: dX, d∧X, ∇X, ∇∧X, ∇∧∇X, ∇∧∇∧X.
"""
import math

import numpy as np
import sympy as sp

from ..tensor.labels import wrap
from ..tensor.slot import Slot
from ..tensor.tensor import Tensor


# ---------------------------------------------------------------- helpers
def _check_tensor(X):
    if not isinstance(X, Tensor):
        raise TypeError("the operators act on tensors")


def _insert_position(X):
    """Where new form slots go: first after the bar."""
    return next((i for i, s in enumerate(X.slots) if s.is_form), len(X.slots))


def _geometry(X):
    """The manifold's (Γ, R) as arrays (R computed only when asked), or (None, None) with no metric."""
    M = X.manifold
    if not M.has_connection():
        return None, None
    return M.connection().data, (lambda: M.curvature().data)


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


def _act(C, m, X, pos, values_only=False):
    """C (Γ or R) acts on the slots of X: +C from the left on up slots, −C from the right on down slots.
    C[a, b, forms…] maps slot b to slot a; its m form slots are placed at pos (first after the bar).
    values_only: skip X's form slots (the wedged operators)."""
    n = X.manifold.dim
    total = np.zeros(X.data.shape[:pos] + (n,) * m + X.data.shape[pos:], dtype=object)
    for k, s in enumerate(X.slots):
        if values_only and s.is_form:
            continue
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"slot {s.name} is not in {X.manifold.tangent.name}")
        Xk = np.moveaxis(X.data, k, 0)                           # bring slot k to the front
        if s.is_up:
            T, sign = np.tensordot(C, Xk, axes=([1], [0])), 1    # +C X
        else:
            T, sign = np.tensordot(C, Xk, axes=([0], [0])), -1   # −X C
        rest = list(range(m + 1, T.ndim))
        xorder = rest[:k] + [0] + rest[k:]                       # put slot k back where it was
        total = total + sign * np.transpose(T, xorder[:pos] + list(range(1, m + 1)) + xorder[pos:])
    return total


def _wedge_new(Y, p, new):
    """Wedge `new` new form slots with X's p form slots: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))


# ---------------------------------------------------------------- the operators
def d(X):
    """Plain d: one new form slot, first after the bar. Needs only the coordinates."""
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
        data = data + _act(G, 1, X, pos)
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
        data = data + _act(G, 1, Xa, pos, values_only=True)
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
        data = _act(R(), 2, X, pos)
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
        data = _act(R(), 2, Xa, pos, values_only=True)
    Y = _wedge_new(Tensor(X.manifold, data, slots), p, 2)
    return Y._new(label=_label("∇∧∇∧", X))
