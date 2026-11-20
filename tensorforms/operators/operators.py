"""The four operators, each written with the formula we proved (no torsion).

                 every slot              wedged (form slots antisymmetrized)
    once         nabla     = d + Γ       wedge_nabla      = d∧ + Γ∧  (value slots)
    twice        curvature = R           wedge_curvature  = R∧       (value slots)

Corrections: + from the left on up slots, − from the right on down slots.
nabla and curvature correct every slot. The wedged two correct only value slots:
the form-slot corrections cancel because Γ is S-symmetric (no torsion).
The new form slot always goes first among the form slots ("first after the bar").
With no metric on the manifold (or conn=False): nabla is d, wedge_nabla is d∧,
and both curvatures are zero.
"""
import math

import numpy as np
import sympy as sp

from ..tensor.slot import Slot
from ..tensor.tensor import Tensor


# ---------------------------------------------------------------- helpers
def _check_tensor(X):
    if not isinstance(X, Tensor):
        raise TypeError("the operators act on tensors")


def _insert_position(X):
    """Where new form slots go: first after the bar."""
    return next((i for i, s in enumerate(X.slots) if s.is_form), len(X.slots))


def _get_conn(X, conn):
    if conn is False:
        return None
    if conn is not None:
        return conn
    M = X.manifold
    return M.connection() if M.has_connection() else None


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
    """C acts on the slots of X: +C on up slots, −C on down slots.
    C[a, b, dirs…] maps slot b to slot a; its m direction axes are placed at pos."""
    n = X.manifold.dim
    total = np.zeros(X.data.shape[:pos] + (n,) * m + X.data.shape[pos:], dtype=object)
    for k, s in enumerate(X.slots):
        if values_only and s.is_form:
            continue
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"slot {s.name} is not in {X.manifold.tangent.name}")
        Xk = np.moveaxis(X.data, k, 0)                           # move slot k into place
        if s.is_up:
            T, sign = np.tensordot(C, Xk, axes=([1], [0])), 1    # +C X
        else:
            T, sign = np.tensordot(C, Xk, axes=([0], [0])), -1   # −X C
        rest = list(range(m + 1, T.ndim))
        xorder = rest[:k] + [0] + rest[k:]                       # move it back
        total = total + sign * np.transpose(T, xorder[:pos] + list(range(1, m + 1)) + xorder[pos:])
    return total


def _wedge_new(Y, p, new):
    """Wedge `new` new form slots with X's p form slots: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))


# ---------------------------------------------------------------- the four operators
def d(X):
    """Plain d: one new form slot, first after the bar. Needs only the manifold."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 1)
    return Tensor(X.manifold, _d(X, pos), slots)


def nabla(X, conn=None):
    """Covariant derivative: ∇X = dX + Γ on every slot."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 1)
    data = _d(X, pos)
    C = _get_conn(X, conn)
    if C is not None:
        data = data + _act(C.Gamma, 1, X, pos)
    return Tensor(X.manifold, data, slots)


def wedge_nabla(X, conn=None):
    """Wedge derivative: ∇∧X = d∧X + Γ∧ on value slots."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt()                                   # wedge factors are antisymmetric first
    pos, slots = _with_new_slots(Xa, 1)
    data = _d(Xa, pos)
    C = _get_conn(X, conn)
    if C is not None:
        data = data + _act(C.Gamma, 1, Xa, pos, values_only=True)
    return _wedge_new(Tensor(X.manifold, data, slots), p, 1)


def curvature(X, conn=None):
    """Curvature operator: ∇∧∇X = R on every slot. X is never differentiated."""
    _check_tensor(X)
    pos, slots = _with_new_slots(X, 2)
    C = _get_conn(X, conn)
    if C is None:
        return Tensor(X.manifold, np.zeros(tuple(s.space.dim for s in slots), dtype=object), slots)
    return Tensor(X.manifold, _act(C.curvature_array(), 2, X, pos), slots)


def wedge_curvature(X, conn=None):
    """Wedge curvature: ∇∧∇∧X = R∧ on value slots. X is never differentiated."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt()
    pos, slots = _with_new_slots(Xa, 2)
    C = _get_conn(X, conn)
    if C is None:
        return Tensor(X.manifold, np.zeros(tuple(s.space.dim for s in slots), dtype=object), slots)
    Y = Tensor(X.manifold, _act(C.curvature_array(), 2, Xa, pos, values_only=True), slots)
    return _wedge_new(Y, p, 2)
