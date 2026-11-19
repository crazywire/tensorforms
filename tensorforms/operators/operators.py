"""The four operators.

                 every slot          wedged (form slots antisymmetrized)
    once         nabla   = d + Γ     wedge_nabla      = d∧ + Γ∧
    twice        curvature = R       wedge_curvature  = R∧

Each corrects every slot: + from the left for up slots, − from the right for
down slots. With no connection (a bare manifold, or conn=False) they become the
flat versions: nabla is d, wedge_nabla is d∧, and both curvatures vanish.
The new form slot always goes first among the form slots ("first after the bar").
"""
import numpy as np
import sympy as sp

from ..manifold.connection import Connection
from ..tensor.slot import Slot
from ..tensor.tensor import Tensor


def _check_tensor(X):
    if isinstance(X, Connection):
        raise TypeError("Γ is not a tensor, so the operators don't apply to it (∇Γ has no meaning)")
    if not isinstance(X, Tensor):
        raise TypeError("the operators act on tensors")


def _insert_position(X):
    forms = [i for i, s in enumerate(X.slots) if s.role == "form"]
    return forms[0] if forms else len(X.slots)


def _get_conn(X, conn):
    if conn is False:
        return None
    if conn is not None:
        return conn
    M = X.manifold
    return M.connection() if M.has_connection() else None


def _d(X):
    M = X.manifold
    name = M.fresh_name(X.names)
    D = np.stack([np.vectorize(lambda e, xi=xi: sp.diff(e, xi), otypes=[object])(X.data)
                  for xi in M.coords], axis=-1)
    pos = _insert_position(X)
    D = np.moveaxis(D, -1, pos)
    new = Slot(name, M.tangent, "down", "form")
    return Tensor(M, D, X.slots[:pos] + [new] + X.slots[pos:]), pos


def _act(C, m, X, pos):
    """C acts on every slot of X: +C on up slots, −C on down slots.
    C[a, b, dirs…] maps slot b to slot a; its m direction axes are placed at pos."""
    n = X.manifold.dim
    if not X.slots:
        return np.zeros((n,) * m, dtype=object)
    total = None
    for k, s in enumerate(X.slots):
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"slot {s.name} is not in {X.manifold.tangent.name}; Γ only acts on the tangent bundle")
        Xk = np.moveaxis(X.data, k, 0)                       # move slot k into place
        if s.variance == "up":
            T, sign = np.tensordot(C, Xk, axes=([1], [0])), 1    # +C X
        else:
            T, sign = np.tensordot(C, Xk, axes=([0], [0])), -1   # −X C
        rest = list(range(m + 1, T.ndim))
        xorder = rest[:k] + [0] + rest[k:]                   # move it back
        final = xorder[:pos] + list(range(1, m + 1)) + xorder[pos:]
        T = sign * np.transpose(T, final)
        total = T if total is None else total + T
    return total


def d(X):
    """Plain d: one new form slot, first after the bar. Needs only the manifold."""
    _check_tensor(X)
    return _d(X)[0]


def nabla(X, conn=None):
    """Covariant derivative: d + Γ on every slot."""
    _check_tensor(X)
    D, pos = _d(X)
    C = _get_conn(X, conn)
    if C is None:
        return D
    return Tensor(D.manifold, D.data + _act(C.Gamma, 1, X, pos), D.slots)


def wedge_nabla(X, conn=None):
    """Exterior covariant derivative: ∇, then antisymmetrize all form slots.
    (p+1)!/(p!·1!) · Alt — the form-slot corrections cancel when there is no torsion."""
    _check_tensor(X)
    p = len(X.form_slots)
    Xa = X.alt() if p > 1 else X
    Y = nabla(Xa, conn)
    if p == 0:
        return Y
    return Y.alt(*Y.form_slots) * (p + 1)


def curvature(X, conn=None):
    """Curvature operator ∇∧∇: ∇ twice, antisymmetrize only the two new form slots.
    Acts on every slot of X with R, never differentiates X."""
    _check_tensor(X)
    pos = _insert_position(X)
    Y = nabla(nabla(X, conn), conn)
    return Y - Y.swap_positions(pos, pos + 1)


def curvature_by_R(X, conn=None):
    """The same operator written directly: R on every slot of X."""
    _check_tensor(X)
    C = _get_conn(X, conn)
    M = X.manifold
    pos = _insert_position(X)
    used = list(X.names)
    n1 = M.fresh_name(used)
    n2 = M.fresh_name(used + [n1])
    slots = X.slots[:pos] + [Slot(n1, M.tangent, "down", "form"), Slot(n2, M.tangent, "down", "form")] + X.slots[pos:]
    if C is None:
        return Tensor(M, np.zeros(tuple(s.space.dim for s in slots), dtype=object), slots)
    return Tensor(M, _act(C.curvature_array(), 2, X, pos), slots)


def wedge_curvature(X, conn=None):
    """Exterior twice, ∇∧∇∧ = R∧ on value slots."""
    return wedge_nabla(wedge_nabla(X, conn), conn)
