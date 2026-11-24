"""The operators, each written with the formula we proved (no torsion). Plus lie(v, X), the Lie
derivative along a vector field (needs no metric).

                 every index              wedged (form indices antisymmetrized)
    plain        d                       wedge_d          = d∧
    once         nabla     = d + Γ       wedge_nabla      = d∧ + Γ∧  (value indices)
    twice        curvature = R           wedge_curvature  = R∧       (value indices)

Γ and R always come from the manifold (no torsion: Γ is built from the metric).
Corrections: + from the left on up indices, − from the right on down indices.
nabla and curvature correct every index. The wedged ones correct only value indices:
the form-index corrections cancel because Γ is S-symmetric.
New form indices always go first among the form indices.
On a manifold with no metric: nabla is d, wedge_nabla is d∧, and both curvatures are zero.
Results have no label (their indices carry the information); name one with X.label = ….
"""
import math

import numpy as np
import sympy as sp

from ..tensor.index import Index
from ..tensor.tensor import Tensor


# ---------------------------------------------------------------- the operators
def d(X):
    """Plain d: one new form index, first among the form indices. Needs only the coordinates."""
    _check_tensor(X)
    pos, indices = _with_new_indices(X, 1)
    return Tensor(X.manifold, _d(X, pos), indices, None, copy=False)


def wedge_d(X):
    """Wedge d: d∧X. Antisymmetrize X's form indices, take d, then (p+1)·Alt. No Γ."""
    _check_tensor(X)
    p = len(X._form_positions())
    Xa = X.alt()
    pos, indices = _with_new_indices(Xa, 1)
    Y = _wedge_new(Tensor(X.manifold, _d(Xa, pos), indices, copy=False), p, 1)
    return Y._new(label=None)


def nabla(X):
    """Covariant derivative: ∇X = dX + Γ on every index."""
    _check_tensor(X)
    pos, indices = _with_new_indices(X, 1)
    data = _d(X, pos)
    G, _ = _geometry(X)
    if G is not None:
        data = data + _act(G, X, indices)
    return Tensor(X.manifold, data, indices, None, copy=False)


def wedge_nabla(X):
    """Wedge derivative: ∇∧X = d∧X + Γ∧ on value indices."""
    _check_tensor(X)
    p = len(X._form_positions())
    Xa = X.alt()                                   # wedge factors are antisymmetric first
    pos, indices = _with_new_indices(Xa, 1)
    data = _d(Xa, pos)
    G, _ = _geometry(X)
    if G is not None:
        data = data + _act(G, Xa, indices, values_only=True)
    Y = _wedge_new(Tensor(X.manifold, data, indices, copy=False), p, 1)
    return Y._new(label=None)


def curvature(X):
    """Curvature operator: ∇∧∇X = R on every index. X is never differentiated."""
    _check_tensor(X)
    pos, indices = _with_new_indices(X, 2)
    _, R = _geometry(X)
    if R is None:
        data = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
    else:
        data = _act(R(), X, indices)
    return Tensor(X.manifold, data, indices, None, copy=False)


def wedge_curvature(X):
    """Wedge curvature: ∇∧∇∧X = R∧ on value indices. X is never differentiated."""
    _check_tensor(X)
    p = len(X._form_positions())
    Xa = X.alt()
    pos, indices = _with_new_indices(Xa, 2)
    _, R = _geometry(X)
    if R is None:
        data = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
    else:
        data = _act(R(), Xa, indices, values_only=True)
    Y = _wedge_new(Tensor(X.manifold, data, indices, copy=False), p, 2)
    return Y._new(label=None)


def lie(v, X):
    """Lie derivative L_v X: how X changes as the flow of the vector field v carries it.
        L_v X = (dX)v − (dv) on each up index + (dv) on each down or form index
    (dX)v: X's derivative along v. dv, the Jacobian of v, says how the flow stretches and turns
    what it carries; it takes the place of Γ in ∇, with the opposite signs. Needs no metric and
    no Γ, only d. For a vector it is the Lie bracket: L_v w = (dw)v − (dv)w = [v, w].
    The result has X's indices. L_v g = 0 says v is a symmetry of the metric (a Killing vector)."""
    _check_tensor(v)
    _check_tensor(X)
    M = X.manifold
    if v.manifold is not M or len(v.indices) != 1 or not v.indices[0].is_up or v.indices[0].is_form \
            or v.indices[0].space is not M.tangent:
        raise TypeError(f"lie needs a vector field v (one up value index, same manifold); got {v.declaration()}")
    coords = M.coords
    diff = lambda e, x: sp.diff(e, x)
    # (dX)v: v^c ∂_c X
    data = sum((v.data[c] * np.vectorize(diff, otypes=[object])(X.data, coords[c]) for c in range(len(coords))),
               np.zeros(X.data.shape, dtype=object))
    # J[a, c] = ∂_c v^a
    J = np.array([[sp.diff(v.data[a], x) for x in coords] for a in range(len(coords))], dtype=object)
    for i, s in enumerate(X.indices):
        if s.space is not M.tangent:
            raise NotImplementedError(f"index {s.label} is not in {M.tangent.name}")
        if s.is_up:                       # −J^a_c X^{…c…}: J's a takes index i's place
            data = data - np.moveaxis(np.tensordot(J, X.data, axes=([1], [i])), 0, i)
        else:                             # +X_{…c…} J^c_b: J's b takes index i's place
            data = data + np.moveaxis(np.tensordot(X.data, J, axes=([i], [0])), -1, i)
    return Tensor(M, data, X.indices, None, copy=False)


# ---------------------------------------------------------------- helpers
def _check_tensor(X):
    if not isinstance(X, Tensor):
        raise TypeError("the operators act on tensors")


def _insert_position(X):
    """Where new form indices go: first among the form indices."""
    return next((i for i, s in enumerate(X.indices) if s.is_form), len(X.indices))


def _geometry(X):
    """The manifold's Γ and R as tensors (R computed only when asked), or (None, None) with no metric."""
    M = X.manifold
    if not M.has_metric():
        return None, None
    return M.levi_civita(), M.riemann


def _new_form_indices(X, count):
    """`count` fresh form indices, with names unused by X."""
    M, used = X.manifold, X._index_labels()
    indices = []
    for _ in range(count):
        name = M.tangent.fresh_name(used)
        used.append(name)
        indices.append(Index(name, M.tangent, "down", "form"))
    return indices


def _with_new_indices(X, count):
    pos = _insert_position(X)
    return pos, (*X.indices[:pos], *_new_form_indices(X, count), *X.indices[pos:])


def _d(X, pos):
    """Plain d: one copy per coordinate, stacked as a new axis at pos."""
    D = np.stack([np.vectorize(lambda e, xi=xi: sp.diff(e, xi), otypes=[object])(X.data)
                  for xi in X.manifold.coords], axis=-1)
    return np.moveaxis(D, -1, pos)


def _act(C, X, indices, values_only=False):
    """C (Γ or R) acts on each index of X (numbers with numpy's tensordot, by position):
        up index s:    +C X   (C's down index pairs with s; C's up index takes s's place)
        down index s:  −X C   (s pairs with C's up index; C's down index takes s's place)
    C's form indices become the new form indices, first among the form indices.
    values_only: skip X's form indices (the wedged operators).
    C must be (↑ ↓ forms…): a matrix (one up, one down value index) with form indices, like Γ and R.
    The result's order is X's indices, s replaced in place, with C's forms inserted first among
    X's form indices (that is `indices`)."""
    if (len(C.indices) < 2 or not C.indices[0].is_up or C.indices[1].is_up
            or C.indices[0].is_form or C.indices[1].is_form or not all(s.is_form for s in C.indices[2:])):
        raise TypeError(f"_act needs C shaped (↑ ↓ forms…) like Γ or R; got {C.declaration()}")
    n, k = len(X.indices), len(C.indices) - 2                   # X's indices; C's form indices (the new ones)
    pos = _insert_position(X)                               # where the new form indices go
    total = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
    for j, s in enumerate(X.indices):
        if values_only and s.is_form:
            continue
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"index {s.label} is not in {X.manifold.tangent.name}")
        rest = lambda t: t if t < j else t - 1             # X's index t among X's other indices
        if s.is_up:
            # tensordot(C, X): [C's up, C's forms, X's other indices]
            data, sign = np.tensordot(C.data, X.data, axes=([1], [j])), 1
            replaced, form = 0, lambda a: 1 + a
            other = lambda t: 1 + k + rest(t)
        else:
            # tensordot(X, C): [X's other indices, C's down, C's forms]
            data, sign = np.tensordot(X.data, C.data, axes=([j], [0])), -1
            replaced, form = n - 1, lambda a: n + a
            other = rest
        source = [replaced if t == j else other(t) for t in range(n)]
        order = source[:pos] + [form(a) for a in range(k)] + source[pos:]
        total = total + sign * np.transpose(data, order)
    return total


def _wedge_new(Y, p, new):
    """Wedge `new` new form indices with X's p form indices: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))
