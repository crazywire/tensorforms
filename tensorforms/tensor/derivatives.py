"""The derivatives, each written with the formula we proved (no torsion). Plus X.lie(v), the Lie
derivative along a vector field (needs no metric). They are tensor methods: Tensor inherits Derivatives.

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

from .index import Index
from ..errors import IndexMismatchError


# ---------------------------------------------------------------- the derivatives
class Derivatives:
    """The derivatives, as tensor methods (Tensor inherits them): X.d(), X.nabla(), X.lie(v), …"""

    def d(self):
        """Plain d: one new form index, first among the form indices. Needs only the coordinates."""
        pos, indices = _with_new_indices(self, 1)
        return type(self)(self.manifold, _d(self, pos), indices, None, copy=False)

    def wedge_d(self):
        """Wedge d: d∧X. Antisymmetrize X's form indices, take d, then (p+1)·Alt. No Γ."""
        p = len(self._form_positions())
        Xa = self.alt()
        pos, indices = _with_new_indices(Xa, 1)
        Y = _wedge_new(type(self)(self.manifold, _d(Xa, pos), indices, copy=False), p, 1)
        return Y._new(label=None)

    def nabla(self):
        """Covariant derivative: ∇X = dX + Γ on every index."""
        pos, indices = _with_new_indices(self, 1)
        data = _d(self, pos)
        G, _ = _geometry(self)
        if G is not None:
            data = data + _act(G, self, indices)
        return type(self)(self.manifold, data, indices, None, copy=False)

    def wedge_nabla(self):
        """Wedge derivative: ∇∧X = d∧X + Γ∧ on value indices."""
        p = len(self._form_positions())
        Xa = self.alt()                                   # wedge factors are antisymmetric first
        pos, indices = _with_new_indices(Xa, 1)
        data = _d(Xa, pos)
        G, _ = _geometry(self)
        if G is not None:
            data = data + _act(G, Xa, indices, values_only=True)
        Y = _wedge_new(type(self)(self.manifold, data, indices, copy=False), p, 1)
        return Y._new(label=None)

    def curvature(self):
        """Curvature operator: ∇∧∇X = R on every index. X is never differentiated."""
        pos, indices = _with_new_indices(self, 2)
        _, R = _geometry(self)
        if R is None:
            data = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
        else:
            data = _act(R(), self, indices)
        return type(self)(self.manifold, data, indices, None, copy=False)

    def wedge_curvature(self):
        """Wedge curvature: ∇∧∇∧X = R∧ on value indices. X is never differentiated."""
        p = len(self._form_positions())
        Xa = self.alt()
        pos, indices = _with_new_indices(Xa, 2)
        _, R = _geometry(self)
        if R is None:
            data = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
        else:
            data = _act(R(), Xa, indices, values_only=True)
        Y = _wedge_new(type(self)(self.manifold, data, indices, copy=False), p, 2)
        return Y._new(label=None)

    def lie(self, v):
        """Lie derivative of X along the vector field v. The result has X's indices; no metric needed.
            L_v X = (dX)v − J on each up index + J on each down or form index,  J = (dv).as_value()"""
        M = self.manifold
        if (not isinstance(v, type(self)) or v.manifold is not M or len(v.indices) != 1
                or not v.indices[0].is_up or v.indices[0].is_form or v.indices[0].space is not M.tangent):
            raise IndexMismatchError("lie: v must be a vector field (one up value index, same manifold); "
                            f"got {getattr(v, 'declaration', lambda: v)()}")
        # (dX)v: d's new form index (first among the form indices) paired with v; the order is kept
        data = np.tensordot(v.data, self.d().data, axes=([0], [_insert_position(self)]))
        J = v.d().as_value().data                                   # J[a, c] = ∂_c v^a
        for i, s in enumerate(self.indices):
            if s.space is not M.tangent:
                raise NotImplementedError(f"lie: index {s.label} is not in {M.tangent.name}; only tangent-space indices are supported")
            if s.is_up:                       # −J^a_c X^{…c…}: J's a takes index i's place
                data = data - np.moveaxis(np.tensordot(J, self.data, axes=([1], [i])), 0, i)
            else:                             # +X_{…c…} J^c_b: J's b takes index i's place
                data = data + np.moveaxis(np.tensordot(self.data, J, axes=([i], [0])), -1, i)
        return type(self)(M, data, self.indices, None, copy=False)


# ---------------------------------------------------------------- helpers

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
    values_only: skip X's form indices (the wedged derivatives).
    C must be (↑ ↓ forms…): a matrix (one up, one down value index) with form indices, like Γ and R.
    The result's order is X's indices, s replaced in place, with C's forms inserted first among
    X's form indices (that is `indices`)."""
    if (len(C.indices) < 2 or not C.indices[0].is_up or C.indices[1].is_up
            or C.indices[0].is_form or C.indices[1].is_form or not all(s.is_form for s in C.indices[2:])):
        raise TypeError(f"_act: C must be shaped (↑ ↓ forms…) like Γ or R; got {C.declaration()}")
    n, k = len(X.indices), len(C.indices) - 2                   # X's indices; C's form indices (the new ones)
    pos = _insert_position(X)                               # where the new form indices go
    total = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
    for j, s in enumerate(X.indices):
        if values_only and s.is_form:
            continue
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"derivative: index {s.label} is not in {X.manifold.tangent.name}; only tangent-space indices are supported")
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
