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
from .labels import _new_label
from ..errors import IndexMismatchError
from .products import _assert_same_manifold


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
            data = data + sum((1 if s.is_up else -1) * _apply(self, i, G)    # + on up, − on down and form
                              for i, s in enumerate(self.indices))
        return type(self)(self.manifold, data, indices, None, copy=False)

    def wedge_nabla(self):
        """Wedge derivative: ∇∧X = d∧X + Γ∧ on value indices."""
        p = len(self._form_positions())
        Xa = self.alt()                                   # wedge factors are antisymmetric first
        pos, indices = _with_new_indices(Xa, 1)
        data = _d(Xa, pos)
        G, _ = _geometry(self)
        if G is not None:
            data = data + sum((1 if s.is_up else -1) * _apply(Xa, i, G)      # value indices only
                              for i, s in enumerate(Xa.indices) if not s.is_form)
        Y = _wedge_new(type(self)(self.manifold, data, indices, copy=False), p, 1)
        return Y._new(label=None)

    def curvature(self):
        """Curvature operator: ∇∧∇X = R on every index. X is never differentiated."""
        pos, indices = _with_new_indices(self, 2)
        _, R = _geometry(self)
        if R is None:
            data = np.zeros(tuple(s.space.dim for s in indices), dtype=object)
        else:
            R = R()
            data = sum(((1 if s.is_up else -1) * _apply(self, i, R) for i, s in enumerate(self.indices)),
                       np.zeros(tuple(s.space.dim for s in indices), dtype=object))
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
            R = R()
            data = sum(((1 if s.is_up else -1) * _apply(Xa, i, R) for i, s in enumerate(Xa.indices) if not s.is_form),
                       np.zeros(tuple(s.space.dim for s in indices), dtype=object))
        Y = _wedge_new(type(self)(self.manifold, data, indices, copy=False), p, 2)
        return Y._new(label=None)

    def lie(self, v):
        """Lie derivative of X along the vector field v. The result has X's indices; no metric needed.
            L_v X = dX(v) − J on each up index + J on each down or form index,  J = (dv).as_value()
        dX(v) evaluates d's new form index on v."""
        M = self.manifold
        if isinstance(v, type(self)):
            _assert_same_manifold(v, M)
        if (not isinstance(v, type(self)) or len(v.indices) != 1
                or not v.indices[0].is_up or v.indices[0].is_form or v.indices[0].space is not M.tangent):
            raise IndexMismatchError("lie: v must be a vector field (one up value index); "
                            f"got {getattr(v, 'declaration', lambda: v)()}")
        # dX(v): d's new form index (first among the form indices) evaluated on v; the order is kept
        data = np.tensordot(v.data, self.d().data, axes=([0], [_insert_position(self)]))
        J = v.d().as_value()                                        # J^a_c = ∂_c v^a
        data = data + sum((-1 if s.is_up else 1) * _apply(self, i, J)   # − on up, + on down and form
                          for i, s in enumerate(self.indices))
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
        name = _new_label(M.tangent, M, used)
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


def _apply(X, i, A):
    """The numbers of A applied to X's index i, in X's layout. A has exactly two value indices
    (and any form indices). The one of A's value indices that pairs with X's index i is summed
    against it (A's second if both could); the other takes index i's place. A's form indices are
    added first among X's form indices, where d puts new ones. Used by raise_/lower (g, g⁻¹),
    the Lie derivative (J) and the covariant and curvature derivatives (Γ, R)."""
    s = X.indices[i]
    vals, forms = A._value_positions(), A._form_positions()
    if len(vals) != 2:
        raise IndexMismatchError(f"apply: needs exactly two value indices; got {A.declaration()}")
    if s.space is not X.manifold.tangent:
        raise NotImplementedError(f"apply: index {s.label} is not in {X.manifold.tangent.label}; only tangent-space indices are supported")
    first, second = vals
    eat, put = (second, first) if A.index(second).pairs_with(s) else (first, second)
    if not A.index(eat).pairs_with(s):
        raise IndexMismatchError(f"apply: {A.declaration()} has no index that pairs with {s.label}")
    data = np.tensordot(A.data, X.data, axes=([eat], [i]))      # [A's put and forms, in A's order; X's others]
    a_rest = [m for m in range(len(A.indices)) if m != eat]
    put_axis = a_rest.index(put)
    form_axes = [a_rest.index(f) for f in forms]
    n, k = len(X.indices), len(forms)
    x_axis = {m: k + 1 + pos for pos, m in enumerate(m for m in range(n) if m != i)}
    source = [put_axis if m == i else x_axis[m] for m in range(n)]
    pos = _insert_position(X)
    return np.transpose(data, source[:pos] + form_axes + source[pos:])


def _wedge_new(Y, p, new):
    """Wedge `new` new form indices with X's p form indices: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))
