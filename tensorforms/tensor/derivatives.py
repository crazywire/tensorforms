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
        return _wedge_new(self.alt().d(), p, 1)._new(label=None)

    def nabla(self):
        """Covariant derivative: ∇X = dX + Γ on every index."""
        G, _ = _geometry(self)
        if G is None:
            return self.d()
        _tangent_only(self, "nabla")
        return self.d() + sum((1 if s.is_up else -1) * self.apply(G, i)    # + on up, − on down and form
                              for i, s in enumerate(self.indices))

    def wedge_nabla(self):
        """Wedge derivative: ∇∧X = d∧X + Γ∧ on value indices."""
        p = len(self._form_positions())
        Xa = self.alt()                                   # wedge factors are antisymmetric first
        Y = Xa.d()
        G, _ = _geometry(self)
        if G is not None:
            _tangent_only(self, "wedge_nabla")
            Y = Y + sum((1 if s.is_up else -1) * Xa.apply(G, i)          # value indices only
                        for i, s in enumerate(Xa.indices) if not s.is_form)
        return _wedge_new(Y, p, 1)._new(label=None)

    def curvature(self):
        """Curvature operator: ∇∧∇X = R on every index. X is never differentiated."""
        Y = _zero_with_new_indices(self, 2)
        _, R = _geometry(self)
        if R is None:
            return Y
        _tangent_only(self, "curvature")
        R = R()
        return Y + sum((1 if s.is_up else -1) * self.apply(R, i) for i, s in enumerate(self.indices))

    def wedge_curvature(self):
        """Wedge curvature: ∇∧∇∧X = R∧ on value indices. X is never differentiated."""
        p = len(self._form_positions())
        Xa = self.alt()
        Y = _zero_with_new_indices(Xa, 2)
        _, R = _geometry(self)
        if R is not None:
            _tangent_only(self, "wedge_curvature")
            R = R()
            Y = Y + sum((1 if s.is_up else -1) * Xa.apply(R, i) for i, s in enumerate(Xa.indices) if not s.is_form)
        return _wedge_new(Y, p, 2)._new(label=None)

    def lie(self, v):
        """Lie derivative of X along the vector field v. The result has X's indices; no metric needed.
            L_v X = dX(v) − J on each up index + J on each down or form index,  J = (dv).as_value()
        dX(v) evaluates d's new form index (first among the form indices) on v."""
        M = self.manifold
        if isinstance(v, type(self)):
            _assert_same_manifold(self, v)
        if (not isinstance(v, type(self)) or len(v.indices) != 1
                or not v.indices[0].is_up or v.indices[0].is_form or v.indices[0].space is not M.tangent):
            raise IndexMismatchError("lie: v must be a vector field (one up value index); "
                            f"got {getattr(v, 'declaration', lambda: v)()}")
        _tangent_only(self, "lie")
        J = v.d().as_value()                                        # J^a_c = ∂_c v^a
        return self.d()(v) + sum((-1 if s.is_up else 1) * self.apply(J, i)   # − on up, + on down and form
                                 for i, s in enumerate(self.indices))


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


def _tangent_only(X, what):
    """The derivatives correct tangent-space indices only (Γ and R act there)."""
    for s in X.indices:
        if s.space is not X.manifold.tangent:
            raise NotImplementedError(f"{what}: index {s.label} is not in {X.manifold.tangent.label}; "
                                      "only tangent-space indices are supported")


def _zero_with_new_indices(X, count):
    """The zero tensor with X's indices and `count` new form indices (the curvatures start here)."""
    pos, indices = _with_new_indices(X, count)
    return type(X)(X.manifold, np.zeros(tuple(s.space.dim for s in indices), dtype=object), indices, None, copy=False)


def _with_new_indices(X, count):
    pos = _insert_position(X)
    return pos, (*X.indices[:pos], *_new_form_indices(X, count), *X.indices[pos:])


def _d(X, pos):
    """Plain d: one copy per coordinate, stacked as a new axis at pos."""
    D = np.stack([np.vectorize(lambda e, xi=xi: sp.diff(e, xi), otypes=[object])(X.data)
                  for xi in X.manifold.coords], axis=-1)
    return np.moveaxis(D, -1, pos)


def _wedge_new(Y, p, new):
    """Wedge `new` new form indices with X's p form indices: C(p+new; new, p) · Alt."""
    if p == 0:
        return Y
    return Y.alt() * sp.Rational(math.factorial(p + new), math.factorial(new) * math.factorial(p))
