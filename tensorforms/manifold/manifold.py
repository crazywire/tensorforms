"""The Manifold: coordinates, and everything that lives on them.

The metric is given when the manifold is made and never changes, so nothing computed
from it can go stale. Without one the manifold is bare: forms, d and d∧ still work.
Connection and curvature are computed when asked; Γ always comes from the metric.
"""

import numpy as np
import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from ..tensor.index import Index
from ..tensor.tensor import Tensor
from .connection import levi_civita
from .curvature import einstein, ricci, riemann, ricci_scalar
from .metric import build_metric, inverse_metric
from .space import Space
from ..tensor.labels import GREEK_LETTERS, LATIN_LETTERS
from .volume import volume_form
from ..symbolic.simplify import is_zero as _is_zero
from ..errors import IndexMismatchError, InvalidCoordinateError, MissingMetricError, NonInvertibleError


class Manifold:
    def __init__(self, coords, label=None, metric=None, ambient=None, map=None):
        """label: just a name for printing (optional).
        metric: a matrix, or a tensor with two down value indices; checked here.
        ambient, map: for a submanifold, the manifold it sits in and every ambient coordinate
        written in this one's coordinates, e.g. {x: cos(u), y: sin(u)}. The metric then comes
        from the ambient's (pulled back); C.map is the map as a tensor and C.map.d() is E, C's coordinate frame in ambient components."""
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.label = label
        self.tangent = Space(f"T{self}", self.dim)      # value and form indices live here
        self._g = self._ginv = None
        self._cache = {}        # Γ, R, Ric, R_s, G: computed once
        self.ambient, self.map = ambient, None
        if (ambient is None) != (map is None):
            raise TypeError("Manifold: ambient and map go together")
        if ambient is not None:
            if metric is not None:
                raise TypeError("Manifold: a submanifold's metric comes from its ambient manifold; don't give one")
            self.map = _map_tensor(self, ambient, map)
            if ambient.has_metric():
                metric = self.pullback(ambient.g).simplify()    # substituting leaves e.g. sin² + cos²
        if metric is not None:
            self._g = build_metric(self, metric)
            self._ginv = inverse_metric(self._g)

    def __repr__(self):
        g = "with a metric" if self._g is not None else "no metric"
        return f"Manifold {self.label}{tuple(self.coords)} ({g})" if self.label else f"Manifold{tuple(self.coords)} ({g})"

    def __str__(self):
        """The label, or the coordinates if there is none: what messages call this manifold."""
        return self.label or str(tuple(self.coords))

    # ---------- ready-made manifolds (class methods: Manifold.sphere(), …) ----------
    @classmethod
    def euclidean(cls, n=None, coords="cartesian"):
        """Flat space: coords "cartesian" (any n, default 3), "polar", "spherical", "cylindrical"."""
        from .presets import euclidean
        return euclidean(n, coords)

    @classmethod
    def sphere(cls, r=None):
        """Sphere of radius r (symbolic if not given): K = 1/r²."""
        from .presets import sphere
        return sphere(r)

    @classmethod
    def cylinder(cls, r=None):
        """Cylinder of radius r: flat, K = 0."""
        from .presets import cylinder
        return cylinder(r)

    @classmethod
    def torus(cls, R=None, r=None):
        """Torus with radii R (around the hole) and r (of the tube): K changes sign."""
        from .presets import torus
        return torus(R, r)

    @classmethod
    def hyperbolic_plane(cls):
        """Upper half-plane, (dx² + dy²)/y²: K = −1."""
        from .presets import hyperbolic_plane
        return hyperbolic_plane()

    @classmethod
    def minkowski(cls, n=None, coords="cartesian"):
        """Flat spacetime: −dt² plus flat space in the given coordinates (default 4D Cartesian)."""
        from .presets import minkowski
        return minkowski(n, coords)

    @classmethod
    def schwarzschild(cls, m=None):
        """Outside a round, non-rotating mass m (G = c = 1): vacuum."""
        from .presets import schwarzschild
        return schwarzschild(m)

    @classmethod
    def flrw(cls, a=None, k=0):
        """Expanding universe with scale factor a(t) and spatial curvature k."""
        from .presets import flrw
        return flrw(a, k)

    @classmethod
    def de_sitter(cls, H=None):
        """Accelerating universe, −dt² + e^{2Ht}(dx² + dy² + dz²): R_s = 12H²."""
        from .presets import de_sitter
        return de_sitter(H)

    # ---------- indices and names ----------
    def indices(self, n, names=None):
        """A list of n indices in the tangent space (always a list, even for n = 1)."""
        names = list(names) if names else GREEK_LETTERS[:n]
        return [Index(nm, self.tangent) for nm in names]

    def index(self, name=None):
        """One index in the tangent space."""
        return Index(name or GREEK_LETTERS[0], self.tangent)

    # ---------- tensors ----------
    def tensor(self, *indices, data, label=None):
        return Tensor(self, data, indices, label)

    def scalar(self, value, label=None):
        return Tensor(self, np.array(sp.sympify(value), dtype=object), [], label, copy=False)

    def vector(self, data, index=None, label=None):
        a = index or self.index()
        return Tensor(self, data, [a.up], label)

    def covector(self, data, index=None, label=None):
        """A row (μ↓): a down value index. Pairs with vectors by default; lower(v) gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.down], label)

    def one_form(self, data, index=None, label=None):
        """A 1-form (μ): one form index. For d∧ and wedge; d of a function gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.form], label)

    # ---------- submanifold ----------
    def pullback(self, X):
        """X, a tensor on the ambient manifold, carried onto this submanifold: substitute the map
        into the components, then multiply each down or form index by the frame E = self.map.d(). Up indices
        only when this manifold has the ambient's dimension (a change of coordinates): each is
        multiplied by E⁻¹. Otherwise an up index may point off this manifold, so it is refused.
        Each index keeps its label, arrow and type; only its space changes."""
        if self.ambient is None:
            raise TypeError(f"pullback: {self} has no ambient manifold (make it with ambient=..., map=...)")
        if not isinstance(X, Tensor):
            raise TypeError(f"pullback: needs a tensor; got {type(X).__name__}")
        P = self.ambient
        if X.manifold is not P:
            raise IndexMismatchError(f"pullback: the tensor is on {X.manifold}, but {self} sits in {P}")
        for s in X.indices:
            if s.is_up and self.dim != P.dim:
                raise IndexMismatchError(f"pullback: {s.label} is an up index; only down and form indices carry over "
                                         f"(up indices only when the dimensions match: a change of coordinates)")
            if s.space is not P.tangent:
                raise NotImplementedError(f"pullback: index {s.label} is not in {P.tangent.label}")
        at = dict(zip(P.coords, self.map.data))
        data = np.vectorize(lambda e: sp.sympify(e).subs(at, simultaneous=True), otypes=[object])(X.data)
        E = self.map.d().data                                 # the frame: E[a, i] = ∂x^a/∂u^i
        if any(s.is_up for s in X.indices):
            try:                                              # E⁻¹[i, a] = ∂u^i/∂x^a
                Einv = np.array(sp.Matrix(E.tolist()).inv(iszerofunc=_is_zero).tolist(), dtype=object)
            except NonInvertibleMatrixError:
                raise NonInvertibleError("pullback: the map's frame E is not invertible (determinant 0), "
                                         "so up indices can't be carried") from None
        for k, s in enumerate(X.indices):                     # each index, kept in its position
            if s.is_up:
                data = np.moveaxis(np.tensordot(data, Einv, axes=([k], [1])), -1, k)
            else:
                data = np.moveaxis(np.tensordot(data, E, axes=([k], [0])), -1, k)
        indices = [Index(s.label, self.tangent, s.variance, s.type) for s in X.indices]
        return Tensor(self, data, indices, copy=False)

    # ---------- metric ----------
    @property
    def g(self):
        if self._g is None:
            raise MissingMetricError("metric: this manifold has none yet; make the manifold with metric=...")
        return self._g._new(label=self._g.label)

    @property
    def ginv(self):
        self.g                                                 # raises if there is no metric
        return self._ginv._new(label=self._ginv.label)

    # ---------- connection and curvature ----------
    def levi_civita(self):
        """Γ (ρ↑ σ↓ μ), from the metric."""
        return self._cached("Γ", lambda: levi_civita(self.g))

    def volume_form(self):
        """ε (μ ν …) = √|det g| dx¹ ∧ … ∧ dxⁿ: n-dimensional volume, from the metric."""
        return self._cached("ε", lambda: volume_form(self))

    def has_metric(self):
        return self._g is not None

    def _cached(self, key, compute):
        """Compute once and keep it; hand out a new tensor sharing the numbers (copy on write),
        so changing yours never changes the stored one."""
        if key not in self._cache:
            self._cache[key] = compute()
        X = self._cache[key]
        return X._new(label=X.label)

    def riemann(self):
        """The Riemann curvature R (ρ↑ σ↓ μ ν)."""
        return self._cached("R", lambda: riemann(self.levi_civita()))

    def ricci(self):
        return self._cached("Ric", lambda: ricci(self.riemann()))

    def ricci_scalar(self):
        return self._cached("Rs", lambda: ricci_scalar(self.ricci()))

    def gaussian_curvature(self):
        """K = R_s / 2, a scalar tensor (2D surfaces only)."""
        if self.dim != 2:
            raise ValueError("gaussian_curvature: only for 2D surfaces")

        def K():
            k = (self.ricci_scalar() / 2).simplify()
            k.label = "K"
            return k
        return self._cached("K", K)

    def einstein(self):
        return self._cached("G", lambda: einstein(self.ricci(), self.ricci_scalar(), self.g))


def _map_tensor(C, P, map):
    """The map as a tensor on C: one up index in P's tangent space (so a Latin label)."""
    if not isinstance(map, dict):
        raise TypeError(f"Manifold: map must be a dictionary {{ambient coordinate: expression}}; got {type(map).__name__}")
    if set(map) != set(P.coords):
        raise InvalidCoordinateError(f"map: must give every ambient coordinate {tuple(P.coords)}; got {tuple(map)}")
    data = [sp.sympify(map[x]) for x in P.coords]
    stray = {s for e in data for s in e.free_symbols} & (set(P.coords) - set(C.coords))
    if stray:
        raise InvalidCoordinateError(f"map: uses {sorted(stray, key=str)}, ambient coordinates that aren't among {tuple(C.coords)}")
    a = Index(LATIN_LETTERS[0], P.tangent)
    return Tensor(C, data, [a.up])
