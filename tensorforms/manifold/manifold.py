"""The Manifold: coordinates, and everything that lives on them.

The metric is given when the manifold is made and never changes, so nothing computed
from it can go stale. Without one the manifold is bare: forms, d and d∧ still work.
Connection and curvature are computed when asked; Γ always comes from the metric.
"""

import numpy as np
import sympy as sp

from ..tensor.index import Index
from ..tensor.tensor import Tensor
from .connection import levi_civita
from .curvature import einstein, ricci, riemann, ricci_scalar
from .metric import build_metric, inverse_metric
from .space import TANGENT_NAMES, Space
from .volume import volume_form
from ..errors import MissingMetricError


class Manifold:
    def __init__(self, coords, label=None, metric=None):
        """label: just a name for printing (optional).
        metric: a matrix, or a tensor with two down value indices; checked here."""
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.label = label
        self.tangent = Space(f"T{self}", self.dim, names=TANGENT_NAMES)      # value and form indices live here
        self._g = self._ginv = None
        self._cache = {}        # Γ, R, Ric, R_s, G: computed once
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
        names = list(names) if names else self.tangent.names[:n]
        return [Index(nm, self.tangent) for nm in names]

    def index(self, name=None):
        """One index in the tangent space."""
        return Index(name or self.tangent.names[0], self.tangent)

    # ---------- tensors ----------
    def tensor(self, *indices, data, label=None):
        return Tensor(self, data, indices, label)

    def scalar(self, value):
        return Tensor(self, np.array(sp.sympify(value), dtype=object), [], copy=False)

    def vector(self, data, index=None):
        a = index or self.index()
        return Tensor(self, data, [a.up], label="v")

    def covector(self, data, index=None):
        """A row (μ↓): a down value index. Pairs with vectors by default; lower(v) gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.down], label="w")

    def one_form(self, data, index=None):
        """A 1-form (μ): one form index. For d∧ and wedge; d of a function gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.form], label="α")

    def map(self, data, indices=None):
        a, b = indices or self.indices(2)
        return Tensor(self, data, [a.up, b.down], label="A")

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
        return self._cached("Γ", lambda: levi_civita(self.g, self.ginv))

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
