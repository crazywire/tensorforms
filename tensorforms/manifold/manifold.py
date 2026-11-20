"""The Manifold: coordinates, and everything that lives on them.

A bare manifold has no metric. Metric, connection and curvature are optional
properties: attached with use_metric, computed when asked. Γ always comes from the metric.
"""
import copy

import numpy as np
import sympy as sp

from ..tensor.slot import Index
from ..tensor.tensor import Tensor
from .connection import Connection
from .curvature import einstein, ricci, scalar_curvature
from .metric import Metric, inverse_metric, make_metric
from .space import TANGENT_NAMES, Space


class Manifold:
    def __init__(self, coords, name="M"):
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.name = name
        self.tangent = Space(f"T{name}", self.dim, names=TANGENT_NAMES)      # value and form slots live here
        self._g = self._ginv = None
        self._cache = {}        # Γ, R, Ric, R_s, G: computed once, cleared when the metric changes

    def __repr__(self):
        g = "with a metric" if self._g is not None else "no metric"
        return f"Manifold {self.name}{tuple(self.coords)} ({g})"

    # ---------- indices and names ----------
    def indices(self, n, names=None):
        """A list of n indices in the tangent space (always a list, even for n = 1)."""
        names = list(names) if names else self.tangent.names[:n]
        return [Index(nm, self.tangent) for nm in names]

    def index(self, name=None):
        """One index in the tangent space."""
        return Index(name or self.tangent.names[0], self.tangent)

    # ---------- tensors ----------
    def tensor(self, *slots, data, label=None):
        return Tensor(self, data, slots, label)

    def scalar(self, value):
        return Tensor(self, np.array(sp.sympify(value), dtype=object), [])

    def vector(self, data, index=None):
        a = index or self.index()
        return Tensor(self, data, [a.up], label="v")

    def covector(self, data, index=None):
        """A row (μ↓ |): a down value slot. Pairs with vectors by default; lower(v) gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.down], label="w")

    def one_form(self, data, index=None):
        """A 1-form (| μ): one form slot. For d∧ and wedge; d of a function gives one."""
        a = index or self.index()
        return Tensor(self, data, [a.form], label="α")

    def map(self, data, indices=None):
        a, b = indices or self.indices(2)
        return Tensor(self, data, [a.up, b.down], label="A")

    # ---------- metric ----------
    def metric(self, data=None, signature=None):
        """data → that metric; no data → unknown functions with the given signature."""
        return make_metric(self, data, signature)

    def use_metric(self, g):
        if not isinstance(g, Metric):
            raise TypeError("use_metric needs a metric made by M.metric(...)")
        if not g.is_lower:
            raise TypeError("use_metric needs g with two down slots; "
                            "this looks like the inverse metric g⁻¹ (M.ginv)")
        self._g, self._ginv = g, inverse_metric(g)
        self._cache.clear()
        return self

    @property
    def g(self):
        if self._g is None:
            raise AttributeError("this manifold has no metric yet: call M.use_metric(M.metric(...))")
        return copy.copy(self._g)

    @property
    def ginv(self):
        self.g                                                 # raises if there is no metric
        return copy.copy(self._ginv)

    @property
    def signature(self):
        return self.g.signature_string

    # ---------- connection and curvature ----------
    def connection(self):
        return self._cached("Γ", lambda: Connection.from_metric(self.g))

    def has_connection(self):
        return self._g is not None

    def _cached(self, key, compute):
        """Compute once and keep it; hand out a copy so changing a label never changes the stored one.
        The copy shares the numbers (nothing in the package edits them in place)."""
        if key not in self._cache:
            self._cache[key] = compute()
        X = self._cache[key]
        return copy.copy(X) if isinstance(X, Tensor) else X     # Γ is shared: it keeps its own R cache

    def curvature(self):
        return self._cached("R", lambda: self.connection().curvature())

    def ricci(self):
        return self._cached("Ric", lambda: ricci(self.curvature()))

    def scalar_curvature(self):
        return self._cached("Rs", lambda: scalar_curvature(self.ricci(), self.ginv))

    def gaussian_curvature(self):
        if self.dim != 2:
            raise ValueError("Gaussian curvature is for 2D surfaces")
        return sp.simplify(self.scalar_curvature() / 2)

    def einstein(self):
        return self._cached("G", lambda: einstein(self.ricci(), self.scalar_curvature(), self.g))
