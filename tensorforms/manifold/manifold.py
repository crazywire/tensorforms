"""The Manifold: coordinates, and everything that lives on them.

A bare manifold has no metric. M.metric(...) makes, checks and attaches one;
connection and curvature are then computed when asked. Γ always comes from the metric.
"""
import copy

import numpy as np
import sympy as sp

from ..tensor.slot import Index
from ..tensor.tensor import Tensor
from .connection import levi_civita
from .curvature import einstein, ricci, riemann, scalar_curvature
from .metric import build_metric, inverse_metric
from .space import TANGENT_NAMES, Space


class Manifold:
    def __init__(self, coords, name="M"):
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.name = name
        self.tangent = Space(f"T{name}", self.dim, names=TANGENT_NAMES)      # value and form slots live here
        self._g = self._ginv = self._signature = None
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
        """Make, check and attach the metric; returns g.
        data: a matrix (or a tensor with two down value slots). No data: unknown functions
        g_ij(x), with the signature you declare (e.g. "-+++"). Calling it again replaces
        the metric and clears everything computed from the old one."""
        g, sig = build_metric(self, data, signature)
        self._g, self._ginv, self._signature = g, inverse_metric(g), sig
        self._cache.clear()
        return copy.copy(g)

    @property
    def g(self):
        if self._g is None:
            raise AttributeError("this manifold has no metric yet: call M.metric(...)")
        return copy.copy(self._g)

    @property
    def ginv(self):
        self.g                                                 # raises if there is no metric
        return copy.copy(self._ginv)

    @property
    def signature(self):
        """The metric's sign pattern, e.g. "-+++" (None if it couldn't be told)."""
        self.g                                                 # raises if there is no metric
        return self._signature

    # ---------- connection and curvature ----------
    def connection(self):
        """Γ (ρ↑ σ↓ | μ), from the metric."""
        return self._cached("Γ", lambda: levi_civita(self.g, self.ginv))

    def has_connection(self):
        return self._g is not None

    def _cached(self, key, compute):
        """Compute once and keep it; hand out a copy so changing a label never changes the stored one.
        The copy shares the numbers (nothing in the package edits them in place)."""
        if key not in self._cache:
            self._cache[key] = compute()
        X = self._cache[key]
        return copy.copy(X)

    def curvature(self):
        return self._cached("R", lambda: riemann(self.connection()))

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
