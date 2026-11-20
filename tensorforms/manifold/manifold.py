"""The Manifold: coordinates, and everything that lives on them.

A bare manifold has no metric. Metric, connection and curvature are optional
properties: attached with use_metric, computed when asked. Γ always comes from the metric.
"""
import numpy as np
import sympy as sp

from ..tensor.slot import Index, Space
from ..tensor.tensor import Tensor, contract
from .connection import Connection
from .curvature import einstein, ricci, scalar_curvature
from .metric import Metric, inverse_metric, make_metric

_NAME_POOL = ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "p", "q",
              "α", "β", "γ", "δ", "ε", "ζ", "η", "κ", "λ", "ξ", "τ", "χ", "ψ", "ω"]


class Manifold:
    def __init__(self, coords, name="M"):
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.name = name
        self.tangent = Space(f"T{name}", self.dim, self)      # value and form slots live here
        self._g = self._ginv = self._conn = None
        self._cache = {}

    def __repr__(self):
        g = "with a metric" if self._g is not None else "no metric"
        return f"Manifold {self.name}{tuple(self.coords)} ({g})"

    # ---------- indices and names ----------
    def indices(self, n, names=None):
        names = list(names) if names else _NAME_POOL[:n]
        idx = [Index(nm, self.tangent) for nm in names]
        return idx if n > 1 else idx[0]

    def fresh_name(self, used):
        used = set(used)
        for nm in _NAME_POOL:
            if nm not in used:
                return nm
        k = 1
        while f"x{k}" in used:
            k += 1
        return f"x{k}"

    # ---------- tensors ----------
    def tensor(self, *slots, data, label=None):
        return Tensor(self, data, slots, label)

    def scalar(self, value):
        return Tensor(self, np.array(sp.sympify(value), dtype=object), [])

    def vector(self, data, index=None):
        a = index or self.indices(1)
        return Tensor(self, data, [a.up], label="v")

    def covector(self, data, index=None):
        a = index or self.indices(1)
        return Tensor(self, data, [a.form], label="w")

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
        self._g, self._ginv, self._conn = g, inverse_metric(g), None
        self._cache.clear()
        return self

    @property
    def g(self):
        if self._g is None:
            raise AttributeError("this manifold has no metric yet: call M.use_metric(M.metric(...))")
        return self._g

    @property
    def ginv(self):
        self.g                                                 # raises if there is no metric
        return self._ginv

    @property
    def signature(self):
        return self.g.signature_string

    def lower(self, v):
        """gv: pair g's second slot with v's (single) up slot."""
        return contract(self.g, v, self.g.names[1], v.names[0])

    def raise_(self, w):
        return contract(self.ginv, w, self.ginv.names[1], w.names[0])

    # ---------- connection and curvature ----------
    def connection(self):
        if self._conn is None:
            self._conn = Connection.from_metric(self.g)
        return self._conn

    def has_connection(self):
        return self._g is not None

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

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
