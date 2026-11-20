"""The Manifold: coordinates, and everything that lives on them.

A bare manifold has no metric. Metric, connection and curvature are optional
properties: attached with use_metric, computed when asked. Γ always comes from the metric.
"""
import numpy as np
import sympy as sp

from ..tensor.slot import TANGENT_NAMES, Index, Slot, Space
from ..tensor.tensor import Tensor, contract
from .connection import Connection
from .curvature import einstein, ricci, scalar_curvature
from .metric import Metric, inverse_metric, make_metric

class Manifold:
    def __init__(self, coords, name="M"):
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.name = name
        self.tangent = Space(f"T{name}", self.dim, self, names=TANGENT_NAMES)      # value and form slots live here
        self._g = self._ginv = self._conn = None
        self._cache = {}

    def __repr__(self):
        g = "with a metric" if self._g is not None else "no metric"
        return f"Manifold {self.name}{tuple(self.coords)} ({g})"

    # ---------- indices and names ----------
    def indices(self, n, names=None):
        names = list(names) if names else self.tangent.names[:n]
        idx = [Index(nm, self.tangent) for nm in names]
        return idx if n > 1 else idx[0]

    def fresh_name(self, used, space=None):
        """The first unused name from the space's list (Greek for the tangent space)."""
        used, pool = set(used), (space or self.tangent).names
        for nm in pool:
            if nm not in used:
                return nm
        k = 1
        while f"{pool[0]}{k}" in used:
            k += 1
        return f"{pool[0]}{k}"

    # ---------- tensors ----------
    def tensor(self, *slots, data, label=None):
        return Tensor(self, data, slots, label)

    def scalar(self, value):
        return Tensor(self, np.array(sp.sympify(value), dtype=object), [])

    def vector(self, data, index=None):
        a = index or self.indices(1)
        return Tensor(self, data, [a.up], label="v")

    def covector(self, data, index=None):
        """A row (μ↓ |): a down value slot. Pairs with vectors by default; lower(v) gives one."""
        a = index or self.indices(1)
        return Tensor(self, data, [a.down], label="w")

    def one_form(self, data, index=None):
        """A 1-form (| μ): one form slot. For d∧ and wedge; d of a function gives one."""
        a = index or self.indices(1)
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

    def _move_index(self, X, slot, metric, want_up, verb):
        """g (or g⁻¹) on one value slot of X: only up/down changes; name, position and role stay."""
        if slot is None:
            options = [s.name for s in X.slots if not s.is_form and s.is_up == want_up]
            if len(options) != 1:
                raise TypeError(f"{verb}: say which slot (candidates: {options})")
            slot = options[0]
        i = X.position(slot)
        s = X.slots[i]
        if s.is_form:
            raise TypeError(f"{verb}: {s.name} is a form slot; form slots are never raised or lowered "
                            "(use as_value() first)")
        if s.is_up != want_up:
            raise TypeError(f"{verb}: {s.name} is already {'down' if want_up else 'up'}")
        if s.space is not self.tangent:
            raise TypeError(f"{verb}: {s.name} is not in the tangent space")
        data = np.moveaxis(np.tensordot(metric.data, X.data, axes=([1], [i])), 0, i)
        flipped = Slot(s.name, s.space, "up" if s.variance == "down" else "down", "value")
        return Tensor(self, data, X.slots[:i] + [flipped] + X.slots[i + 1:], X.label)

    def lower(self, X, slot=None):
        """g on one up value slot: ↑ → ↓. Same physics; raise_ undoes it exactly.
        lower(v) for a vector; lower(R, "ρ") for a named slot."""
        return self._move_index(X, slot, self.g, True, "lower")

    def raise_(self, X, slot=None):
        """g⁻¹ on one down value slot: ↓ → ↑. Same physics; lower undoes it exactly."""
        return self._move_index(X, slot, self.ginv, False, "raise_")

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
