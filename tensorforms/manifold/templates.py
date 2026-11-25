"""Metric shapes with unknown functions, for solving rather than using.

Each takes the coordinates and returns a new Manifold with that metric.
"""
import itertools

import numpy as np
import sympy as sp

from .manifold import Manifold
from ..errors import InvalidCoordinateError


def general(coords, signature=None, name="M"):
    """g_ij(x): every component an unknown function, symmetric. The signature is only
    what you declare (default all +); nothing about unknown functions can be checked."""
    n = len(coords)
    g = np.empty((n, n), dtype=object)
    for a, b in itertools.combinations_with_replacement(range(n), 2):
        g[a, b] = g[b, a] = sp.Function(f"g{a}{b}")(*coords)
    return Manifold(coords, name, metric=g, signature=signature or "+" * n)


def diagonal(coords, prefix="f", name="M"):
    """diag(f0(x), f1(x), …): the simplest general metric."""
    fs = [sp.Function(f"{prefix}{i}")(*coords) for i in range(len(coords))]
    return Manifold(coords, name, metric=sp.diag(*fs))


def conformally_flat(coords, signature=None, name="M"):
    """e^{2ω(x)} × (the flat metric of the given signature). ω is named so it can't be
    confused with a coordinate φ."""
    signature = signature or "+" * len(coords)
    omega = sp.Function("omega")(*coords)
    signs = [1 if s == "+" else -1 for s in signature]
    return Manifold(coords, name, metric=sp.exp(2 * omega) * sp.diag(*signs), signature=signature)


def static_spherical(coords, name="M"):
    """−A(r) dt² + B(r) dr² + r² (dθ² + sin²θ dφ²). Coordinates (t, r, θ, φ) in that order.
    Setting the Einstein tensor to zero and solving for A, B gives Schwarzschild."""
    coords = list(coords)
    if len(coords) != 4 or [str(c) for c in coords][1:3] != ["r", "theta"]:
        raise InvalidCoordinateError(f"static_spherical: needs coordinates (t, r, θ, φ) in that order; got {tuple(coords)}")
    t, r, th, ph = coords
    A, B = sp.Function("A")(r), sp.Function("B")(r)
    return Manifold(coords, name, metric=sp.diag(-A, B, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="-+++")
