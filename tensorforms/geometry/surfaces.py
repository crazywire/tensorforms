"""2D surfaces: positive, zero, mixed and negative curvature."""
import sympy as sp

from ..manifold.manifold import Manifold


def _radius(value, name):
    return sp.Symbol(name, positive=True) if value is None else sp.sympify(value)


def sphere(r=None):
    """Round sphere. r=None keeps the radius symbolic (K = 1/r²)."""
    r = _radius(r, "r")
    th, ph = sp.symbols("theta phi", positive=True)
    M = Manifold([th, ph], name="S2")
    return M.use_metric(M.metric(sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), signature="++"))


def cylinder(r=None):
    """Flat but wraps around: coordinates (φ, z)."""
    r = _radius(r, "r")
    ph = sp.Symbol("phi", positive=True)
    z = sp.Symbol("z", real=True)
    M = Manifold([ph, z], name="Cyl")
    return M.use_metric(M.metric(sp.diag(r ** 2, 1), signature="++"))


def torus(R=None, r=None):
    """Torus: θ around the tube, φ around the hole. K > 0 outside, K < 0 inside."""
    R, r = _radius(R, "R"), _radius(r, "r")
    th, ph = sp.symbols("theta phi", real=True)
    M = Manifold([th, ph], name="T2")
    return M.use_metric(M.metric(sp.diag(r ** 2, (R + r * sp.cos(th)) ** 2)))


def hyperbolic_plane():
    """Upper half-plane, y > 0: K = −1 everywhere."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", positive=True)
    M = Manifold([x, y], name="H2")
    return M.use_metric(M.metric(sp.diag(1 / y ** 2, 1 / y ** 2), signature="++"))
