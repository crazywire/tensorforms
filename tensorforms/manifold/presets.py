"""Ready-made manifolds, reached as Manifold.<name>(...).

Ordered like the docs: flat, then curved Riemannian (+ + +), then curved Lorentzian (− + + +).
Spacetimes use units with G = c = 1 (so a mass m is a length, GM/c²). r is positive; t, θ, φ
and the Cartesian coordinates are real."""
import sympy as sp

from .manifold import Manifold
from ..errors import InvalidCoordinateError


def _radius(value, name):
    return sp.Symbol(name, positive=True) if value is None else sp.sympify(value)


# --- flat: R = 0 everywhere (Γ ≠ 0 in curved coordinates) ---

_SPACE_DIM = {"polar": 2, "spherical": 3, "cylindrical": 3}


def _space(n, coords):
    """Coordinates and metric of flat space with n dimensions in the given coordinates."""
    coords = coords.lower()
    if coords != "cartesian" and coords not in _SPACE_DIM:
        raise InvalidCoordinateError(f"coordinates: unknown {coords!r}; use cartesian, polar, spherical or cylindrical")
    if coords in _SPACE_DIM:
        if n is not None and n != _SPACE_DIM[coords]:
            raise InvalidCoordinateError(f"coordinates: {coords} is {_SPACE_DIM[coords]}D in space, not {n}D")
        n = _SPACE_DIM[coords]
    r = sp.Symbol("r", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    if coords == "cartesian":
        names = "x y z" if n <= 3 else " ".join(f"x{i}" for i in range(1, n + 1))
        xs = sp.symbols(names, real=True)
        xs = list(xs)[:n] if isinstance(xs, (tuple, list)) else [xs]
        return xs, [1] * n
    if coords == "polar":
        return [r, ph], [1, r ** 2]
    if coords == "spherical":
        return [r, th, ph], [1, r ** 2, r ** 2 * sp.sin(th) ** 2]
    z = sp.Symbol("z", real=True)
    return [r, ph, z], [1, r ** 2, 1]


def euclidean(n=None, coords="cartesian"):
    """Flat space. coords: "cartesian" (any n, default 3), "polar" (2D),
    "spherical" (3D) or "cylindrical" (3D); any capitalisation."""
    if n is not None and (not isinstance(n, int) or n < 1):
        raise ValueError(f"euclidean: n must be a whole number, at least 1; got {n!r}")
    n = 3 if (n is None and coords.lower() == "cartesian") else n
    xs, diag = _space(n, coords)
    M = Manifold(xs, name=f"E{len(xs)}")
    M.metric(sp.diag(*diag), signature="+" * len(xs))
    return M


def minkowski(n=None, coords="cartesian"):
    """Flat spacetime: −dt² plus flat space in the given coordinates. n is the spacetime
    dimension (default 4); coords as for euclidean (polar: 3D, spherical and cylindrical: 4D)."""
    if n is not None and (not isinstance(n, int) or n < 2):
        raise ValueError(f"minkowski: n must be a whole number, at least 2 (time and one space dimension); got {n!r}")
    space_n = None if n is None else n - 1
    if space_n is None and coords.lower() == "cartesian":
        space_n = 3
    xs, diag = _space(space_n, coords)
    t = sp.Symbol("t", real=True)
    M = Manifold([t] + xs, name="Minkowski")
    M.metric(sp.diag(-1, *diag), signature="-" + "+" * len(xs))
    return M


def cylinder(r=None):
    """Flat but wraps around: coordinates (φ, z)."""
    r = _radius(r, "r")
    ph = sp.Symbol("phi", real=True)
    z = sp.Symbol("z", real=True)
    M = Manifold([ph, z], name="Cyl")
    M.metric(sp.diag(r ** 2, 1), signature="++")
    return M


# --- curved, Riemannian (+ + +) ---

def sphere(r=None):
    """Round sphere. r=None keeps the radius symbolic (K = 1/r²)."""
    r = _radius(r, "r")
    th, ph = sp.symbols("theta phi", real=True)
    M = Manifold([th, ph], name="S2")
    M.metric(sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), signature="++")
    return M


def torus(R=None, r=None):
    """Torus: θ around the tube, φ around the hole. K > 0 outside, K < 0 inside."""
    R, r = _radius(R, "R"), _radius(r, "r")
    th, ph = sp.symbols("theta phi", real=True)
    M = Manifold([th, ph], name="T2")
    M.metric(sp.diag(r ** 2, (R + r * sp.cos(th)) ** 2), signature="++")
    return M


def hyperbolic_plane():
    """Upper half-plane, y > 0: K = −1 everywhere."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", positive=True)
    M = Manifold([x, y], name="H2")
    M.metric(sp.diag(1 / y ** 2, 1 / y ** 2), signature="++")
    return M


# --- curved, Lorentzian (− + + +) ---

def schwarzschild(m=None):
    """Outside a round, non-rotating mass m (or black hole). Vacuum: Ric = 0.
    The signature −+++ holds outside the horizon, r > 2m; inside, t and r swap roles."""
    m = sp.Symbol("m", positive=True) if m is None else sp.sympify(m)
    t, th, ph = sp.symbols("t theta phi", real=True)
    r = sp.Symbol("r", positive=True)
    f = 1 - 2 * m / r
    M = Manifold([t, r, th, ph], name="Schwarzschild")
    M.metric(sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="-+++")
    return M


def flrw(a=None, k=0):
    """Expanding universe. a(t) is the scale factor; k = 0, +1, −1.
    a: None (an unknown function a(t)), a function of t (e.g. lambda t: t**Rational(2, 3)),
    or an expression in the preset's t, which is sp.Symbol("t", real=True).
    For k = +1 the signature −+++ holds for r < 1."""
    try:
        k = sp.sympify(k, strict=True)
    except sp.SympifyError:
        raise TypeError(f"flrw: k must be a number or an expression (0, 1 or −1); got {k!r}") from None
    t, th, ph = sp.symbols("t theta phi", real=True)
    r = sp.Symbol("r", positive=True)
    coords = [t, r, th, ph]
    if a is None:
        a = sp.Function("a")(t)
    elif callable(a):
        a = sp.sympify(a(t))
    else:
        a = sp.sympify(a)
        names = {c.name: c for c in coords}
        wrong = [s for s in a.free_symbols if s.name in names and s != names[s.name]]
        if wrong:
            raise InvalidCoordinateError(f"flrw: a uses a symbol {wrong[0].name!r} that isn't the manifold's coordinate "
                             f"(different assumptions); use sp.Symbol(\"t\", real=True) or pass a function, "
                             f"a=lambda t: ...")
    M = Manifold(coords, name="FLRW")
    g = sp.diag(-1, a ** 2 / (1 - k * r ** 2), a ** 2 * r ** 2, a ** 2 * r ** 2 * sp.sin(th) ** 2)
    M.metric(g, signature="-+++")
    return M


def de_sitter(H=None):
    """Accelerating universe (flat slicing): −dt² + e^{2Ht}(dx² + dy² + dz²)."""
    H = sp.Symbol("H", positive=True) if H is None else sp.sympify(H)
    t, x, y, z = sp.symbols("t x y z", real=True)
    e = sp.exp(2 * H * t)
    M = Manifold([t, x, y, z], name="deSitter")
    M.metric(sp.diag(-1, e, e, e), signature="-+++")
    return M
