"""4D Lorentzian geometries, signature (−,+,+,+), in units with G = c = 1
(so a mass m is a length, GM/c²). r is positive; t, θ, φ and the Cartesian coordinates are real."""
import sympy as sp

from ..manifold import Manifold


def minkowski():
    t, x, y, z = sp.symbols("t x y z", real=True)
    M = Manifold([t, x, y, z], name="Minkowski")
    M.metric(sp.diag(-1, 1, 1, 1), signature="-+++")
    return M


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
            raise ValueError(f"a uses a symbol named {wrong[0].name!r} that is not the manifold's coordinate "
                             f"(declared with different assumptions). Use sp.Symbol(\"t\", real=True), "
                             f"or pass a function: flrw(a=lambda t: ...)")
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
