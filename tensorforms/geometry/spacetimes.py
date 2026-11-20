"""4D Lorentzian geometries, signature (−,+,+,+)."""
import sympy as sp

from ..manifold.manifold import Manifold


def minkowski():
    t, x, y, z = sp.symbols("t x y z", real=True)
    M = Manifold([t, x, y, z], name="Minkowski")
    M.metric(sp.diag(-1, 1, 1, 1), signature="-+++")
    return M


def schwarzschild(m=None):
    """Outside a round, non-rotating mass m (or black hole). Vacuum: Ric = 0."""
    m = sp.Symbol("m", positive=True) if m is None else sp.sympify(m)
    t = sp.Symbol("t", real=True)
    r, th, ph = sp.symbols("r theta phi", positive=True)
    f = 1 - 2 * m / r
    M = Manifold([t, r, th, ph], name="Schwarzschild")
    M.metric(sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="-+++")
    return M


def flrw(a=None, k=0):
    """Expanding universe. a(t) is the scale factor (unknown function by default); k = 0, +1, −1."""
    t = sp.Symbol("t", real=True)
    r, th, ph = sp.symbols("r theta phi", positive=True)
    a = sp.Function("a")(t) if a is None else a
    M = Manifold([t, r, th, ph], name="FLRW")
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
