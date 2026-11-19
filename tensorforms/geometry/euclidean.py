"""Flat space ℝⁿ in different coordinates. R = 0 always; Γ ≠ 0 in curved coordinates."""
import sympy as sp

from ..manifold.manifold import Manifold


def euclidean(n=3, coords="cartesian"):
    if coords == "cartesian":
        names = "x y z" if n <= 3 else " ".join(f"x{i}" for i in range(1, n + 1))
        xs = sp.symbols(names, real=True)
        xs = list(xs)[:n] if isinstance(xs, (tuple, list)) else [xs]
        M = Manifold(xs, name=f"E{n}")
        return M.use_metric(M.metric(sp.eye(n), signature="+" * n))
    if coords == "polar":
        if n != 2:
            raise ValueError("polar coordinates are 2D")
        r, ph = sp.symbols("r phi", positive=True)
        M = Manifold([r, ph], name="E2")
        return M.use_metric(M.metric(sp.diag(1, r ** 2), signature="++"))
    if coords == "spherical":
        if n != 3:
            raise ValueError("spherical coordinates are 3D")
        r, th, ph = sp.symbols("r theta phi", positive=True)
        M = Manifold([r, th, ph], name="E3")
        return M.use_metric(M.metric(sp.diag(1, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="+++"))
    if coords == "cylindrical":
        if n != 3:
            raise ValueError("cylindrical coordinates are 3D")
        r, ph = sp.symbols("r phi", positive=True)
        z = sp.Symbol("z", real=True)
        M = Manifold([r, ph, z], name="E3")
        return M.use_metric(M.metric(sp.diag(1, r ** 2, 1), signature="+++"))
    raise ValueError(f"unknown coordinates {coords!r}")
