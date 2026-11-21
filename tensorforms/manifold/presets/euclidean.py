"""Flat space ℝⁿ in different coordinates. R = 0 always; Γ ≠ 0 in curved coordinates."""
import sympy as sp

from ..manifold import Manifold

_DIM = {"polar": 2, "spherical": 3, "cylindrical": 3}


def euclidean(n=None, coords="cartesian"):
    """Flat space. coords: "cartesian" (any n, default 3), "polar" (2D),
    "spherical" (3D) or "cylindrical" (3D); any capitalisation.
    For polar, spherical and cylindrical, n can be left out."""
    coords = coords.lower()
    if coords != "cartesian" and coords not in _DIM:
        raise ValueError(f"unknown coordinates {coords!r}: use cartesian, polar, spherical or cylindrical")
    if coords in _DIM:
        if n is not None and n != _DIM[coords]:
            raise ValueError(f"{coords} coordinates are {_DIM[coords]}D, not {n}D")
        n = _DIM[coords]
    n = 3 if n is None else n
    r = sp.Symbol("r", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    if coords == "cartesian":
        names = "x y z" if n <= 3 else " ".join(f"x{i}" for i in range(1, n + 1))
        xs = sp.symbols(names, real=True)
        xs = list(xs)[:n] if isinstance(xs, (tuple, list)) else [xs]
        M = Manifold(xs, name=f"E{n}")
        M.metric(sp.eye(n), signature="+" * n)
    elif coords == "polar":
        M = Manifold([r, ph], name="E2")
        M.metric(sp.diag(1, r ** 2), signature="++")
    elif coords == "spherical":
        M = Manifold([r, th, ph], name="E3")
        M.metric(sp.diag(1, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="+++")
    else:
        z = sp.Symbol("z", real=True)
        M = Manifold([r, ph, z], name="E3")
        M.metric(sp.diag(1, r ** 2, 1), signature="+++")
    return M
