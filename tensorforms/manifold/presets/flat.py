"""Flat geometries in different coordinates: Euclidean space (+ + +) and Minkowski spacetime
(− + + +). R = 0 always; Γ ≠ 0 in curved coordinates."""
import sympy as sp

from ..manifold import Manifold

_SPACE_DIM = {"polar": 2, "spherical": 3, "cylindrical": 3}


def _space(n, coords):
    """Coordinates and metric of flat space with n dimensions in the given coordinates."""
    coords = coords.lower()
    if coords != "cartesian" and coords not in _SPACE_DIM:
        raise ValueError(f"unknown coordinates {coords!r}: use cartesian, polar, spherical or cylindrical")
    if coords in _SPACE_DIM:
        if n is not None and n != _SPACE_DIM[coords]:
            raise ValueError(f"{coords} coordinates are {_SPACE_DIM[coords]}D in space, not {n}D")
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
    n = 3 if (n is None and coords.lower() == "cartesian") else n
    xs, diag = _space(n, coords)
    M = Manifold(xs, name=f"E{len(xs)}")
    M.metric(sp.diag(*diag), signature="+" * len(xs))
    return M


def minkowski(n=None, coords="cartesian"):
    """Flat spacetime: −dt² plus flat space in the given coordinates. n is the spacetime
    dimension (default 4); coords as for euclidean (polar: 3D, spherical and cylindrical: 4D)."""
    space_n = None if n is None else n - 1
    if space_n is None and coords.lower() == "cartesian":
        space_n = 3
    xs, diag = _space(space_n, coords)
    t = sp.Symbol("t", real=True)
    M = Manifold([t] + xs, name="Minkowski")
    M.metric(sp.diag(-1, *diag), signature="-" + "+" * len(xs))
    return M
