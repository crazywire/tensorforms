"""The volume form ε = √|det g| dx¹ ∧ … ∧ dxⁿ, built with the operations themselves."""
import sympy as sp


def volume_form(M):
    """ε (μ ν …): n-dimensional volume (length in 1D, area in 2D, …).
    Orientation: the order of the coordinates. |det g| keeps it real in any signature."""
    w = M.scalar(M.coords[0]).d()                 # dx¹
    for c in M.coords[1:]:
        w = w.wedge(M.scalar(c).d())             # dx¹ ∧ dx² ∧ … ∧ dxⁿ
    size = sp.sqrt(sp.Abs(sp.Matrix(M.g.data).det()))   # real size of a coordinate box
    eps = (size * w).simplify()
    eps.label = "ε"
    return eps
