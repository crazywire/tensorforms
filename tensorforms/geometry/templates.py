"""Metric shapes with unknown functions, for solving rather than using.

Each takes a bare Manifold and returns a metric tensor (not yet in use).
"""
import sympy as sp


def diagonal(M, prefix="f"):
    """diag(f0(x), f1(x), …): the simplest general metric."""
    fs = [sp.Function(f"{prefix}{i}")(*M.coords) for i in range(M.dim)]
    return M.metric(sp.diag(*fs))


def conformally_flat(M, signature=None):
    """e^{2φ(x)} × (the flat metric of the given signature)."""
    signature = signature or "+" * M.dim
    phi = sp.Function("phi")(*M.coords)
    signs = [1 if s == "+" else -1 for s in signature]
    return M.metric(sp.exp(2 * phi) * sp.diag(*signs), signature=signature)


def static_spherical(M):
    """−A(r) dt² + B(r) dr² + r² (dθ² + sin²θ dφ²). M must have coordinates (t, r, θ, φ).
    Setting the Einstein tensor to zero and solving for A, B gives Schwarzschild."""
    if M.dim != 4:
        raise ValueError("static_spherical needs coordinates (t, r, θ, φ)")
    t, r, th, ph = M.coords
    A, B = sp.Function("A")(r), sp.Function("B")(r)
    return M.metric(sp.diag(-A, B, r ** 2, r ** 2 * sp.sin(th) ** 2), signature="-+++")
