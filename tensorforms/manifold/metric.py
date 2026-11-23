"""Building the metric: a plain tensor (μ↓ ν↓) that the manifold uses.

M.metric(...) calls build_metric to make and check it, then attaches it.
"""
import itertools

import numpy as np
import sympy as sp

from ..tensor.tensor import Tensor


def as_matrix(X):
    """A two-slot tensor's numbers as a sympy Matrix."""
    return sp.Matrix(X.data.tolist())


def _infer_signature(m):
    """Signs of a diagonal metric, if sympy can tell them; otherwise None."""
    if not m.is_diagonal():
        return None
    signs = []
    for e in m.diagonal():
        if e.is_positive:
            signs.append("+")
        elif e.is_negative:
            signs.append("-")
        else:
            return None
    return "".join(signs)


def build_metric(M, data=None, signature=None):
    """Make and check a metric. Returns (g, signature).

    data: a matrix, nested lists, or a tensor with two down value slots.
    No data: unknown functions g_ij(x). Their signature is only what you declare;
    nothing about unknown functions can be checked."""
    i, j = M.indices(2, names=("μ", "ν"))
    n = M.dim
    if data is None:
        signature = signature or "+" * n
        if len(signature) != n:
            raise ValueError(f"signature {signature!r} has {len(signature)} signs but M has {n} coordinates")
        g = sp.zeros(n, n)
        for a, b in itertools.combinations_with_replacement(range(n), 2):
            f = sp.Function(f"g{a}{b}")(*M.coords)
            g[a, b] = g[b, a] = f
    else:
        if isinstance(data, Tensor):
            if len(data.slots) != 2 or any(s.is_up or s.is_form for s in data.slots):
                raise TypeError(f"a metric has two down value slots; got {data.declaration()} "
                                "(the inverse metric g⁻¹ has two up slots)")
            data = as_matrix(data)
        g = sp.Matrix(data)
        if g.shape != (n, n):
            raise ValueError(f"metric must be {n}×{n} on this manifold, got {g.shape}")
    G = Tensor(M, np.array(g.tolist(), dtype=object), [i.down, j.down], label="g")
    if data is not None:                                     # checks on given data, with the tensor itself
        if G.equals(G.T()) is not True:                      # refused unless shown symmetric
            raise ValueError("a metric must be symmetric (g = gᵀ is not shown)")
        if M.scalar(g.det()).is_zero() is True:              # refused when shown singular
            raise ValueError("a metric must be invertible, but this one has determinant 0")
        found = _infer_signature(g)
        if signature and found and signature != found:
            raise ValueError(f"signature {signature!r} doesn't match the data ({found!r})")
        signature = signature or found
    return G, signature


def inverse_metric(g):
    M = g.manifold
    i, j = M.indices(2, names=("μ", "ν"))
    inv = sp.simplify(as_matrix(g).inv())
    return Tensor(M, np.array(inv.tolist(), dtype=object), [i.up, j.up], label="g⁻¹")
