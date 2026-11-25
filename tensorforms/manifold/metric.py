"""Building the metric: a plain tensor (μ↓ ν↓) that the manifold uses.

Manifold(..., metric=...) calls build_metric to make and check it, and keeps it with its
inverse g⁻¹ (the tensor inverse). Everything here is done with tensors.
"""
import numpy as np
import sympy as sp

from ..tensor.tensor import Tensor
from ..errors import InvalidMetricError


def _infer_signature(G):
    """Signs of a diagonal metric tensor, if sympy can tell them; otherwise None."""
    n = G.data.shape[0]
    if any(sp.sympify(G[a, b]) != 0 for a in range(n) for b in range(n) if a != b):
        return None
    signs = []
    for a in range(n):
        e = sp.sympify(G[a, a])
        if e.is_positive:
            signs.append("+")
        elif e.is_negative:
            signs.append("-")
        else:
            return None
    return "".join(signs)


def build_metric(M, data, signature=None):
    """Make and check a metric tensor g (μ↓ ν↓). Returns (g, signature).

    data: nested lists, a matrix, or a tensor with two down value indices. With unknown
    functions nothing about the signature can be checked; it is only what you declare."""
    i, j = M.indices(2, names=("μ", "ν"))
    n = M.dim
    if signature is not None and len(signature) != n:
        raise InvalidMetricError(f"metric: signature {signature!r} has {len(signature)} signs but the manifold has {n} coordinates")
    if isinstance(data, Tensor):
        if len(data.indices) != 2 or any(s.is_up or s.is_form for s in data.indices):
            raise InvalidMetricError(f"metric: needs two down value indices; got {data.declaration()} "
                                     "(g⁻¹ has two up indices)")
        data = data.data
    arr = np.asarray(data, dtype=object)
    if arr.shape != (n, n):
        raise InvalidMetricError(f"metric: must be {n}×{n} on this manifold; got {arr.shape}")
    G = Tensor(M, arr, [i.down, j.down], label="g")
    if G.equals(G.T) is not True:                          # refused unless shown symmetric
        raise InvalidMetricError("metric: not shown symmetric (g = gᵀ)")
    found = _infer_signature(G)
    if signature and found and signature != found:
        raise InvalidMetricError(f"metric: signature {signature!r} doesn't match the data ({found!r})")
    return G, signature or found


def inverse_metric(g):
    """g⁻¹ (μ↑ ν↑): the tensor inverse of g. Refused if g is shown to have determinant 0."""
    try:
        G = g.inverse().simplify()
    except ValueError as e:
        raise InvalidMetricError("metric: not invertible (the determinant is 0)") from None
    G = G.relabel_indices({0: "μ", 1: "ν"})
    G.label = "g⁻¹"
    return G
