"""The metric: a symmetric tensor with two down value slots."""
import itertools

import numpy as np
import sympy as sp

from ..symbolic.simplify import is_zero
from ..tensor.tensor import Tensor


class Metric(Tensor):
    """A metric (or its inverse): a tensor that also keeps its sympy matrix and sign pattern.

    `signature_string` is e.g. "-+++" (or None if unknown); `Tensor.signature` is the slot types.
    """

    def __init__(self, manifold, matrix, slots, label, signature_string=None):
        super().__init__(manifold, np.array(matrix.tolist(), dtype=object), slots, label)
        self.matrix = matrix
        self.signature_string = signature_string


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


def make_metric(M, data=None, signature=None):
    """data given → that metric. No data → unknown functions g_ij(x) with the given signature."""
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
        g = sp.Matrix(data)
        if g.shape != (n, n):
            raise ValueError(f"metric must be {n}×{n} on this manifold, got {g.shape}")
        for a, b in itertools.combinations(range(n), 2):
            if not is_zero(g[a, b] - g[b, a]):
                raise ValueError("a metric must be symmetric")
        found = _infer_signature(g)
        if signature and found and signature != found:
            raise ValueError(f"signature {signature!r} doesn't match the data ({found!r})")
        signature = signature or found
    return Metric(M, g, [i.down, j.down], label="g", signature_string=signature)


def inverse_metric(g):
    M = g.manifold
    i, j = M.indices(2, names=("μ", "ν"))
    return Metric(M, sp.simplify(g.matrix.inv()), [i.up, j.up], label="g⁻¹")
