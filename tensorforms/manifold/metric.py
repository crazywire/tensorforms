"""Building the metric: a plain tensor (μ↓ ν↓) that the manifold uses.

Manifold(..., metric=...) calls build_metric to make and check it, and keeps it with its
inverse g⁻¹ (the tensor inverse). Everything here is done with tensors.
"""
import numpy as np

from ..tensor.tensor import Tensor
from ..errors import InvalidMetricError


def build_metric(M, data):
    """Make and check a metric tensor g (μ↓ ν↓).

    data: the components, as nested lists or a matrix. Not a tensor: tensors live on a
    manifold, and this one is still being made (another tensor's numbers are X.data)."""
    i, j = M.indices(2, names=("μ", "ν"))
    n = M.dim
    if isinstance(data, Tensor):
        raise TypeError("metric: give the components, not a tensor; use X.data for a tensor's numbers")
    arr = np.asarray(data, dtype=object)
    if arr.shape != (n, n):
        raise InvalidMetricError(f"metric: must be {n}×{n} on this manifold; got {arr.shape}")
    G = Tensor(M, arr, [i.down, j.down], label="g")
    if G.equals(G.T) is not True:                          # refused unless shown symmetric
        raise InvalidMetricError("metric: not shown symmetric (g = gᵀ)")
    return G


def inverse_metric(g):
    """g⁻¹ (μ↑ ν↑): the tensor inverse of g. Refused if g is shown to have determinant 0."""
    try:
        G = g.inverse().simplify()
    except ValueError:
        raise InvalidMetricError("metric: not invertible (the determinant is 0)") from None
    G = G.relabel_indices({0: "μ", 1: "ν"})
    G.label = "g⁻¹"
    return G
