"""tensorforms: exact calculus of tensors and tensor-valued forms on manifolds.

    manifold/    Manifold (coordinates), its tangent Space, metric, connection, curvature;
                 ready-made manifolds (Manifold.sphere(), …) and metric templates
    tensor/      Index, Tensor: everything you do to a tensor is a method on it (X.tr, X.nabla,
                 X.lie(v), A @ B, A.otimes(B), …); products.py and derivatives.py hold the code
    symbolic/    simplification helpers
"""
from .manifold import Manifold, Space, templates
from .tensor import Index, Tensor

__all__ = ["Manifold", "Index", "Space", "Tensor", "templates"]
__version__ = "0.1.0"
